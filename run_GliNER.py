import json
import os
import random
import torch
import numpy as np
import argparse
from tqdm import tqdm
from collections import Counter
from utils import save_jsonl, load_jsonl, load_json, save_json, extract_entities, evaluate_sent, compute_f1

from gliner import GLiNERConfig, GLiNER
from gliner.training import Trainer, TrainingArguments
from gliner.data_processing.collator import DataCollatorWithPadding, DataCollator
from gliner.utils import load_config_as_namespace
from gliner.data_processing import WordsSplitter, GLiNERDataset
from typing import List, Tuple, Dict
from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS

import pdb

os.environ["TOKENIZERS_PARALLELISM"] = "true" # This is required to avoid warning from transformers
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # This is required to avoid warning from transformers
# typing



LABEL_DIC = {
    'ai': AI_CLASSS,
    'science': SCIENCE_CLASS,
    'literature': LITERATURE_CLASSS,
    'music': MUSIC_CLASSS,
    'politics': POLITICS_CLASSS,
}



# ====== Helper functions ======
def get_labels(data: List[Dict[List, List]]) -> List[str]:
    """
    Get the unique labels from the data.
    Args:
        data: The data in GliNER format.
    Returns:
        A list of unique labels.
    """
    labels = set()
    for item in data:
        for span in item["ner"]:
            labels.add(span[2])
    return list(labels)


def get_gts(tokenized_text, ner):
    gts = []
    for span in ner:
        start, end, label = span
        tokens = tokenized_text[start:end+1]
        text = " ".join(tokens)
        gts.append([text, label])
    return gts

def bio_to_spans(tags: List[str]) -> List[List[int]]:
    """
    Convert a sequence of BIO tags to a list of spans.
    Args:
        tags: A list of BIO tags.
    Returns:
        A list of spans, where each span is a list [start, end, type].
    """
    spans = []
    i = 0
    while i < len(tags):
        # Check for beginning of an entity.
        if tags[i].startswith("B-"):
            entity_type = tags[i][2:]
            start = i
            end = i
            i += 1
            # Continue for subsequent I-tags of the same type.
            while i < len(tags) and tags[i] == f"I-{entity_type}":
                end = i
                i += 1
            spans.append([start, end, entity_type])
        else:
            i += 1
    return spans

def read_data(data_path: str) -> List[Dict[List, List]]:
    """
    Read the training and test data from a file and convert it to GliNER format.
    Args:
        data_path: The path to the data file.
    Returns:
        A list of dictionaries, where each dictionary has the keys "tokenized_text" and "ner".
    """
    with open(data_path, "r") as f:
        data = json.load(f)
    new_data = []
    for item in data:
        # 如果 "tags_ner" 为空，则跳过
        if "tags_ner" not in item:
            continue
        ner = bio_to_spans(item["tags_ner"])
        new_data.append(
            {
                "tokenized_text": item["str_words"],
                "ner": ner,
            }
        )
    return new_data

def read_aug_data(data_path: str, label_map: Dict[str, str]) -> List[Dict[List, List]]:
    """
    Change the label not in label_map to misellaneous
    """
    label_names = list(label_map.keys())
    data = load_json(data_path)
    new_data = []
    for item in data:
        # 如果 "tags_ner" 为空，则跳过
        if "tags_ner" not in item:
            continue
        ner = bio_to_spans(item["tags_ner"])
        # 如果标签不在label_names中，则直接将标签改成misellaneous
        # for i in range(len(ner)):
        #     if ner[i][2] not in label_names:
        #         ner[i][2] = 'misellaneous'
        new_data.append(
            {
                "tokenized_text": item["str_words"],
                "ner": ner,
            }
        )
    return new_data

def compute_performance(output_path):
    data = load_json(output_path)
    counts = Counter()
    for sent in data:
        pred = sent['preds']
        ref = sent['gts']
        counts = evaluate_sent(ref, pred, counts)
    scores_ner = compute_f1(counts["ner_predicted"], counts["ner_gold"], counts["ner_matched"])
    print("NER Scores: ", scores_ner)


# ====== Main functions ======

def train(train_dataset: List[Dict[List, List]], test_dataset: List[Dict[List, List]], ckpts_path: str, output_dir: str):
    # set up device and model
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    model = GLiNER.from_pretrained("gliner-community/gliner_large-v2.5") # Load the base model
    print("Model Loaded")
    data_collator = DataCollator(model.config, data_processor=model.data_processor, prepare_labels = True)
    model.to(device)
    print("Model Loaded")

    # Fix training epochs to ensure similar training epochs
    batch_size = 16
    num_epochs = 30
    print(f"Number of epochs: {num_epochs}")

    training_args = TrainingArguments(
        output_dir= ckpts_path , # The trained model output directory
        learning_rate=5e-6,
        weight_decay=0.01,
        others_lr=1e-5,
        others_weight_decay=0.01,
        lr_scheduler_type="linear", #cosine
        warmup_ratio=0.1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        focal_loss_alpha=0.75,
        focal_loss_gamma=2,
        num_train_epochs=num_epochs,
        eval_strategy="epoch",
        save_steps = 50,
        save_total_limit=200,
        dataloader_num_workers = 0,
        use_cpu = False,
        report_to="none",
        )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=model.data_processor.transformer_tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    print("Training done")

    # Explicitly save the final model state
    trainer.save_model(ckpts_path)  # Ensures the final state is saved
    print(f"Final model saved to {ckpts_path}")

def evaluate(test_data, ckpts_path, output_dir):
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    labels = get_labels(test_data)
    print("Test Data Size: ", len(test_data))
    print("Labels: ", labels)

    # Update ckpts_path 
    # ckpts_path = os.path.join(ckpts_path, "checkpoint-10")
    print("Loading model from: ", ckpts_path)
    model = GLiNER.from_pretrained(ckpts_path, load_tokenizer=True)
    model.to(device)
    print("Model Loaded")
    # test the data one by one
    for item in tqdm(test_data):
        tokenized_text = item["tokenized_text"]
        ner = item["ner"]
        text = " ".join(tokenized_text)
        # Predict the NER spans.
        pred_ner = model.predict_entities(text, labels)
        # print("Predicted NER: ", pred_ner)
        # append to the test_data
        item["pred_ner"] = pred_ner
        item["gts"] = get_gts(tokenized_text, ner)
        item["preds"] = [[pred['text'], pred['label']] for pred in pred_ner]
    # save the test_data
    output_path = f"{output_dir}/test.json"
    with open(output_path, "w") as f:
        json.dump(test_data, f, indent=2)
    print("Test data saved to: ", output_path)
    # compute the F1 score
    compute_performance(output_path)


if __name__ == "__main__":
    # set up the argument parser
    parser = argparse.ArgumentParser(description="Train a NER model")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the training and test data")
    # parser.add_argument("--output_dir", type=str, help="Path to the output directory")
    parser.add_argument("--ckpts", type=str, help="Path to save checkpoints", default="./ckpts")
    parser.add_argument("--model", type=str, help="Used model for training", default="gliner")
    parser.add_argument("--shots", type=int, help="Number of labeled data used for training", default=100)

    # parse the arguments
    args = parser.parse_args()
    shots = args.shots

    # shots_list = [1500]
    # shots_list = [500]
    # for shots in shots_list:
    # print(f"====> Processing shots: {shots}")
    shots = "ai-2500"
    ckpts_path = os.path.join(args.ckpts, str(args.model), f"{args.dataset}_{shots}")
    os.makedirs(ckpts_path, exist_ok=True)
    output_dir = os.path.join("./output", str(args.model), f"{args.dataset}_{shots}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Dataset: {args.dataset}")
    print(f"Output directory: {output_dir}")
    print(f"Checkpoints directory: {ckpts_path}")
    # read the training and test data
    # train_path = f'./datasets/crossner/{args.dataset}_train.json'
    # train_data = read_data(train_path)
    # dev_path = f'./datasets/crossner/{args.dataset}_dev.json'
    # dev_data = read_data(dev_path)
    # # combine train and dev data
    # train_data = train_data + dev_data
    # train_data = read_data(f"./output/augment/{args.dataset}/deepseek-chat/batch-2000.json")
    label_map = LABEL_DIC[args.dataset]
    # train_data = read_aug_data(f"output/augment/{args.dataset}/deepseek-chat/batch-2000.json", label_map)
    # print(f"loaded training data: {args.dataset}/deepseek-chat/batch-2000.json")
    # 处理训练赛数据，把标签改成和test data一致的
    
    reverse_map = {v: k for k, v in label_map.items()}
    print("label_map: ", reverse_map)
    
    # for item in train_data:
    #     item["ner"] = [[item["ner"][i][0], item["ner"][i][1], reverse_map[item["ner"][i][2]]] for i in range(len(item["ner"]))]
        # try:
        #     item["ner"] = [[item["ner"][i][0], item["ner"][i][1], reverse_map[item["ner"][i][2]]] for i in range(len(item["ner"]))]
        # except:
        #     # print error
        #     print("="*20)
        #     print("error: ", item)
    # train_path = f'./datasets/crossner/{args.dataset}_train.json'
    train_path = f'./datasets/aug/{args.dataset}/seed.json'
    add_train = read_data(train_path)
    train_data = add_train
    print("Data Size: ", len(train_data))
    # shuffle the data
    # random.seed(42)
    # randomly select the number of shots for training
    # if shots > len(train_data):
    #     shots = len(train_data)
    # train_data = random.sample(train_data, shots)
    test_path = f'./datasets/crossner/{args.dataset}_test.json'
    test_data = read_data(test_path)
    # print(test_data[0])
    # pdb.set_trace()

    # train the model
    train(train_data, test_data, ckpts_path, output_dir)
    
    # evaluate the model
    eval_steps = [50, 100, 150, 200]
    for step in eval_steps:
        ckpts_path = f"./ckpts/gliner/{args.dataset}_{shots}/checkpoint-{step}"
        evaluate(test_data, ckpts_path, output_dir)
        print(f"====> Done processing shots: {shots}")