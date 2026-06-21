import os
import json
import random
import torch
import torch.nn as nn
from torch.optim import AdamW 
from torch.utils.data import Dataset, DataLoader
from torch.utils.data._utils.collate import default_collate
from transformers import AutoTokenizer, AutoModelForTokenClassification
from tqdm import tqdm
from typing import List
from run_GliNER import read_data
from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS
from seqeval.metrics import precision_score, recall_score, f1_score


LABEL_DIC = {
    'ai': AI_CLASSS,
    'science': SCIENCE_CLASS,
    'literature': LITERATURE_CLASSS,
    'music': MUSIC_CLASSS,
    'politics': POLITICS_CLASSS,
}


def sample_data(data_list, n, seed=42):
    random.seed(seed)
    return random.sample(data_list, n)

def save_checkpoint(model, tokenizer, output_dir, epoch):
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"======Saved checkpoint to {model_path}")

def ner_collate_fn(batch):
    str_words = [item.pop("str_words") for item in batch]
    batch = default_collate(batch)
    batch["str_words"] = str_words  # reattach as list
    return batch

class NERDataset(Dataset):
    def __init__(self, data: List[dict], tokenizer, tag2id, max_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        words = item['str_words']
        labels = item['tags_ner']

        encodings = self.tokenizer(
            words,
            is_split_into_words=True,
            return_offsets_mapping=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        word_ids = encodings.word_ids()
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)
            else:
                label_ids.append(tag2id[labels[word_id]])

        encodings = {k: v.squeeze() for k, v in encodings.items()}
        encodings["labels"] = torch.tensor(label_ids)
        encodings["str_words"] = words  # Add words back for alignment
        return encodings

def train(model_id, data, tokenizer, tag2id, num_epochs, output_dir):
    dataset = NERDataset(data, tokenizer, tag2id)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=ner_collate_fn)
    model = AutoModelForTokenClassification.from_pretrained(
        model_id,
        num_labels=len(tag2id),
        ignore_mismatched_sizes=True  #  Fix the shape mismatch error
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    # num_epochs = 10
    # loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for batch in tqdm(loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
        print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")
        save_checkpoint(model, tokenizer, output_dir, epoch + 1)

def inference(valid_data, tokenizer, tag2id, id2tag, checkpoint_path="./ner_bert_checkpoints/"):
    valid_dataset = NERDataset(valid_data, tokenizer, tag2id)
    valid_loader = DataLoader(valid_dataset, batch_size=1, collate_fn=ner_collate_fn)
    model = AutoModelForTokenClassification.from_pretrained(checkpoint_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    model.eval()
    all_true_tags = []
    all_predicted_tags = []
    
    with torch.no_grad():
        for batch in valid_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)  
            str_words = batch['str_words']  # This is a list, not tensor

            # Run model
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1).squeeze().tolist()
            labels = labels.squeeze().tolist()
            str_words = str_words[0]  # revoking to list since collate_fn makes str_words(list) a list of lists after batching

            # Re-tokenize to get word_ids for alignment
            tokenized = tokenizer(
                str_words,  # batch size = 1
                is_split_into_words=True,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=128
            )
            word_ids = tokenized.word_ids()

            # Map back to word-level tags
            predicted_tags = []
            true_tags = []
            previous_word_idx = None
            for i, word_idx in enumerate(word_ids):
                if word_idx is None or word_idx == previous_word_idx:
                    continue
                predicted_tags.append(id2tag[predictions[i]])
                true_tags.append(id2tag[labels[i]] if labels[i] != -100 else "IGN")
                previous_word_idx = word_idx

            # Store predictions and true tags for metrics
            all_true_tags.extend(true_tags)
            all_predicted_tags.extend(predicted_tags)

            # print("=== Sample ===")
            # for word, true_tag, pred_tag in zip(str_words, true_tags, predicted_tags):
            #     print(f"{word:15} | True: {true_tag:8} | Pred: {pred_tag}")
            # print()

    # Compute metrics
    
    # Convert to list of lists for seqeval
    all_true_tags = [[tag] for tag in all_true_tags]
    all_predicted_tags = [[tag] for tag in all_predicted_tags]
    
    precision = precision_score(all_true_tags, all_predicted_tags)
    recall = recall_score(all_true_tags, all_predicted_tags)
    f1 = f1_score(all_true_tags, all_predicted_tags)
    
    # print("\n=== Evaluation Metrics ===")
    # print(f"Precision: {precision:.4f}")
    # print(f"Recall: {recall:.4f}")
    # print(f"F1 Score: {f1:.4f}")
    
    return precision, recall, f1

if __name__ == '__main__':
    dataset = 'ai'  # 'ai', 'science', 'literature', 'music', 'politics'
    ann_model = "deepseek-chat" # "Qwen2.5-72B", "deepseek-chat"
    model_id = "google-bert/bert-base-cased"
    num_epochs = 15
    output_dir = "./ner_bert_checkpoints"
    sample_num = 2000

    aug_train_path = "output/augment_ablation/ai/deepseek-chat-200-shots/batch-2000.json"
    # aug_train_path = f"output/augment/ai/Qwen2.5-72B/batch-2000.json"
    aug_train_data = json.load(open(aug_train_path, 'r'))
    # aug_train_data = read_data(aug_train_path)
    label_map = LABEL_DIC[dataset]
    reverse_map = {v: k for k, v in label_map.items()}
    print("label_map: ", reverse_map)
    new_aug_train_data = []
    for item in aug_train_data:
        # replace the label with the reverse_map
        # tags_ner中是BIO tag, 需要提取-后面的label, replace 原始的-后面的tag
        if 'tags_ner' in item:
            for i in range(len(item['tags_ner'])):
                if item['tags_ner'][i].startswith('B-'):
                    item['tags_ner'][i] = 'B-' + reverse_map[item['tags_ner'][i].split('-')[-1]]
                elif item['tags_ner'][i].startswith('I-'):
                    item['tags_ner'][i] = 'I-' + reverse_map[item['tags_ner'][i].split('-')[-1]]
            new_aug_train_data.append(item)
    print("new_aug_train_data: ", len(new_aug_train_data))

    seed_path = f'./datasets/aug/{dataset}/seed.json'
    # add_train = read_data(seed_path)
    add_train = json.load(open(seed_path, 'r'))
    # new_aug_train_data = []
    train_data = new_aug_train_data + add_train
    # train_data = add_train # only use seed data
    print("Data Size: ", len(train_data))

    # load test data
    test_path = f'./datasets/crossner/{dataset}_test.json'
    test_data = json.load(open(test_path, 'r'))
    print("Test Data Size: ", len(test_data))

    labels_set = set()
    for d in train_data:
        try:
            labels_set.update(set(d['tags_ner']))
        except:
            print(d)
            exit()
    print("labels_set: ", labels_set)
    # Ensure consistent ordering, e.g., alphabetical or with 'O' first
    sorted_labels = sorted(labels_set, key=lambda x: (x != "O", x))

    # Create mappings
    tag2id = {label: idx for idx, label in enumerate(sorted_labels)}
    id2tag = {idx: label for label, idx in tag2id.items()}
    print(len(sorted_labels))
    print("tag2id", tag2id)
    print("id2tag", id2tag)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    train(model_id, train_data, tokenizer, tag2id, num_epochs=num_epochs, output_dir=output_dir)

    print("=== Inference ===")
    print("Augmented data size: ", sample_num)
    for epoch in range(1, num_epochs+1):
        checkpoint_path = f"./ner_bert_checkpoints/checkpoint-epoch-{epoch}"
        precision, recall, f1 = inference(test_data, tokenizer, tag2id, id2tag, checkpoint_path=checkpoint_path)
        print(f"Epoch {epoch} - Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
