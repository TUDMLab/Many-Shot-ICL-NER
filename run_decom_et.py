import logging
from abc import ABC
from typing import Dict, Optional, List
import re
import pandas as pd
import json
from datasets import load_dataset
from retriv import SparseRetriever
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm, trange
import datetime
import torch
import os
import argparse

from et_dataloader import get_demo_loader
from utils import save_jsonl, load_jsonl, load_json, save_json
from utils import get_ners, data_process, ner_label_map

from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS
from templates import AI_ET_INSTRUCT, LITERATURE_ET_INSTRUCT, SCIENCE_ET_INSTRUCT, MUSIC_ET_INSTRUCT, POLITICS_ET_INSTRUCT


_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

MODEL_PATH = {
    'llama_8b': '/data/LLMs/Llama-3.1-8B-Instruct',
    # 'llama_8b_instruct': '/data/LLMs/Meta-Llama-3.1-8B-Instruct',
    'llama_70b': '/data/LLMs/Llama-3.1-70B-Instruct',
    'qwen_7b': '/data/LLMs/Qwen2.5-7B-Instruct',
    'qwen_14b': '/data/LLMs/Qwen2.5-14B-Instruct',
    'qwen_32b': '/data/LLMs/Qwen2.5-32B-Instruct',
    'qwen_72b': '/data/LLMs/Qwen2.5-72B-Instruct',
}

LABEL_DIC = {
    'ai': AI_CLASSS,
    'science': SCIENCE_CLASS,
    'literature': LITERATURE_CLASSS,
    'music': MUSIC_CLASSS,
    'politics': POLITICS_CLASSS,
}

INSTRUCT = {
    # 'conll2003': CONLL2003_BIO_INSTRUCT,
    'ai': AI_ET_INSTRUCT,
    'literature': LITERATURE_ET_INSTRUCT,
    'music': MUSIC_ET_INSTRUCT,
    'politics': POLITICS_ET_INSTRUCT,
    'science': SCIENCE_ET_INSTRUCT,
}

TEXT_BETWEEN_SHOTS = "\n\n"

def create_retriever(demo_file, dataset_name):
    sr = SparseRetriever(
        index_name=f"{dataset_name}-index",
        model="bm25",
        min_df=1,
        tokenizer="whitespace",
        stemmer="english",
        stopwords="english",
        do_lowercasing=True,
        do_ampersand_normalization=True,
        do_special_chars_normalization=True,
        do_acronyms_normalization=True,
        do_punctuation_removal=True,
    )
    sr.index_file(path=demo_file, 
        show_progress=True,  
        callback=lambda doc: {      # Callback defaults to None.
            "id": doc["id"],
            "text": doc["prompts"]},          
    )
    return sr

def error_save(data, out_dir):
    # 保存结果
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    out_dir = os.path.join(out_dir, f"error_{timestamp}.json")
    # save to json
    save_json(data, out_dir)

def get_ent(str_words, span):
    return ' '.join(str_words[span[0]:span[1]])

def build_test_prompt(sent_str, ent):
    return f"Sentence: {sent_str}\nEntity: {ent}\nType:"

def build_few_shot_prompt(selected: List):
    return TEXT_BETWEEN_SHOTS.join(selected)

def build_prompt(few_shots_prompt, test_prompt, template):
    prompt = template + '\n\nExamples:\n\n' + few_shots_prompt + '\n\nNow, please finish the test sample:\n\n' + test_prompt
    return prompt

def build_msg(prompt):
    msgs = [{"role": "system", "content": "You are a helpful assistant and are good at entity typing task."},
        {"role": "user", "content": prompt}]
    return msgs

def run_llm(model, tokenizer, model_inputs):
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=1024
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]
    # Decode the generated ids
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response




# print(INSTRUCT['music'])
# exit()

# name = 'literature'
# shots = 50
# model = 'qwen_7b' # 'qwen_32b'

parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default='literature')
parser.add_argument("--shots", type=int, default=10)
parser.add_argument("--model", type=str, default='qwen_7b')
args = parser.parse_args()

output_dir = './output'
eval_data_path = f"./datasets/MD/{args.name}_test.json"
eval_data = load_json(eval_data_path)
# training data for demos
demo_path = f"./datasets/demos/{args.name}_train.jsonl"
sr = create_retriever(demo_path, f'{args.name}')
template = INSTRUCT[args.name]

# update output_dir
output_dir = os.path.join(output_dir, args.name)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# set up the model and tokenizer
model_name = args.model # 用来后续保存结果
model_dir = MODEL_PATH[model_name]
model = AutoModelForCausalLM.from_pretrained(
model_dir,
attn_implementation="flash_attention_2", 
torch_dtype= torch.bfloat16,
device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_dir)


results = []
for sent in tqdm(eval_data):
    sent_id = sent['id']
    str_words = sent['str_words']
    sent_str = ' '.join(str_words)
    tags_ner_pred = sent['tags_ner_pred']

    for span in tags_ner_pred:
        ent_str = get_ent(str_words, span)
        test_prompt = build_test_prompt(sent_str, ent_str)
        # print(test_prompt)
        retrieved = sr.search(
        query=test_prompt, 
        cutoff=args.shots,
        )
        selected = [r['text'] for r in retrieved]
        # reverse the order
        selected = selected[::-1]
        few_shot_prompt = build_few_shot_prompt(selected)
        prompt = build_prompt(few_shot_prompt, test_prompt, template)
        msgs = build_msg(prompt)
        # tokenize the prompt
        text = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True
            )
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        # 2. 调用模型
        try:
            response = run_llm(model, tokenizer, model_inputs)
        except:
            # error, save the current test_df
            span_res = {
                'sent_id': sent_id,
                'str_words': str_words,
                'span': span,
                'response': ''
            }
            results.append(span_res)
            error_save(results, output_dir)
        
        span_res = {
            'sent_id': sent_id,
            'str_words': str_words,
            'span': span,
            'response': response
        }
        results.append(span_res)

        # debug
        print(response)
    # break
# save the results
output_file = os.path.join(output_dir, f"{model_name}_et_results.json")
save_json(results, output_file)