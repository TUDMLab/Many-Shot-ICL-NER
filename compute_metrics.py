import ast
import json
import random
import os
from collections import Counter
import pandas as pd
import collections
from utils import save_jsonl, load_jsonl, load_json, save_json, extract_entities, evaluate_sent, compute_f1, analyze_ner_errors, get_present_ner_error_types
from typing import List, Tuple, Dict, Counter as CounterType


model_name = "Qwen-7B-1M" # Llama3.1-8B-Qwen ,"deepseek-chat" or "Qwen2.5-72B", "Llama3.1-70B"
dataset_name = "MIT_Movie" #   "MIT_Restrant", "MIT_Movie", "MIT_Restaurant", "conll2003", "WNUT2017"
BM25 = True # True or False

if BM25:
    shot_list = [5, 10, 25, 50, 100, 200, 300, 400, 500]
    shot_list =[15, 20, 30, 35, 40, 45]
else:
    shot_list = [0, 5, 10, 25, 50, 100, 200, 300, 400, 500]

results = []
for i in shot_list:
    if BM25:
        data_path = f"./output/{dataset_name}/{model_name}_BM25_{i}.json"
    else:
        data_path = f"./output/{dataset_name}/{model_name}_{i}.json"
    data = load_json(data_path)
    counts = Counter()
    for sent in data:
        if "response" not in sent:
            continue
        sent['response'] = sent['response'].replace("Output: ", "")
        pred = extract_entities(sent['response'])
        ref = sent['entities']
        counts = evaluate_sent(ref, pred, counts)
    scores_ner = compute_f1(counts["ner_predicted"], counts["ner_gold"], counts["ner_matched"])
    print(f"======= Shot #: {i} =======")
    print(scores_ner)
    scores_ner["shot"] = i  # 添加 shot number
    results.append(scores_ner)

# 创建 DataFrame 并按 F1 倒序排序
df = pd.DataFrame(results)
df = df.sort_values(by="f1", ascending=True)

# 打印结果
print(df)

df.to_excel('./z-results.xlsx', index=False)  # 保存为 Excel 文件