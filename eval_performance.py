import json
import os
from collections import Counter

from utils import save_jsonl, load_jsonl, load_json, save_json
from utils import save_jsonl, load_jsonl, load_json, save_json, extract_entities, evaluate_sent, compute_f1
from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS

LABEL_DIC = {
    'ai': AI_CLASSS,
    'science': SCIENCE_CLASS,
    'literature': LITERATURE_CLASSS,
    'music': MUSIC_CLASSS,
    'politics': POLITICS_CLASSS,
}

def get_pred(str_words, span, response):
    ent_str = str_words[span[0]:span[1]]
    ent_str = ' '.join(ent_str)
    ent_type = response
    return ent_str, ent_type

def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    # F1 score.
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return dict(precision=precision, recall=recall, f1=f1)


def evaluate_sent(gt_ner, pred_ner,counts):
    # correct_ner = set()
    # Entities.
    counts["ner_gold"] += len(gt_ner)
    counts["ner_predicted"] += len(pred_ner)
    for prediction in pred_ner:
        if any([prediction == actual for actual in gt_ner]):
            counts["ner_matched"] += 1
            # correct_ner.add(prediction[0])
    return counts

# args
name = 'science'
model = 'llama_8b'

#--------------------------------
label_map = LABEL_DIC[name]
label_list = list(label_map.values())

predict_data = load_json(f'./output/{name}/{model}_et_results.json')
refer_data = load_json(f'./datasets/MD/{name}_test.json')

# process the reference data
ref_dic = {}
for ref in refer_data:
    sentence, ners = data_process(ref)
    ners = ner_label_map(label_map, ners)
    ref_dic[ref['id']] = ners

# process the prediction data
pred_dic = {}
for pred in predict_data:
    ent_str, ent_type = get_pred(pred['str_words'], pred['span'], pred['response'])
    sent_id = pred['sent_id']
    if sent_id not in pred_dic:
        pred_dic[sent_id] = []
    pred_dic[sent_id].append([ent_str, ent_type])

print(len(ref_dic), len(pred_dic))
# assert len(ref_dic) == len(pred_dic)

# counts = Counter()
# for key, value in ref_dic.items():
#     ref_ner = value
#     if key not in pred_dic:
#         pred_ner = []
#     else:
#         pred_ner = pred_dic[key]
#     counts = evaluate_sent(ref_ner, pred_ner, counts)

# scores_ner = compute_f1(
#             counts["ner_predicted"], counts["ner_gold"], counts["ner_matched"])
# print(scores_ner)

data = load_json("./output/politics/deepseek-chat_100.json")
counts = Counter()
for sent in data:
    if "response" not in sent:
        continue
    pred = extract_entities(sent['response'])
    ref = sent['entities']
    counts = evaluate_sent(ref, pred, counts)
scores_ner = compute_f1(counts["ner_predicted"], counts["ner_gold"], counts["ner_matched"])
print(scores_ner)