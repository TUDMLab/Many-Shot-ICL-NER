import json
import os
from utils import txt_to_json_list, load_json, compute_entity_tag_probs
from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS
import re
import random

LABEL_DIC = {
    'ai': AI_CLASSS,
    'science': SCIENCE_CLASS,
    'literature': LITERATURE_CLASSS,
    'music': MUSIC_CLASSS,
    'politics': POLITICS_CLASSS,
}

def xml_ner_to_bio(text, label_list=None):
    """
    Convert XML-tagged NER text into whitespace-tokenized tokens and BIO tags.
    
    Args:
        text (str): Input string containing <entity type="TYPE">...</entity> annotations.
    
    Returns:
        tokens (List[str]): Whitespace-tokenized tokens.
        tags (List[str]): Corresponding BIO tags.
    """
    # Regex to find entity annotations
    pattern = re.compile(r'<entity\s+type="([^"]+)">(.*?)</entity>')
    
    tokens = []
    tags = []
    last_end = 0
    
    # Iterate over all entity matches
    for match in pattern.finditer(text):
        # Text before the current entity
        before = text[last_end:match.start()]
        for t in before.split():
            tokens.append(t)
            tags.append("O")
        
        # Entity text and type
        ent_type = match.group(1)
        ent_text = match.group(2)
        for i, t in enumerate(ent_text.split()):
            prefix = "B-" if i == 0 else "I-"
            tokens.append(t)
            # if ent_type in label_list:
            if label_list is not None and ent_type in label_list:
                tags.append(f"{prefix}{ent_type}")
            else:
                # If the entity type is not in the label list, use "O"
                tags.append("O")
            # tags.append(f"{prefix}{ent_type}")
        
        last_end = match.end()
    
    # Text after the last entity
    after = text[last_end:]
    for t in after.split():
        tokens.append(t)
        tags.append("O")
    
    return tokens, tags


dataset = 'science'  # 'ai', 'science', 'literature', 'music', 'politics'
ann_model = 'deepseek-chat'  # 'Qwen2.5-72B', 'gemini', 'deepseek-chat'
labels = LABEL_DIC[dataset]
label_list = list(labels.values())

batch_list = [4]
for total_batch_num in batch_list:
    data = []
    for i in range(total_batch_num):
        # load data
        print(f"loading batch-{i} data")
        if ann_model == 'gemini':
            sub_data = load_json(f"output/augment/{dataset}/{ann_model}/batch-{i}.json")
        else:
            # sub_data = load_json(f"output/augment/{dataset}/{ann_model}/batch-{i}-logits.json")
            # output/augment_ablation/ai/deepseek-chat-1-shots
            sub_data = load_json(f"output/augment_ablation/{dataset}/{ann_model}-1-shots/batch-{i}.json")
            # sub_data = load_json(f"output/augment/{dataset}/{ann_model}/batch-{i}.json")
        # add data to total_data
        data.extend(sub_data)

    length = len(data)
    print(f"total data length: {length}")
    new_data = []
    for i in range(len(data)):
        # remove "Output: "
        if 'response' not in data[i]:
            continue
        else:
            data[i]['response'] = data[i]['response'].replace("Output: ", "")
            tokens, labels = xml_ner_to_bio(data[i]['response'], label_list)
            data[i]['str_words'] = tokens
            data[i]['tags_ner'] = labels
            new_data.append(data[i])
    # print(len(new_data))
    # # save to file
    with open(f"output/augment/{dataset}/{ann_model}/batch-{length}.json", "w") as f:
        json.dump(new_data, f, ensure_ascii=False, indent=4)
    print(f"output/augment/{dataset}/{ann_model}/batch-{length}.json")

    # randomly sample 1600, 1700, 1800 , 1900 and then save
    sample_nums = [1200, 1300, 1400, 1500, 1600, 1700]
    for sample_num in sample_nums:
        # randomly sample 1600, 1700, 1800 , 1900
        random.shuffle(new_data)
        sampled_data = new_data[:sample_num]
        # save to file
        with open(f"output/augment/{dataset}/{ann_model}/batch-{sample_num}.json", "w") as f:
            json.dump(sampled_data, f, ensure_ascii=False, indent=4)
        print(f"output/augment/{dataset}/{ann_model}/batch-{sample_num}.json")