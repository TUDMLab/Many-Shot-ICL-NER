import os, time
import random
from run_aug import LLM
from run_deepseek import PROMPT_MAP, gen_examples
from datasets import load_dataset
from dataloader import get_ners, data_process, ner_label_map, convert_ner_to_xml, feed_BM25, get_retrieved_examples
from tqdm import tqdm, trange
from utils import extract_entities, save_json_as_csv, load_json, get_ners, save_json

NER_TAGS_TO_IDX_MAPPING = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
NER_IDX_TO_TAGS_MAPPING = {0: 'O' , 1: 'B-PER', 2:'I-PER', 3:'B-ORG', 4:'I-ORG', 5:'B-LOC', 6:'I-LOC', 7:'B-MISC', 8:'I-MISC'}
ENTITY_MAPPING = {
    "PER": "person", 
    "ORG": "organization",
    "LOC": "location",
    "MISC": "miscellaneous"
}
GENIA_NER = ['O', 'B-DNA', 'I-DNA', 'B-RNA', 'I-RNA', 'B-cell line', 'I-cell line', 'B-cell type', 'I-cell type', 'B-protein', 'I-protein']

GENIA_ENTITY_MAPPING = {
    "DNA": "DNA",
    "RNA": "RNA",
    "cell line": "cell line",
    "cell type": "cell type",
    "protein": "protein"
}

MIT_RESTRANT_IDX_TO_TAG_MAPPING = {
    0: "O",
    1: "B-Rating",
    2: "I-Rating",
    3: "B-Amenity",
    4: "I-Amenity",
    5: "B-Location",
    6: "I-Location",
    7: "B-Restaurant_Name",
    8: "I-Restaurant_Name",
    9: "B-Price",
    10: "B-Hours",
    11: "I-Hours",
    12: "B-Dish",
    13: "I-Dish",
    14: "B-Cuisine",
    15: "I-Price",
    16: "I-Cuisine"
}

MIT_RESTRANT_ENTITY_MAPPING = {
    "Rating": "Rating", 
    "Amenity":"Amenity", 
    "Location": "Location", 
    "Restaurant_Name": "Restaurant_Name", 
    "Price": "Price", 
    "Hours": "Hours", 
    "Dish": "Dish", 
    "Cuisine": "Cuisine"
}

MIT_MOVIE_IDX_TO_TAG_MAPPING = {
    0: "O",
    1: "B-Actor",
    2: "I-Actor",
    3: "B-Plot",
    4: "I-Plot",
    5: "B-Opinion",
    6: "I-Opinion",
    7: "B-Award",
    8: "I-Award",
    9: "B-Year",
    10: "B-Genre",
    11: "B-Origin",
    12: "I-Origin",
    13: "B-Director",
    14: "I-Director",
    15: "I-Genre",
    16: "I-Year",
    17: "B-Soundtrack",
    18: "I-Soundtrack",
    19: "B-Relationship",
    20: "I-Relationship",
    21: "B-Character_Name",
    22: "I-Character_Name",
    23: "B-Quote",
    24: "I-Quote"
}

MIT_MOVIE_ENTITY_MAPPING = {
    "Actor": "Actor", 
    "Plot": "Plot", 
    "Opinion": "Opinion", 
    "Award": "Award", 
    "Year": "Year", 
    "Genre": "Genre", 
    "Origin": "Origin", 
    "Director":"Director", 
    "Soundtrack": "Soundtrack", 
    "Relationship": "Relationship", 
    "Character_Name": "Character_Name", 
    "Quote": "Quote"
}

WNUT2017_IDX_TO_TAG_MAPPING = {
    0: "B-corporation",
    1: "B-creative-work",
    2: "B-group",
    3: "B-location",
    4: "B-person",
    5: "B-product",
    6: "I-corporation",
    7: "I-creative-work",
    8: "I-group",
    9: "I-location",
    10: "I-person",
    11: "I-product",
    12: "O"
}

WNUT2017_ENTITY_MAPPING = {
    "corporation": "corporation",
    "creative-work": "creative-work",
    "group": "group",
    "location": "location", 
    "person": "person", 
    "product": "product"
}

FIN_IDX_TO_TAG_MAPPING = {
    0: "O",
    1: "B-PER",
    2: "B-LOC",
    3: "B-ORG",
    4: "B-MISC",
    5: "I-PER",
    6: "I-LOC",
    7: "I-ORG",
    8: "I-MISC"
}

FIN_ENTITY_MAPPING = {
    "ORG":"organization", 
    "LOC":"location", 
    "PER": "person", 
    "MISC": "miscellaneous"
}

def convert_entities(entities):
    new_entities = []
    for word, entity in entities:
        new_entities.append((word, ENTITY_MAPPING[entity]))
    return new_entities

def get_conll2003_data(subset: str, style: str = 'default'):
    conll_dataset = load_dataset("eriktks/conll2003", trust_remote_code=True)
    data = conll_dataset[subset]

    label_set = [label for label in ENTITY_MAPPING.values()]

    new_data = []
    for example in data:
        tokens = example["tokens"]
        ner_tags_ids = example["ner_tags"]
        ner_tags = []
        for ner_tag in ner_tags_ids:
            tag = NER_IDX_TO_TAGS_MAPPING[ner_tag]
            ner_tags.append(tag)
        
        # entities = get_ners(tokens, ner_tags)
        # entities = convert_entities(entities)
        # print("tokens", tokens)
        # print("ner_tags", ner_tags)
        # print("entities", entities)
        sentence, ners = data_process({'str_words': tokens, 'tags_ner': ner_tags})
        ners = ner_label_map(ENTITY_MAPPING, ners)
        # print(sentence, ners)

        if style == 'tagging':
            response = convert_ner_to_xml(tokens, ner_tags, ENTITY_MAPPING)
        else:
            response = ners

        new_data.append({'text': sentence, 'target': response, 'entities': ners, 'sent_id': example['id']})
        # print(new_data)
        # exit()
    return new_data, label_set

def get_GENIA_NER_data(subset: str, style: str = 'default'):
    genia_dataset = load_dataset("chufangao/GENIA-NER", trust_remote_code=True)
    data = genia_dataset[subset]

    label_set = [label for label in GENIA_ENTITY_MAPPING.values()]

    new_data = []
    for example in data:
        tokens = example["tokens"]
        ner_tags_ids = example["ner_tags"]
        ner_tags = []
        for ner_tag in ner_tags_ids:
            tag = GENIA_NER[ner_tag]
            ner_tags.append(tag)
        # print(tokens)
        # print(ner_tags)

        sentence, ners = data_process({'str_words': tokens, 'tags_ner': ner_tags})
        ners = ner_label_map(GENIA_ENTITY_MAPPING, ners)
        # print(sentence, ners)

        if style == 'tagging':
            response = convert_ner_to_xml(tokens, ner_tags, GENIA_ENTITY_MAPPING)
        else:
            response = ners

        new_data.append({'text': sentence, 'target': response, 'entities': ners, 'sent_id': example['id']})
        # print(new_data)
        # exit()
    return new_data, label_set

def get_MIT_Restrant_data(subset: str, style: str = 'default'):
    mit_dataset = load_dataset("tner/mit_restaurant", trust_remote_code=True)
    data = mit_dataset[subset]

    label_set = [label for label in MIT_RESTRANT_ENTITY_MAPPING.values()]

    new_data = []
    for _id, example in enumerate(data):
        tokens = example["tokens"]
        ner_tags_ids = example["tags"]
        ner_tags = []
        for ner_tag in ner_tags_ids:
            tag = MIT_RESTRANT_IDX_TO_TAG_MAPPING[ner_tag]
            ner_tags.append(tag)
        # print(tokens)
        # print(ner_tags)

        sentence, ners = data_process({'str_words': tokens, 'tags_ner': ner_tags})
        ners = ner_label_map(MIT_RESTRANT_ENTITY_MAPPING, ners)
        # print(sentence, ners)

        if style == 'tagging':
            response = convert_ner_to_xml(tokens, ner_tags, MIT_RESTRANT_ENTITY_MAPPING)
        else:
            response = ners

        new_data.append({'text': sentence, 'target': response, 'entities': ners, 'sent_id': _id})
        # print(new_data)
        # exit()
    return new_data, label_set

def get_MIT_Movie_data(subset: str, style: str = 'default'):
    mit_dataset = load_dataset("tner/mit_movie_trivia", trust_remote_code=True)
    data = mit_dataset[subset]

    label_set = [label for label in MIT_MOVIE_ENTITY_MAPPING.values()]

    new_data = []
    for _id, example in enumerate(data):
        tokens = example["tokens"]
        ner_tags_ids = example["tags"]
        ner_tags = []
        for ner_tag in ner_tags_ids:
            tag = MIT_MOVIE_IDX_TO_TAG_MAPPING[ner_tag]
            ner_tags.append(tag)
        # print(tokens)
        # print(ner_tags)

        sentence, ners = data_process({'str_words': tokens, 'tags_ner': ner_tags})
        ners = ner_label_map(MIT_MOVIE_ENTITY_MAPPING, ners)
        # print(sentence, ners)

        if style == 'tagging':
            response = convert_ner_to_xml(tokens, ner_tags, MIT_MOVIE_ENTITY_MAPPING)
        else:
            response = ners

        new_data.append({'text': sentence, 'target': response, 'entities': ners, 'sent_id': _id})
        # print(new_data)
        # exit()
    return new_data, label_set

def get_WNUT2017_data(subset: str, style: str = 'default'):
    tner_dataset = load_dataset("tner/wnut2017", trust_remote_code=True)
    data = tner_dataset[subset]

    label_set = [label for label in WNUT2017_ENTITY_MAPPING.values()]

    new_data = []
    num = 0
    for _id, example in enumerate(data):
        tokens = example["tokens"]
        if len(tokens) > 10000:
            num += 1
            continue
        ner_tags_ids = example["tags"]
        ner_tags = []
        for ner_tag in ner_tags_ids:
            tag = WNUT2017_IDX_TO_TAG_MAPPING[ner_tag]
            ner_tags.append(tag)
        # print(tokens)
        # print(ner_tags)

        sentence, ners = data_process({'str_words': tokens, 'tags_ner': ner_tags})
        ners = ner_label_map(WNUT2017_ENTITY_MAPPING, ners)
        # print(sentence, ners)

        if style == 'tagging':
            response = convert_ner_to_xml(tokens, ner_tags, WNUT2017_ENTITY_MAPPING)
        else:
            response = ners

        new_data.append({'text': sentence, 'target': response, 'entities': ners, 'sent_id': _id})
        # print(new_data)
        # exit()
    print(f"{num} data exceed 10000 tokens")
    return new_data, label_set

def get_FIN_data(subset: str, style: str = 'default'):
    tner_dataset = load_dataset("tner/fin", trust_remote_code=True)
    data = tner_dataset[subset]

    label_set = [label for label in FIN_ENTITY_MAPPING.values()]

    new_data = []
    for _id, example in enumerate(data):
        tokens = example["tokens"]
        ner_tags_ids = example["tags"]
        ner_tags = []
        for ner_tag in ner_tags_ids:
            tag = FIN_IDX_TO_TAG_MAPPING[ner_tag]
            ner_tags.append(tag)
        # print(tokens)
        # print(ner_tags)

        sentence, ners = data_process({'str_words': tokens, 'tags_ner': ner_tags})
        ners = ner_label_map(FIN_ENTITY_MAPPING, ners)
        # print(sentence, ners)

        if style == 'tagging':
            response = convert_ner_to_xml(tokens, ner_tags, FIN_ENTITY_MAPPING)
        else:
            response = ners

        new_data.append({'text': sentence, 'target': response, 'entities': ners, 'sent_id': _id})
        # print(new_data)
        # exit()
    return new_data, label_set

def run_few_shot_CoNLL2003(model: str, demo_nums: int = 100, style: str = 'default'):

    # 获取数据
    train_data, labels = get_conll2003_data(subset='train', style='tagging')
    if not os.path.exists("datasets/conll2003/conll2003_train.json"):
        save_json(train_data, "datasets/conll2003/conll2003_train.json")
    if len(train_data) < demo_nums:
        dev_data, _ = get_conll2003_data(subset='validation', style='tagging') # 额外的dev数据
        if not os.path.exists("datasets/conll2003/conll2003_validation.json"):
            save_json(train_data, "datasets/conll2003/conll2003_validation.json")
        train_data.extend(dev_data)
        if len(train_data) < demo_nums:
            return
    test_data, _ = get_conll2003_data(subset='test', style='tagging')
    if not os.path.exists("datasets/conll2003/conll2003_test.json"):
        save_json(test_data, "datasets/conll2003/conll2003_test.json")
    prompt_format = PROMPT_MAP[style]

    # 生成示例
    if style == 'zero_shot_tagging':
        examples = ""
    else:
        examples = gen_examples(train_data, demo_nums)

    # 如果输出目录不存在，则创建
    output_dir = f'./output/conll2003'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}_{demo_nums}.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        test_data = load_json(output_file)
    elif os.path.exists("datasets/conll2003/test_samples_200.json"):
        print(f"Loading existing test samples: datasets/conll2003/test_samples_200.json")
        test_data = load_json("datasets/conll2003/test_samples_200.json")
    else:
        print("Randomly select 300 samples from the test set")
        test_data = load_json("datasets/conll2003/conll2003_test.json")
        # 只保留300条数据
        test_data = random.sample(test_data, 200)
        save_json(test_data, "datasets/conll2003/test_samples_200.json")

    llm = LLM(model=model, system_prompt="", max_tokens=2048)

    for i in trange(len(test_data)):
        # check if the test data has been processed
        if 'response' in test_data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = test_data[i]['text']
        # 生成prompt
        if style == 'zero_shot_tagging':
            prompt = prompt_format.format(entity_types=labels, input_text=input_text)
        else:
            prompt = prompt_format.format(entity_types=labels, examples=examples, input_text=input_text)

        # 最多重试三次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response, prob = llm.call_with_logits(prompt=prompt)
                test_data[i]['response'] = response
                test_data[i]['probability'] = prob
                break
            except Exception as e:
                print(f"API 调用失败，重试次数: {attempt+1}/{max_retries}，错误信息: {e}")
                # 可根据需要添加 sleep 或其它逻辑
                time.sleep(1)
        # 直接调用
        # response = llm_call(prompt=prompt)
        # test_data[i]['response'] = response

        # 关键：在每次调用完 llm_call 后就立刻保存
        save_json(test_data, output_file)
    save_json(test_data, output_file)

def run_few_shot(model: str, dataset:str="conll2003", data_func=get_conll2003_data, demo_nums: int = 100, style: str = 'default'):

    # 获取数据
    train_data, labels = data_func(subset='train', style='tagging')
    os.makedirs(f"datasets/{dataset}", exist_ok=True)
    if not os.path.exists(f"datasets/{dataset}/{dataset}_train.json"):
        save_json(train_data, f"datasets/{dataset}/{dataset}_train.json")
    if len(train_data) < demo_nums:
        dev_data, _ = data_func(subset='validation', style='tagging') # 额外的dev数据
        if not os.path.exists(f"datasets/{dataset}/{dataset}_validation.json"):
            save_json(train_data, f"datasets/{dataset}/{dataset}_validation.json")
        train_data.extend(dev_data)
        if len(train_data) < demo_nums:
            return
    test_data, _ = data_func(subset='test', style='tagging')
    if not os.path.exists(f"datasets/{dataset}/{dataset}_test.json"):
        save_json(test_data, f"datasets/{dataset}/{dataset}_test.json")
    prompt_format = PROMPT_MAP[style]

    # 生成示例
    if style == 'zero_shot_tagging':
        examples = ""
    else:
        examples = gen_examples(train_data, demo_nums)
    # 如果输出目录不存在，则创建
    output_dir = f'./output/{dataset}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}_{demo_nums}.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        test_data = load_json(output_file)
    elif os.path.exists(f"datasets/{dataset}/test_samples_200.json"):
        print(f"Loading existing test samples: datasets/{dataset}/test_samples_200.json")
        test_data = load_json(f"datasets/{dataset}/test_samples_200.json")
    else:
        print("Randomly select 200 samples from the test set")
        test_data = load_json(f"datasets/{dataset}/{dataset}_test.json")
        # 只保留300条数据
        test_data = random.sample(test_data, 200)
        save_json(test_data, f"datasets/{dataset}/test_samples_200.json")

    llm = LLM(model=model, system_prompt="", max_tokens=2048)
    print(f"Initializing LLM {model} successfully!!!")

    for i in trange(len(test_data)):
        # check if the test data has been processed
        if 'response' in test_data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = test_data[i]['text']
        # 生成prompt
        if demo_nums == 0 and style == 'zero_shot_tagging':
            prompt = prompt_format.format(entity_types=labels, input_text=input_text)
        else:
            prompt = prompt_format.format(entity_types=labels, examples=examples, input_text=input_text)
        # 最多重试三次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response, prob = llm.call_with_logits(prompt=prompt)
                test_data[i]['response'] = response
                test_data[i]['probability'] = prob
                break
            except Exception as e:
                print(f"API 调用失败({demo_nums} shots on {dataset})，重试次数: {attempt+1}/{max_retries}，错误信息: {e}")
                # 可根据需要添加 sleep 或其它逻辑
                
                time.sleep(1)
                
        # 直接调用
        # response = llm_call(prompt=prompt)
        # test_data[i]['response'] = response

        # 关键：在每次调用完 llm_call 后就立刻保存
        save_json(test_data, output_file)
    save_json(test_data, output_file)

def run_few_shot_CoNLL2003_BM25(model: str, demo_nums: int = 100, style: str = 'default'):

    # 获取数据
    train_data, labels = get_conll2003_data(subset='train', style='tagging')
    if not os.path.exists("datasets/conll2003/conll2003_train.json"):
        save_json(train_data, "datasets/conll2003/conll2003_train.json")
    if len(train_data) < demo_nums:
        dev_data, _ = get_conll2003_data(subset='validation', style='tagging') # 额外的dev数据
        if not os.path.exists("datasets/conll2003/conll2003_validation.json"):
            save_json(train_data, "datasets/conll2003/conll2003_validation.json")
        train_data.extend(dev_data)
        if len(train_data) < demo_nums:
            return
    test_data, _ = get_conll2003_data(subset='test', style='tagging')
    if not os.path.exists("datasets/conll2003/conll2003_test.json"):
        save_json(test_data, "datasets/conll2003/conll2003_test.json")
    prompt_format = PROMPT_MAP[style]
    retriever = feed_BM25(train_data, "conll2003")

    # 生成示例
    # if style == 'zero_shot_tagging':
    #     examples = ""
    # else:
    #     examples = gen_examples(train_data, demo_nums)

    # 如果输出目录不存在，则创建
    output_dir = f'./output/conll2003'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}_BM25_{demo_nums}.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        test_data = load_json(output_file)

    llm = LLM(model=model, system_prompt="")

    # only test 300 samples, randomly selected 300
    if not os.path.exists("datasets/conll2003/test_samples_200.json"):
        test_data = random.sample(test_data, 200)
        save_json(test_data, "datasets/conll2003/test_samples_200.json")
    else:
        test_data = load_json("datasets/conll2003/test_samples_200.json")
    # test all the test data
    for i in trange(len(test_data)):
        # check if the test data has been processed
        if 'response' in test_data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = test_data[i]['text']

        # retrieve examples according to relevance
        topN_examples = get_retrieved_examples(retriever=retriever, query=input_text, dataset=train_data, topN=demo_nums)

        examples = gen_examples(topN_examples, len(topN_examples)) # all top N samples

        # 生成prompt
        if style == 'zero_shot_tagging':
            prompt = prompt_format.format(entity_types=labels, input_text=input_text)
        else:
            prompt = prompt_format.format(entity_types=labels, examples=examples, input_text=input_text)
        
        # 最多重试三次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response, prob = llm.call_with_logits(prompt=prompt)
                test_data[i]['response'] = response
                test_data[i]['probability'] = prob
                break
            except Exception as e:
                print(f"API 调用失败，重试次数: {attempt+1}/{max_retries}，错误信息: {e}")
                # 可根据需要添加 sleep 或其它逻辑
                time.sleep(1)
        # 直接调用
        # response = llm_call(prompt=prompt)
        # test_data[i]['response'] = response

        # 关键：在每次调用完 llm_call 后就立刻保存
        save_json(test_data, output_file)
    save_json(test_data, output_file)

def run_few_shot_BM25(model: str, dataset:str="conll2003", data_func=get_conll2003_data, demo_nums: int = 100, style: str = 'default'):

    # 获取数据
    train_data, labels = data_func(subset='train', style='tagging')
    if not os.path.exists(f"datasets/{dataset}/{dataset}_train.json"):
        save_json(train_data, f"datasets/{dataset}/{dataset}_train.json")
    if len(train_data) < demo_nums:
        dev_data, _ = data_func(subset='validation', style='tagging') # 额外的dev数据
        if not os.path.exists(f"datasets/{dataset}/{dataset}_validation.json"):
            save_json(train_data, f"datasets/{dataset}/{dataset}_validation.json")
        train_data.extend(dev_data)
        if len(train_data) < demo_nums:
            return
    test_data, _ = data_func(subset='test', style='tagging')
    if not os.path.exists(f"datasets/{dataset}/{dataset}_test.json"):
        save_json(test_data, f"datasets/{dataset}/{dataset}_test.json")
    prompt_format = PROMPT_MAP[style]
    retriever = feed_BM25(train_data, f"{dataset}")

    # 生成示例
    # if style == 'zero_shot_tagging':
    #     examples = ""
    # else:
    #     examples = gen_examples(train_data, demo_nums)

    # 如果输出目录不存在，则创建
    output_dir = f'./output/{dataset}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}_BM25_{demo_nums}.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        test_data = load_json(output_file)
    elif os.path.exists(f"datasets/{dataset}/test_samples_200.json"):
        print(f"Loading existing test samples: datasets/{dataset}/test_samples_200.json")
        test_data = load_json(f"datasets/{dataset}/test_samples_200.json")
    else:
        print("Randomly select 200 samples from the test set")
        test_data = load_json(f"datasets/{dataset}/{dataset}_test.json")
        # 只保留300条数据
        test_data = random.sample(test_data, 200)
        save_json(test_data, f"datasets/{dataset}/test_samples_200.json")

    llm = LLM(model=model, system_prompt="", max_tokens=2048)
    print(f"Initializing LLM {model} successfully!!!")
    for i in trange(len(test_data)):
        # check if the test data has been processed
        if 'response' in test_data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = test_data[i]['text']

        # retrieve examples according to relevance
        topN_examples = get_retrieved_examples(retriever=retriever, query=input_text, dataset=train_data, topN=demo_nums)

        examples = gen_examples(topN_examples, len(topN_examples)) # all top N samples

        # 生成prompt
        if style == 'zero_shot_tagging':
            prompt = prompt_format.format(entity_types=labels, input_text=input_text)
        else:
            prompt = prompt_format.format(entity_types=labels, examples=examples, input_text=input_text)
        
        # 最多重试三次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response, prob = llm.call_with_logits(prompt=prompt)
                test_data[i]['response'] = response
                test_data[i]['probability'] = prob
                break
            except Exception as e:
                print(f"API 调用失败(BM25 {demo_nums} shots on {dataset}), 重试次数: {attempt+1}/{max_retries}，错误信息: {e}")
                # 可根据需要添加 sleep 或其它逻辑
                # print(prompt)
                # for example in topN_examples:
                #     print(len(example['text']), len(example['target']))
                # input()
                time.sleep(1)
        # 直接调用
        # response = llm_call(prompt=prompt)
        # test_data[i]['response'] = response

        # 关键：在每次调用完 llm_call 后就立刻保存
        save_json(test_data, output_file)
    save_json(test_data, output_file)


if __name__ == '__main__':
    # get_GENIA_NER_data(subset='train', style='tagging')
    # get_conll2003_data(subset='train', style='tagging')
    # get_MIT_Restrant_data(subset='train', style='tagging')
    # get_MIT_Movie_data(subset='train', style='tagging')
    # get_WNUT2017_data(subset='train', style='tagging')
    # get_FIN_data(subset='train', style='tagging')
    # shots = [5, 0, 500, 400, 300, 200, 100, 50, 25, 10]
    # shots = [400, 200, 50, 5]
    
    # funcs = [get_WNUT2017_data]
    # funcs = [get_WNUT2017_data, get_MIT_Movie_data, get_MIT_Restrant_data, get_conll2003_data]
    # datasets = ['WNUT2017', 'MIT_Movie', 'MIT_Restrant', 'conll2003']
    funcs = [get_WNUT2017_data]
    datasets = ['WNUT2017']
    model = "Qwen2.5-7B" # "Qwen2.5-72B", "deepseek-chat", "Llama3.1-70B"
    # shots = [25, 50]
    shots = [0, 5, 10, 25, 50, 75, 100, 200, 250, 275, 300, 350]
    for _id, func in enumerate(funcs):
        for shot in shots:
            dataset = datasets[_id]
            print(f"Run {shot} on {dataset}...")
            style = 'tagging'
            if shot == 0:
                style = 'zero_shot_tagging'
            
            print(f"===>Using {style} style")
            run_few_shot(model=model, dataset=dataset, data_func=func, demo_nums=shot, style=style)
            
            print(f"Run {shot} BM25 on {dataset}...")
            if shot == 0:
                continue
            run_few_shot_BM25(model=model, dataset=dataset, data_func=func, demo_nums=shot, style=style)