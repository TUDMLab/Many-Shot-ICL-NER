import pickle
import os
import json
import re
import ast
from utils import save_jsonl, load_jsonl, load_json, save_json
from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS
import re
from retriv import SparseRetriever
from collections import defaultdict
import random

LABEL_DIC = {
    'ai': AI_CLASSS,
    'science': SCIENCE_CLASS,
    'literature': LITERATURE_CLASSS,
    'music': MUSIC_CLASSS,
    'politics': POLITICS_CLASSS,
}

def convert_ner_to_xml(words: list, tags: list, label_map: dict) -> str:
    output = []
    i = 0
    n = len(tags)
    while i < n:
        if tags[i].startswith('B-'):
            entity_type = tags[i][2:]
            j = i + 1
            while j < n and tags[j] == f'I-{entity_type}':
                j += 1
            entity_words = words[i:j]
            entity_str = ' '.join(entity_words)
            output.append(f'<entity type="{label_map[entity_type]}">{entity_str}</entity>')
            i = j
        else:
            output.append(words[i])
            i += 1
    return ' '.join(output)

def get_ners(str_words, tags_ner):
    entities = []
    entity = []
    entity_type = None

    for word, tag in zip(str_words, tags_ner):
        if tag.startswith('B-'):
            if entity:  # if there is an ongoing entity, save it
                entities.append([" ".join(entity), entity_type])
                entity = []
            entity_type = tag[2:]  # start a new entity
            entity.append(word)
        elif tag.startswith('I-') and entity:
            entity.append(word)
        else:
            if entity:  # if there is an ongoing entity, save it
                entities.append([" ".join(entity), entity_type])
                entity = []
            entity_type = None

    # Catch any entity left at the end
    if entity:
        entities.append([" ".join(entity), entity_type])

    return entities

def data_process(sample):
    sentence = ' '.join(sample['str_words'])
    ners = get_ners(sample['str_words'], sample['tags_ner'])

    return sentence, ners

def ner_label_map(label_map, ners):
    for ner in ners:
        ner[1] = label_map[ner[1]]
    return ners

def get_label_data(dataset: str, subset: str, style: str = 'default'):
    label_map = LABEL_DIC[dataset] # mapping the label to the specific domain for LLMs
    label_set = [label for label in label_map.values()]
    sents = load_json(f'./datasets/aug/{dataset}/unlabel.json')

    new_data = []
    for idx, sample in enumerate(sents):
        sentence, ners = data_process(sample)
        ners = ner_label_map(label_map, ners)
        if style == 'tagging':
            response = convert_ner_to_xml(sample['str_words'], sample['tags_ner'], label_map)
        else:
            response = ners
        new_data.append({'text': sentence, 'target': response,'entities': ners})
    return new_data, label_set

def get_seed_data(dataset: str, subset: str, style: str = 'default'):
    label_map = LABEL_DIC[dataset] # mapping the label to the specific domain for LLMs
    label_set = [label for label in label_map.values()]
    sents = load_json(f'./datasets/aug/{dataset}/{subset}.json')

    return sents, label_set

def get_crossNER_data(dataset: str, subset: str, style: str = 'default'):
    label_map = LABEL_DIC[dataset] # mapping the label to the specific domain for LLMs
    label_set = [label for label in label_map.values()]
    sents = load_json(f'./datasets/crossner/{dataset}_{subset}.json')

    new_data = []
    for idx, sample in enumerate(sents):
        sentence, ners = data_process(sample)
        ners = ner_label_map(label_map, ners)
        # for ner in ners:
        #     new_data.append({'text': sentence, 'entity': ner[0], 'label': ner[1]})
        if style == 'tagging':
            response = convert_ner_to_xml(sample['str_words'], sample['tags_ner'], label_map)
        else:
            response = ners
        new_data.append({'text': sentence, 'target': response,'ners': ners, 'sent_id': sample['sent_id']})
    return new_data, label_set

def feed_BM25(data, dataset, idx_col = 'text'):
    """
    dataset: return by get_crossNER_data()
    {
        'text': 'Further , in the case of estimation based on a single sample , it demonstrates philosophical issues and possible misunderstandings in the use of maximum likelihood estimators and likelihood functions .', 
        'target': 'Further , in the case of estimation based on a single sample , it demonstrates philosophical issues and possible misunderstandings in the use of <entity type="metrics">maximum likelihood estimators and likelihood functions</entity> .', 
        'entities': [['maximum likelihood estimators and likelihood functions', 'metrics']]}
    """
    feed_corpus = []
    for _id, row in enumerate(data):
        text = row[idx_col]
        feed_corpus.append({"id": str(_id), "text": text})

    # Initialize retriever with training corpus
    # generate the index_name with the dataset name and some random characters
    index_name = f"training-examples-{dataset}-{random.randint(1000, 9999)}"
    retriever = SparseRetriever(
        index_name=index_name,
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
    retriever.index(feed_corpus)  # Corpus = training sentences
    return retriever

def get_retrieved_examples(retriever, query:str, dataset:list, topN:int=5):
    """
    dataset: return by get_xxx_data()
    """
    # Retrieve scores for the query
    results = retriever.search(query=query, return_docs=False, cutoff=topN)  # get all scores
    new_data = []
    topN_doc_ids = []
    for doc_id, score in results.items():
        doc_id = int(doc_id)
        # print(f"{rank + 1}. doc_id: {doc_id}, Score: {score:.4f}, Sentence: {processed_train_sentences[doc_id]}")
        sample = dataset[doc_id]
        new_data.append(sample)
        topN_doc_ids.append(doc_id)

    # if len(new_data) < topN, randomly select additional samples from the dataset. Do not repeat the samples in new_data
    if len(new_data) < topN:
        print(f"Only {len(new_data)} samples retrieved, randomly selecting additional samples to reach {topN}.")
        additional_samples = random.sample([sample for i, sample in enumerate(dataset) if i not in topN_doc_ids], topN - len(new_data))
        new_data.extend(additional_samples)
    # shuffle the new_data
    random.shuffle(new_data)
    return new_data


# args
dataset = 'ai'
# data, label_set = get_topN_RM25_score_from_crossNER_data(domain=dataset, topN=10, style='tagging')
#--------------------------------

# label_map = LABEL_DIC[dataset] # mapping the label to the specific domain for LLMs
# label_set = [label for label in label_map.values()]
# sents = load_json(f'/home/tuo96248/projects/LLM-CrossNER/dataset/crossner/{dataset}_train.json')

# new_data = []
# for sample in sents:
#     sentence, ners = data_process(sample)
#     ners = ner_label_map(label_map, ners)
#     # for ner in ners:
#     #     new_data.append({'text': sentence, 'entity': ner[0], 'label': ner[1]})
#     new_data.append({'text': sentence, 'entities': ners})
# print(new_data[0])

# new_data, labels = get_crossNER_data(dataset='ai', subset='train')
# print(new_data[1])
# print(labels)
