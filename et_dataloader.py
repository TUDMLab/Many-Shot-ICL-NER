import logging
from abc import ABC
from typing import Dict, Optional
import re
from utils import save_jsonl, load_jsonl, load_json, save_json
import pandas as pd
import json
from datasets import load_dataset

from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS

import pdb

LABEL_DIC = {
    'ai': AI_CLASSS,
    'science': SCIENCE_CLASS,
    'literature': LITERATURE_CLASSS,
    'music': MUSIC_CLASSS,
    'politics': POLITICS_CLASSS,
}

LLM_TEXT = 'text'
LLM_NER_TEXT = 'ner_text'
PROMPTS = 'prompts'
SPLIT_TOKEN = "=="
TEXT_BETWEEN_SHOTS = f"\n{SPLIT_TOKEN}\n"
LABEL_TOKENS = 'ner_text'

_logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

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


class CrossNER(ABC):
    name = "ai"
    dataset = "./datasets/"
    x_prefix = "Sentence: "
    ent_prefix = "Entity: "
    y_prefix = "Type: "
    mode = "train" # train or eval

    def __init__(self):
        super().__init__()
        self.label_map = LABEL_DIC[self.name]
        if self.mode == "train":
            train_df = self._load_dataset()
            _logger.info(f"loaded {len(train_df)} training samples")
        else:
            raise ValueError(f"Invalid mode: {self.mode}")

        # format the dataset for LLMs
        self.train_df = self.apply_format(train_df)
    
    def apply_format(self, df, test=False):
        """
        - generate the x_text and y_text
        - generate the prompt column
        """
        # generate the x_text and y_text
        # df = self.generate_x_text(df)
        # df = self.generate_y_text(df)
        if test:
            df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x['text']}\n{self.y_prefix}".rstrip(), axis=1)
        else:
                df[PROMPTS] = df.apply(lambda x: f"{self.x_prefix}{x['text']}\n{self.ent_prefix}{x['entity']}\n{self.y_prefix}{x['label']}",
                                   axis=1)
        return df
    
    def extract_ents(self, data_path):
        sents = load_json(data_path)
        new_data = []
        idx = 0
        for sample in sents:
            sentence, ners = data_process(sample)
            ners = ner_label_map(self.label_map, ners)
            for ner in ners:
                new_data.append({'id': idx,'text': sentence, 'entity': ner[0], 'label': ner[1]})
                idx += 1
        return new_data

    def _load_dataset(self):
        train_data = self.extract_ents(f"{self.dataset}/crossner/{self.name}_train.json")

        # convert to dataframe
        train_df = pd.DataFrame(train_data)
        return train_df


class AI(CrossNER):
    name = "ai"
    mode = "train"

class LITERATURE(CrossNER):
    name = "literature"
    mode = "train"

class MUSIC(CrossNER):
    name = "music"
    mode = "train"

class POLITICS(CrossNER):
    name = "politics"
    mode = "train"

class SCIENCE(CrossNER):
    name = "science"
    mode = "train"

DATASET_NAME2LOADERS = {
    # "conll2003": CONLL2003,
    "ai": AI,
    'literature': LITERATURE,
    'music': MUSIC,
    'politics': POLITICS,
    'science': SCIENCE
}

def get_demo_loader(dataset_name,):
    if dataset_name in DATASET_NAME2LOADERS:
        return DATASET_NAME2LOADERS[dataset_name]()
    else:
        raise ValueError(f"Unknown {dataset_name} !!!!")


if __name__ == '__main__':
    data = SCIENCE()
    print(data.train_df.head())
    # print the first row of the dataframe
    print(data.train_df.iloc[0]['prompts'])
    print(data.train_df.shape)
    print(data.train_df.columns)
    # save it to a jsonl file
    data.train_df.to_json("./datasets/demos/science_train.jsonl", orient='records', lines=True)