import json
import os
from tqdm import tqdm, trange
from gliner import GLiNER

def read_json_file(json_path):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)
    return data

def save_json_file(data, json_path):
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

def get_ent_labels(chunk):
    ent_labels = []
    for ent in chunk["entities"]:
        ent_labels.append(ent["label"])
    return ent_labels
CLIMATE_LABELS = [
        "project",
        "location",
        "model",
        "experiment",
        "platform",
        "instrument",
        "provider",
        "variable",
        "weather event",
        "natural hazard",
        "teleconnection",
        "ocean circulation",
    ]

def run_GliNER():
    # load the data
    data_path = './datasets/climate/'
    doc_files = os.listdir('./datasets/climate/')
    output_path = './output/baselines/gliner/'
    # load model
    model = GLiNER.from_pretrained("urchade/gliner_base")
    labels = CLIMATE_LABELS
    for file in tqdm(doc_files):
        data = read_json_file(f'./datasets/climate/{file}')
        doc_res = {}
        doc_res['doc_key'] = file
        # iterate all chunks
        gt = []
        pred = []
        for chunk in data:
            text = chunk['text']
            chunk_labels = chunk['entities']
            # get the entities
            chunk_gt = [(ent['substring'], ent['label']) for ent in chunk_labels]
            # run the model
            entities = model.predict_entities(text, labels)
            chunk_pred = [(ent['text'], ent['label']) for ent in entities]
            # append the results
            gt.append(chunk_gt)
            pred.append(chunk_pred)
        doc_res['pred'] = pred
        doc_res['gt'] = gt
        # save the results
        save_json_file(doc_res, f'{output_path}GliNER_{file}')

def merge_entities(text, entities):
    if not entities:
        return []
    merged = []
    current = entities[0]
    for next_entity in entities[1:]:
        if next_entity['label'] == current['label'] and (next_entity['start'] == current['end'] + 1 or next_entity['start'] == current['end']):
            current['text'] = text[current['start']: next_entity['end']].strip()
            current['end'] = next_entity['end']
        else:
            merged.append(current)
            current = next_entity
    # Append the last entity
    merged.append(current)
    return merged

def run_nuNER():
    doc_files = os.listdir('./datasets/climate/')
    output_path = './output/baselines/nuNER/'
    # load model
    model = GLiNER.from_pretrained("numind/NuNerZero")
    labels = CLIMATE_LABELS
    for file in tqdm(doc_files):
        data = read_json_file(f'./datasets/climate/{file}')
        doc_res = {}
        doc_res['doc_key'] = file
        # iterate all chunks
        gt = []
        pred = []
        for chunk in data:
            text = chunk['text']
            chunk_labels = chunk['entities']
            # get the entities
            chunk_gt = [(ent['substring'], ent['label']) for ent in chunk_labels]
            # run the model
            entities = model.predict_entities(text, labels)
            entities = merge_entities(text, entities)
            chunk_pred = [(ent['text'], ent['label']) for ent in entities]
            # append the results
            gt.append(chunk_gt)
            pred.append(chunk_pred)
        doc_res['pred'] = pred
        doc_res['gt'] = gt
        # save the results
        save_json_file(doc_res, f'{output_path}{file}')


if __name__ == '__main__':
    # run_GliNER()
    run_nuNER()