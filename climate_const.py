PATH = {
    "ground_truth_dir": "../experiments/eval_data/2024-11-04.json",
    "weakly_supervised": {
        "text": "../data/parsed_text2",
        "RAG_preprocessed": "./outputs_others/RAGpreprocessed",
        # "RAG_preprocessed": "../RAG/preprocessed_outputs",
    },
    "RAG": {
        # "vector_index": "../RAG/index/VS_Index_all_llmdefin",
        # "prev_retrieved": "../RAG/outputs/RETRIEVED.json",
        "vector_index": "./outputs_others/Index_all_llmdefin_NV-Embed-v2",
        "prev_retrieved": "./outputs_others/RETRIEVED.json",
    },
    "LLM": {
        # from initial_1-3
        # "examples": "../RAG/outputs/formatted_examples_w_retrieved_entities_v3.json",
        # from chat-gpt
        "examples": "./outputs_others/few_shot.json",
        "chunked_text": "../data/chuncked/600tokens",
    },
    # "GCMD": "../data/GCMD_04192024/w_addon.json",
    "GCMD": "./outputs_others/GCMD_12142024.json",
}
LABELS_DICT = {
    "entities": [
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
    ],
    "relations": [
        "ComparedTo",
        "Outputs",
        "RunBy",
        "ProvidedBy",
        "ValidatedBy",
        "UsedIn",
        "MeasuredAt",
        "MountedOn",
        "TargetsLocation",
    ],
    "label_mapper": {
        "Projects": "project",
        "Locations": "location",
        "MODELS": "model",
        "Experiments": "experiment",
        "Experiments Forcing": "experiment",
        "Experiments Scenario": "experiment",
        "Platforms": "platform",
        "Instruments": "instrument",
        "Providers": "provider",
        "Variables": "variable",
        "Measurement Name": "variable",
        "WEATHER EVENTS": "weather event",
        "NATURAL HAZARDS": "natural hazard",
        "TELECONNECTIONS": "teleconnection",
        "OCEAN CIRCULATION": "ocean circulation",
    },
}
text_template = "<heading>{}</heading>\n{}\n"