from openai import OpenAI
from typing import List, Dict, Any
from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS
from const import AI_PROMPTS, SCIENCE_PROMPTS, LITERATURE_PROMPTS, MUSIC_PROMPTS, POLITICS_PROMPTS
from utils import extract_entities, save_json_as_csv, load_json, get_ners, save_json
from utils import evaluate_sent, compute_f1
from tqdm import trange, tqdm
import argparse
from collections import Counter

PROMPT_MAPPING = {
    "ai": AI_PROMPTS,
    "science": SCIENCE_PROMPTS,
    "literature": LITERATURE_PROMPTS,
    "music": MUSIC_PROMPTS,
    "politics": POLITICS_PROMPTS
}

LABEL_MAPPING = {
    "ai": AI_CLASSS,
    "science": SCIENCE_CLASS,
    "literature": LITERATURE_CLASSS,
    "music": MUSIC_CLASSS,
    "politics": POLITICS_CLASSS
}

def llm_call(prompt: str, system_prompt: str = "", model="deepseek-chat") -> str:
    """
    Calls the model with the given prompt and returns the response.

    Args:
        prompt (str): The user prompt to send to the model.
        system_prompt (str, optional): The system prompt to send to the model. Defaults to "".
        model (str, optional): The model to use for the call. Defaults to "claude-3-5-sonnet-20241022".

    Returns:
        str: The response from the language model.
    """
    client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com") # TODO: Change the API key
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        stream=False
    )
    return response.choices[0].message.content

def chain(input: str, prompts: List[str]) -> str:
    """Chain multiple LLM calls sequentially, passing results between steps."""
    response = input
    result = []
    result.append(response)
    for i, prompt in enumerate(prompts, 1):
        # print(response)
        response = llm_call(f"{prompt}\nInput: {response}")
        result.append(response)
    return result


if __name__ == "__main__":
    # using command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='literature')
    
    args = parser.parse_args()

    dataset = args.dataset # ai, science, literature, music, politics
    data = load_json(f"/home/tuo96248/projects/NER-DAug/datasets/crossner/{dataset}_test.json")
    zero_shot_data_processing_steps = PROMPT_MAPPING[dataset]

    for i in trange(len(data)):
        str_words = data[i]['str_words']
        str_tags = data[i]['tags_ner']
        ner_labels = get_ners(str_words, str_tags)
        input_sent = " ".join(str_words)
        data[i]['ner_labels'] = ner_labels
        # ---- LLM predict ----
        try:
            output = chain(input_sent, zero_shot_data_processing_steps)
        except:
            output = [["No"]]
            print("Error in LLM predict")
        data[i]['llm_output'] = output
        ner_predicts = extract_entities(output[-1])
        data[i]['ner_predicts'] = ner_predicts
    # save the results
    save_json(data, f"/home/tuo96248/projects/NER-DAug/output/workflow/{dataset}_test.json")
    # compute the performance
    counts = Counter()
    data = load_json(f"/home/tuo96248/projects/NER-DAug/output/workflow/{dataset}_test.json")
    mapping = LABEL_MAPPING[dataset]
    for i in range(len(data)):
        ner_labels = data[i]['ner_labels']
        ner_predicts = data[i]['ner_predicts']
        # update ner_labels
        for idx in range(len(ner_labels)):
            ner_labels[idx][1] = mapping[ner_labels[idx][1]]
        counts = evaluate_sent(ner_labels, ner_predicts, counts)
    scores_ner = compute_f1(
                counts["ner_predicted"], counts["ner_gold"], counts["ner_matched"])
    print(scores_ner)