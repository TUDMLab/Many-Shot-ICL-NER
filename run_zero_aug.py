"""
LLMs as Data Annotators
SLMs/LLMs as Data Quality Evaluators
"""

from openai import OpenAI
from typing import List, Dict, Any
from tqdm import tqdm, trange
import os, time
import random


# self-defined
from dataloader import get_crossNER_data
from utils import extract_entities, save_json_as_csv, load_json, get_ners, save_json
from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS
from prompts import ONE_STAGE_FEW_SHOT_PROMPTS, ONE_STAGE_TAG_FEW_SHOT_PROMPTS, ZERO_SHOT_TAG_PROMPT


LABEL_DIC = {
    'ai': AI_CLASSS,
    'science': SCIENCE_CLASS,
    'literature': LITERATURE_CLASSS,
    'music': MUSIC_CLASSS,
    'politics': POLITICS_CLASSS,
}


def gen_examples(data, shots=100):
    # randomly select shots examples
    random.shuffle(data)
    data = data[:shots] # 
    examples = []
    for item in data:
        output = item["response"]
        # remove the "Output:  " prefix
        if output.startswith("Output: "):
            output = output[len("Output: "):]
        one_example = "Input: {}\nOutput: {}".format(item["text"], output)
        examples.append(one_example)
    exp_str = "\n\n".join(examples)
    return exp_str

# ======= LLM API Code =======
def llm_call(prompt: str, system_prompt: str = "", model_name="deepseek-chat") -> str:
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
        model=model_name, # deepseek-chat
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        stream=False
    )
    return response.choices[0].message.content

def process_logprobs(logprobs):
    """
    Process the logprobs to get the token probabilities.
    Args:
        logprobs: The logprobs from the model response.
    Returns:
        A list of token probabilities.
    """
    new_logprobs = []
    for i in range(len(logprobs)):
        new_logprobs.append({
            'token': logprobs[i].token,
            'bytes': logprobs[i].bytes,
            'logprob': logprobs[i].logprob,
        })
    return new_logprobs

def llm_call_with_token_probs(prompt: str, system_prompt: str = "", model_name="deepseek-chat") -> str:
    """
    Calls the model with the given prompt and returns the response.

    Args:
        prompt (str): The user prompt to send to the model.
        system_prompt (str, optional): The system prompt to send to the model. Defaults to "".
        model (str, optional): The model to use for the call. Defaults to "claude-3-5-sonnet-20241022".

    Returns:
        str: The response from the language model.
        dict: The token probabilities from the model.
    """
    client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com") # TODO: Change the API key

    response = client.chat.completions.create(
        model=model_name, # deepseek-chat
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        stream=False,
        logprobs=True,  # Get token probabilities
    )
    # return two parts of data
    # 1. The pure response
    ans = response.choices[0].message.content
    # 2. The token probabilities
    ans_token_logits = response.choices[0].logprobs.content
    ans_token_logits = process_logprobs(ans_token_logits)
    return ans, ans_token_logits
    

# ======= Data Processing =======
def read_data(data_path: str) -> List[Dict[str, Any]]:
    """
    Read the training and test data from a file and convert it to GliNER format.
    Args:
        dataset: The path to the data file.
    Returns:
        A list of dictionaries containing the data.
    """
    data = load_json(data_path)
    return data

# ======= Confidence Estimiation Module =======
"""
方法一：基于GliNER等小模型的confidence score来估计整个句子的confidence score
    1. 计算输出的NER的confidence score。
    2. 根据Confidence Score来进行排序，对样本进行分类。

方法二：基于大模型的first token probability来估计confidence score
    1. 计算prompt的token
    2. 计算第一个token的概率
    3. 计算confidence score
"""

PROMPT_MAP = {
    "default": ONE_STAGE_FEW_SHOT_PROMPTS,
    "few_shot": ONE_STAGE_TAG_FEW_SHOT_PROMPTS,
    "zero_shot": ZERO_SHOT_TAG_PROMPT
}

# ======= Run LLMs-Evaluation =======
def run_LLM_Evaluator(dataset: str, model: str, demo_nums: int = 0, style: str = 'zero_shot', iteration: int = 0) -> List[Dict]:
    """
    Run the LLM for NER Evaluation.
    """
    # load data
    # path is the iteration output file
    data_dir = f'./output/zero_aug/{dataset}/iteration_{iteration}/'
    data_file = f'{data_dir}/{model}_0.json' # 改成itration的
    data = read_data(data_file)
    labels = list(LABEL_DIC[dataset].values())

    test_data, _ = get_crossNER_data(dataset=dataset, subset='test', style='tagging')
    prompt_format = PROMPT_MAP[style]
    
    # build demos
    examples = gen_examples(data, demo_nums)
    # output dir
    output_dir = f'./output/zero_aug/{dataset}/iteration_{iteration}/'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}_{demo_nums}_LLM_Ann_Test.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        test_data = load_json(output_file)
    # only test 200 samples, randomly selected 200
    test_data = random.sample(test_data, 200)
    # 处理数据
    for i in trange(len(test_data)):
        if 'response' in test_data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = test_data[i]['text']
        # 生成prompt
        if style == 'zero_shot':
            prompt = prompt_format.format(entity_types=labels, input_text=input_text)
        else:
            prompt = prompt_format.format(entity_types=labels, input_text=input_text, examples=examples)
        # print(f"Prompt: {prompt}")
        # exit()
        # 最多重试三次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response, response_token_logits = llm_call_with_token_probs(prompt=prompt)
                test_data[i]['response'] = response
                test_data[i]['response_token_logprobs'] = response_token_logits
                break
            except Exception as e:
                print(f"API 调用失败，重试次数: {attempt+1}/{max_retries}，错误信息: {e}")
                # 可根据需要添加 sleep 或其它逻辑
                time.sleep(1)
        # break
        # 关键：在每次调用完 llm_call 后就立刻保存
        save_json(test_data, output_file)
    save_json(test_data, output_file)

# ======= Run LLMs-Annotation =======

def run_LLM_Annotator(dataset: str, model: str, demo_nums: int = 0, style: str = 'zero_shot', iteration: int = 0) -> List[Dict]:
    """
    Run the LLM for NER Annotation.
    """
    # load data
    data_path = f"./datasets/zero_aug/{dataset}.json"
    data = read_data(data_path)
    labels = list(LABEL_DIC[dataset].values())
    # propmpts
    prompt_format = PROMPT_MAP[style]
    # Prepare the prompt: entity type
    # 先不考虑few-shot，直接跑zero-shot
    if iteration != 0:
        # 需要增加few shot
        few_shot = []
    
    # output dir
    output_dir = f'./output/zero_aug/{dataset}/iteration_{iteration}/'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}_{demo_nums}.json'
    # 如果文件已存在，则更新要处理的数据为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        data = load_json(output_file)
        # check if the test data has been processed
    
    # 处理数据
    for i in trange(len(data)):
        if 'response' in data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = data[i]['text']
        # 生成prompt
        if style == 'zero_shot':
            prompt = prompt_format.format(entity_types=labels, input_text=input_text)
        else:
            pass

        # 最多重试三次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response, response_token_logits = llm_call_with_token_probs(prompt=prompt)
                data[i]['response'] = response
                data[i]['response_token_logprobs'] = response_token_logits
                break
            except Exception as e:
                print(f"API 调用失败，重试次数: {attempt+1}/{max_retries}，错误信息: {e}")
                # 可根据需要添加 sleep 或其它逻辑
                time.sleep(1)

        # 关键：在每次调用完 llm_call 后就立刻保存
        save_json(data, output_file)
    save_json(data, output_file)

if __name__ == '__main__':
    # args
    dataset = 'ai' # ai, science, literature, music, politics, mit_res, mit_movie
    iteration = 0 # 0, 1, 2, 3
    mode = 'evaluation' # 'evaluation', 'annotation'
    # Load the data

    if iteration == 0 and mode == 'annotation':
        style = 'zero_shot'
    else:
        style = 'few_shot'
    
    if mode == 'evaluation':
        print("Running LLM Evaluator...")
        for demo_nums in [10, 20, 50, 100, 200, 300]:
            run_LLM_Evaluator(dataset=dataset, model='deepseek-chat', demo_nums=demo_nums, style=style, iteration=iteration)
        # run_LLM_Evaluator(dataset=dataset, model='deepseek-chat', demo_nums=100, style=style, iteration=iteration)
    elif mode == 'annotation':
        run_LLM_Annotator(dataset=dataset, model='deepseek-chat', demo_nums=0, style=style, iteration=iteration)
    else:
        raise ValueError("Invalid mode. Choose either 'evaluation' or 'annotation'.")