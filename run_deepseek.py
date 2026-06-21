from openai import OpenAI
from typing import List, Dict, Any
from const import CLIMATE_CHAIN_PROMPTS_ZERO, CLIMATE_CHAIN_PROMPTS_FEW
from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS
from prompts import ONE_STAGE_FEW_SHOT_PROMPTS, ONE_STAGE_TAG_FEW_SHOT_PROMPTS, ZERO_SHOT_TAG_PROMPT
from dataloader import get_crossNER_data, feed_BM25, get_retrieved_examples
from tqdm import tqdm, trange
import os, time
from utils import extract_entities, save_json_as_csv, load_json, get_ners, save_json
import random

API_KEYS = {
    "deepseek-chat": "YOUR_API_KEY",
    "gpt-4o": "YOUR_OPENAI_API_KEY"
}

LABEL_DIC = {
    'ai': AI_CLASSS,
    'science': SCIENCE_CLASS,
    'literature': LITERATURE_CLASSS,
    'music': MUSIC_CLASSS,
    'politics': POLITICS_CLASSS,
}

PROMPT_MAP = {
    "default": ONE_STAGE_FEW_SHOT_PROMPTS,
    "tagging": ONE_STAGE_TAG_FEW_SHOT_PROMPTS,
    "zero_shot_tagging": ZERO_SHOT_TAG_PROMPT
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
    api_key = "YOUR_API_KEY"
    base_url = "https://api.deepseek.com"
    client = OpenAI(api_key=api_key, base_url=base_url) # TODO: Change the API key

    response = client.chat.completions.create(
        model=model, # deepseek-chat
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        stream=False
    )
    return response.choices[0].message.content

def chain(input: str, prompts: List[str]) -> List[str]:
    """Chain multiple LLM calls with retries, passing results between steps."""
    response = input
    result = [response]  # 初始化结果列表，包含初始输入
    
    for i, prompt in enumerate(prompts, 1):
        max_retries = 3
        current_response = None
        last_exception = None
        
        # 单个prompt的重试逻辑
        for attempt in range(max_retries):
            try:
                current_response = llm_call(f"{prompt}\nInput: {response}")
                break  # 成功则跳出重试循环
            except Exception as e:
                last_exception = e
                print(f"Prompt {i} 第 {attempt+1} 次调用失败: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # 指数退避：1s, 2s, 4s, 8s, 16s
                    print(f"等待 {wait_time} 秒后重试...")
                    time.sleep(wait_time)
        
        # 判断是否成功
        if current_response is not None:
            response = current_response  # 更新response供下一步使用
            result.append(response)
        else:
            # 所有重试失败，抛出异常并包含已处理的结果（便于调试）
            raise RuntimeError(
                f"Prompt {i} 全部 {max_retries} 次尝试均失败。\n"
                f"最后错误: {str(last_exception)}\n"
                f"已处理结果: {result}"
            )
    
    return result


# def filter_entities(entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """Filter out entities that are not from the Climate domain."""
#     return [ent for ent in entities if ent['label'] != 'other']

one_stage_prompt = """Extract all entities from the text below and output them in JSON format. Each entity should include:
- text: The entity mention
- type: Entity type. The interested types are: 'project', 'location', 'model', 'experiment', 'platform', 'instrument', 'provider', 'variable', 'weather event', 'natural hazard', 'teleconnection', 'ocean circulation'.

Note: If the same entity appears multiple times in the text, include it multiple times in the results.
"""

def run_one_stage():
    doc_files = os.listdir('./datasets/climate/')
    for file in tqdm(doc_files):
        print(f"Processing {file}")
        data = load_json(f'./datasets/climate/{file}')
        doc_res = {}
        doc_res['doc_key'] = file
        gt = []
        pred = []
        for chunk in tqdm(data):
            text = data[0]['text']
            chunk_labels = chunk['entities']
            chunk_gt = [(ent['substring'], ent['label']) for ent in chunk_labels]
            # run llm
            output = llm_call(f"{one_stage_prompt}\n\nInput: {text}")
            # append the results
            gt.append(chunk_gt)
            pred.append(output)
            # break
        doc_res['pred'] = pred
        doc_res['gt'] = gt
        save_json(doc_res, f'./output/baselines/llm_one_stage/LLM_{file}')

def run_tag_workflow():
    doc_files = os.listdir('./datasets/climate/')
    out_path = os.listdir('./output/baselines/llm_definition/')
    for file in tqdm(doc_files):
        print(f"Processing {file}")
        save_name = "LLM_" + file
        if save_name in out_path:
            print(f"Skipping {file}")
            continue
        data = load_json(f'./datasets/climate/{file}')
        doc_res = {}
        doc_res['doc_key'] = file
        gt = []
        pred = []
        for chunk in tqdm(data):
            text = data[0]['text']
            chunk_labels = chunk['entities']
            chunk_gt = [(ent['substring'], ent['label']) for ent in chunk_labels]
            # run llm
            output = chain(text, CLIMATE_CHAIN_PROMPTS_ZERO)
            ner_predicts = extract_entities(output[-1])
            chunk_pred = [ent for ent in ner_predicts if ent[1] != 'other']
            # append the results
            gt.append(chunk_gt)
            pred.append(chunk_pred)
            # break
        doc_res['pred'] = pred
        doc_res['gt'] = gt
        save_json(doc_res, f'./output/baselines/llm_definition/LLM_{file}')

def run_tag_workflow_few_shot():
    """
    Runs the few-shot workflow, saving results incrementally for each chunk and file.
    """
    print("Running few-shot")
    
    # Paths
    doc_files = os.listdir('./datasets/climate/')
    output_dir = './output/baselines/ds_few/'
    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    # Check existing files
    out_files = os.listdir(output_dir)
    
    for file in tqdm(doc_files):
        print(f"Processing {file}")
        save_name = "LLM_" + file
        save_path = os.path.join(output_dir, save_name)

        # Skip already completed files
        if save_name in out_files:
            print(f"Skipping {file}")
            continue

        # Load input data
        data = load_json(f'./datasets/climate/{file}')

        # Initialize or resume processing
        doc_res = {"doc_key": file, "pred": [], "gt": []}
        if os.path.exists(save_path):
            print(f"Resuming {file}...")
            doc_res = load_json(save_path)

        # Extract previously processed chunks
        processed_chunks = len(doc_res["pred"])

        # Process each chunk
        for idx, chunk in enumerate(tqdm(data)):
            if idx < processed_chunks:
                continue  # Skip already processed chunks

            # Prepare chunk data
            text = chunk['text']
            chunk_labels = chunk['entities']
            chunk_gt = [(ent['substring'], ent['label']) for ent in chunk_labels]

            try:
                # Run LLM inference
                output = chain(text, CLIMATE_CHAIN_PROMPTS_FEW)
                ner_predicts = extract_entities(output[-1])
                chunk_pred = [ent for ent in ner_predicts if ent[1] != 'other']
            except Exception as e:
                print(f"Error processing chunk {idx} in {file}: {e}")
                continue

            # Append results
            doc_res["gt"].append(chunk_gt)
            doc_res["pred"].append(chunk_pred)

            # Save intermediate results after each chunk
            save_json(doc_res, save_path)

        print(f"Finished processing {file}")

def gen_examples(data, shots=100):
    # randomly select shots examples
    random.shuffle(data)
    data = data[:shots] # 
    examples = []
    for item in data:
        one_example = "Input: {}\nOutput: {}".format(item["text"], item["target"])
        examples.append(one_example)
    exp_str = "\n\n".join(examples)
    return exp_str


def run_few_shot_CrossNER(dataset: str, model: str, demo_nums: int = 100, style: str = 'default'):

    # 获取数据
    train_data, labels = get_crossNER_data(dataset=dataset, subset='train', style='tagging')
    if len(train_data) < demo_nums:
        dev_data, _ = get_crossNER_data(dataset=dataset, subset='dev', style='tagging') # 额外的dev数据
        train_data.extend(dev_data)
        if len(train_data) < demo_nums:
            return
    test_data, _ = get_crossNER_data(dataset=dataset, subset='test', style='tagging')
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

    # only test 200 samples, randomly selected 200
    test_data = random.sample(test_data, 200)
    # test all the test data
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
                response = llm_call(prompt=prompt)
                test_data[i]['response'] = response
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

def run_few_shot_CrossNER_RM25(dataset:str, model:str, demo_nums:int=25, style: str = 'default'):
    # 获取数据
    train_data, labels = get_crossNER_data(dataset=dataset, subset='train', style=style)
    dev_data, _ = get_crossNER_data(dataset=dataset, subset='dev', style=style) # 额外的dev数据
    train_data.extend(dev_data)
    test_data, _ = get_crossNER_data(dataset=dataset, subset='test', style=style)
    prompt_format = PROMPT_MAP[style]
    retriever = feed_BM25(train_data, dataset)
    
    # 如果输出目录不存在，则创建
    output_dir = f'./output/{dataset}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}_BM25_{demo_nums}.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        test_data = load_json(output_file)

    # only test 200 samples, randomly selected 200
    test_data = random.sample(test_data, 200)
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
        
        # debug
        # print(len(topN_examples))
        # print(prompt)
        # exit()
        
        # 最多重试三次
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm_call(prompt=prompt, model=model)
                test_data[i]['response'] = response
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

if __name__ == '__main__':
    model = 'deepseek-chat'
    # domains = ['ai', 'science']
    domains = ['ai', 'science', 'literature', 'politics']
    # shots = [1, 5, 10, 25, 50, 100, 200]
    shots = [200, 300]
    for domain in domains:
        for shot in shots:
            print(f"Running {domain} with {shot} examples")
            run_few_shot_CrossNER_RM25(dataset=domain, model=model, demo_nums=shot, style='tagging')
        # run_few_shot_CrossNER(dataset=domain, model='deepseek-chat', demo_nums=shot, style='tagging')