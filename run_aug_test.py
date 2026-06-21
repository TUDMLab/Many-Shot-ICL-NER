"""
This is the files for running the low-resource NER with few-shot ICL.
Step 1: Many-shot ICL
    100 examples for each domain as the ICL examples
Step 2: Few-shot LLM-as-Judge
    Retrieval Augmentation to evaluate the label of step 1
"""


from openai import OpenAI
from typing import List, Dict, Any
from const import CLIMATE_CHAIN_PROMPTS_ZERO, CLIMATE_CHAIN_PROMPTS_FEW
from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS
from prompts import ONE_STAGE_FEW_SHOT_PROMPTS, ONE_STAGE_TAG_FEW_SHOT_PROMPTS, ZERO_SHOT_TAG_PROMPT
from dataloader import get_crossNER_data, get_seed_data, feed_BM25, get_retrieved_examples_from_crossNER
from tqdm import tqdm, trange
import os, time
from utils import extract_entities, save_json_as_csv, load_json, get_ners, save_json
import random

# ====== AUG PROMPTS ======
from aug_prompt import (
    ANNOTATION_PROMPT,
    ERROR_ANALYSIS_PROMPT
)

# ======== LLM Class ========
API_DIC = {
    "deepseek-chat": "YOUR_API_KEY",
    "Qwen2.5-72B": "EMPTY",
    "gpt-4o": "YOUR_OPENAI_API_KEY"
}
BASE_URL = {
    "deepseek-chat": "https://api.deepseek.com",
    "Qwen2.5-72B": "http://localhost:8000/v1",
    "gpt-4o": "https://api.openai.com/v1"
}

class LLM:
    def __init__(self, model: str = "deepseek-chat", system_prompt: str = "", max_tokens: int = 8192):
        self.model = model
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.api_key = API_DIC[model]
        self.base_url = BASE_URL[model]
        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def call(self, prompt: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model, # deepseek-chat
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            stream=False
        )
        return response.choices[0].message.content
    def call_with_logits(self, prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            max_tokens=self.max_tokens,
            stream=False,
            logprobs=True, # Enable logprobs
        )
        # return two parts of data
        # 1. The pure response
        ans = response.choices[0].message.content
        # 2. The token probabilities
        ans_token_logits = response.choices[0].logprobs.content
        ans_token_logits = self.process_logprobs(ans_token_logits)
        return ans, ans_token_logits
    
    def process_logprobs(self, logprobs):
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

def gen_pos_examples(data):
    examples = []
    for item in data:
        one_example = "Input: {}\nOutput: {}".format(item["text"], item["target"])
        examples.append(one_example)
    exp_str = "\n\n".join(examples)
    return exp_str

def gen_neg_examples(data):
    examples = []
    for item in data:
        one_example = "Input: {}\n**Wrong** Output: {}\n**Correct** Output: {}".format(item["text"], item["response"], item["target"])
        examples.append(one_example)
    exp_str = "\n\n".join(examples)
    # add one more sentence to inform the model those are examples with wrong and the correct output
    exp_str = "**Below are some examples with wrong and correct outputs**:\n\n" + exp_str
    return exp_str


def get_label_definition(llm, dataset, data, labels):
    """
    Generate the label definition based on the seed data and labels.
    """
    # 生成Label Definition based on the 100 seed examples.
    # system_prompt = "You are an expert in the field of {}. You are given a text and you need to annotate the interested entities with the given labels.".format(dataset)
    examples = [d['target'] for d in data]
    examples = "\n\n".join(examples)
    prompt = ANNOTATION_PROMPT.format(dataset=dataset, labels=labels, examples=examples)
    # print(prompt)
    response = llm.call(prompt)
    return response

def get_error_analysis(llm, dataset, labels):
    """
    Generate the error analysis based on the seed data and labels.
    """
    # read txt file
    with open(f'./datasets/aug/{dataset}/annotation_guideline.txt', 'r') as f:
        annotation_guideline = f.read()
    # get error examples
    data = load_json(f"./datasets/aug/{dataset}/deepseek-chat_50-ann4reflection.json")
    errors = []
    error_text = []
    for sent in data:
        pred = extract_entities(sent['response'])
        # removing the "Output: " prefix from the response
        sent["response"] = sent["response"].replace("Output: ", "")
        if pred != sent['entities']:
            errors.append(sent)
            text = ""
            text += "Prediction: " + str(sent["response"]) + "\n"
            text += "Reference: " + str(sent["target"]) + "\n"
            error_text.append(text)
    print(len(errors))
    examples = "\n\n".join(error_text)
    prompt = ERROR_ANALYSIS_PROMPT.format(dataset=dataset, labels=labels, examples=examples, annotation_guideline=annotation_guideline)
    # print(prompt)
    response = llm.call(prompt)
    return response


def main_annotator(llm, model, dataset, data, pos_num = 50, neg_num = 10, retrieval=True):
    """
    Annotator: 用Seed数据来标注unlabel数据集
        Many-Shot ICL 来标注unlabel数据集
    """
    # 获取测试/要标注的数据
    unlabel_data = load_json(f'./datasets/aug/{dataset}/unlabel.json')
    # 获取Positive数据和构建pos_retriever
    pos_data, labels = get_seed_data(dataset=dataset, subset='unlabel', style='tagging')
    if retrieval:
        pos_retriever = feed_BM25(pos_data, dataset)
        # 获取Negative数据
        neg_data = load_json(f'./datasets/aug/{dataset}/negatives.json')
        neg_retriever = feed_BM25(neg_data, dataset)
    else:
        # randomly select pos_num examples from the seed data
        demo_data = random.sample(pos_data, pos_num)
        demo_str = gen_pos_examples(demo_data)
    
    # Read prompt
    with open(f'./datasets/aug/{dataset}/annotation_guideline_old.txt', 'r') as f:
        task_prompt = f.read()

    # 如果输出目录不存在，则创建
    output_dir = f'./output/aug/{dataset}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}-pos{pos_num}-neg{neg_num}-ann-all-no-definition-retrive-{retrieval}.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        unlabel_data = load_json(output_file)
    # 只测试100个样本，随机选择100个
    # unlabel_data = random.sample(unlabel_data, 100)

    for i in trange(len(unlabel_data)):
        # check if the test data has been processed
        if 'response' in unlabel_data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = unlabel_data[i]['text']
        # retrieve positive and negative examples
        if retrieval:
            topN_pos_examples = get_retrieved_examples_from_crossNER(retriever=pos_retriever, query=input_text, dataset=pos_data, topN=pos_num)
            demo_str = gen_pos_examples(topN_pos_examples)
        if neg_num > 0:
            topN_neg_examples = get_retrieved_examples_from_crossNER(retriever=neg_retriever, query=input_text, dataset=neg_data, topN=neg_num)
            neg_str = gen_neg_examples(topN_neg_examples)
            demo_str = "## Correct/Positive Examples\n" + demo_str + "\n\n## Wrong/Negative Examples\n" + neg_str
        
        # 组装prompt
        prompt = task_prompt + "\n\n# Examples\n\n" + demo_str + "\n\n# Test Data:\nInput: " + input_text
        # print(prompt)
        # exit()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.call(prompt=prompt)
                unlabel_data[i]['response'] = response
                break
            except Exception as e:
                print(f"API 调用失败，重试次数: {attempt+1}/{max_retries}，错误信息: {e}")
                # 可根据需要添加 sleep 或其它逻辑
                time.sleep(1)
        # response = llm.call(prompt=prompt)
        # unlabel_data[i]['response'] = response

        # 关键：在每次调用完 llm_call 后就立刻保存
        save_json(unlabel_data, output_file)
    save_json(unlabel_data, output_file)



def main():
    """
    0. 用Seed数据来生成Annotation Guidelines
        a. 生成Label Definition
        b. 根据Error Analysis，生成Annotation Guidelines
    1. Annotator: 用Seed数据来标注unlabel数据集
        Many-Shot ICL 来标注unlabel数据集
    2. Judger: 用Seed数据来judge 标注的数据集
        Few-Shot LLM-as-Judge 来judge标注的数据集
        根据Error Types的不同，让LLMs来进行判断
    3. Modifier: (few-shot)
        用Seed数据来修改标注的数据集
    """

    dataset = "ai"
    model = "Qwen2.5-72B"


    # 获取数据
    data, labels = get_seed_data(dataset=dataset, subset='seed', style='tagging')

    system_prompt = "You are an expert in the field of {}. You are given a text and you need to annotate the interested entities with the given labels.".format(dataset)

    llm = LLM(model=model, system_prompt=system_prompt)

    # ==> a. 生成Label Definition based on the 100 seed examples.
    
    # response = get_label_definition(llm, dataset, data, labels)
    # # write the response to a file
    # output_dir = f'./datasets/aug/{dataset}'
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = f'{output_dir}/annotation_guideline_{model}.txt'
    # with open(output_file, 'w') as f:
    #     f.write(response)
    # print("Label Definition: ", response)

    # ==> b. 根据新的Annotation Guidelines 来去进行50-shot ICL, 然后找到所有的errors再来进行Error Analysis和Polishing Annotation Guidelines
    # response = get_error_analysis(llm, dataset, labels)
    # print(response)
    # # write the response to a file
    # output_dir = f'./datasets/aug/{dataset}'
    # os.makedirs(output_dir, exist_ok=True)
    # output_file = f'{output_dir}/annotation_guideline_refined.txt'
    # with open(output_file, 'w') as f:
    #     f.write(response)

    # ==> Many-Shot ICL to label the unlabel data
    main_annotator(llm, model, dataset, data, pos_num=100, neg_num=0, retrieval=True)
    main_annotator(llm, model, dataset, data, pos_num=50, neg_num=0, retrieval=True)
    main_annotator(llm, model, dataset, data, pos_num=50, neg_num=0, retrieval=False)

    main_annotator(llm, model, dataset, data, pos_num=100, neg_num=10, retrieval=True)
    main_annotator(llm, model, dataset, data, pos_num=100, neg_num=20, retrieval=True)
    main_annotator(llm, model, dataset, data, pos_num=50, neg_num=10, retrieval=True)
    main_annotator(llm, model, dataset, data, pos_num=50, neg_num=20, retrieval=True)

    # main_annotator(llm, model, dataset, data, pos_num=100, neg_num=10, retrieval=True)
    



if __name__ == '__main__':
    main()
    
    