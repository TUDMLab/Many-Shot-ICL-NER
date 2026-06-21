from openai import OpenAI
from typing import List, Dict, Any
from const import CLIMATE_CHAIN_PROMPTS_ZERO, CLIMATE_CHAIN_PROMPTS_FEW
from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS
from prompts import ONE_STAGE_FEW_SHOT_PROMPTS, ONE_STAGE_TAG_FEW_SHOT_PROMPTS, ZERO_SHOT_TAG_PROMPT
from dataloader import get_crossNER_data, get_seed_data, feed_BM25, get_retrieved_examples
from tqdm import tqdm, trange
import os, time
from utils import extract_entities, save_json_as_csv, load_json, get_ners, save_json, compute_entity_tag_probs, parse_llm_list_string_fix_first
import random
import argparse
import re
import ast
import pdb

# ====== AUG PROMPTS ======
from aug_prompt import (
    ANNOTATION_PROMPT,
    ERROR_ANALYSIS_PROMPT,
    TYPE_ERROR_PROMPT,
    TYPE_ERROR_SYSTEM_PROMPT,
    SPAN_ERROR_PROMPT,
    SPAN_SYSTEM_PROMPT,
    MISSING_SYSTEM_PROMPT,
    MISSING_ERROR_PROMPT,
    SPURIOUS_SYSTEM_PROMPT,
    SPURIOUS_ERROR_PROMPT,
)

# ======== LLM Class ========
API_DIC = {
    # "deepseek-chat": "YOUR_API_KEY", 
    "deepseek-chat": "YOUR_API_KEY", # "YOUR_API_KEY", "YOUR_API_KEY", YOUR_API_KEY, YOUR_API_KEY
    "Qwen2.5-72B": "EMPTY",
    "Llama3.1-70B": "EMPTY",
    "Qwen2.5-7B": "EMPTY",
    "Llama3.1-8B": "EMPTY",
    "Qwen-7B-1M": "EMPTY",
    "gpt-4o": "YOUR_OPENAI_API_KEY",
    "gemini": "YOUR_GEMINI_API_KEY"
}

BASE_URL = {
    "deepseek-chat": "https://api.deepseek.com",
    "Qwen2.5-72B": "http://localhost:8000/v1",
    "gpt-4o": "https://api.openai.com/v1",
    "Llama3.1-70B": "http://localhost:8000/v1",
    "Qwen2.5-7B": "http://localhost:8001/v1",
    "Llama3.1-8B": "http://localhost:8002/v1",
    "Qwen-7B-1M": "http://localhost:8003/v1",
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

def run_icl(model, dataset, demo_num=0):
    """
    Run the ICL with the given model and dataset.
    """
    # get the data
    data, labels = get_seed_data(dataset=dataset, subset='seed', style='tagging')
    # get the prompt
    with open(f'./datasets/aug/{dataset}/annotator_prompt.txt', 'r') as f:
        task_prompt = f.read()
    
    # 定义LLM
    system_prompt = """You are a Named Entity Recognition (NER) annotator specialized in {}-domain texts. Your task is to identify and tag entities in the given text EXACTLY as follows:
    - Wrap each entity with `<entity type="[TYPE]">...</entity>` tags, where `[TYPE]` must be one of the provided entity types.  
    - Preserve all non-entity text unchanged.""".format(dataset)
    print(system_prompt)
    llm = LLM(model=model, system_prompt=system_prompt)

    # 构建输出目录
    output_dir = f'./datasets/aug/{dataset}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}-{demo_num}-shots-ICL.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        data = load_json(output_file)
    
    for i in trange(len(data)):
        # check if the test data has been processed
        if 'response' in data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = data[i]['text']
        # 组装prompt
        task_prompt = task_prompt + "\n\n" + """
# Output Format:
- Identify all text segments that correspond exactly to one of the **valid entity types** listed above.
- Leave all text segments that are *not* valid entities completely unchanged.
- Tag all valid entities, even if they appear more than once.
- Using XML format to wrap the entity text segments. Ensure strict XML formatting: correctly nested tags, proper attribute quoting (`type="..."`).
- The final output must *only* contain the processed text with the XML tags. Do not include any introductory phrases, explanations, summaries, or any text other than the annotated original content.

# Example Output Format:
<entity type=\"algorithm\">Collaborative filtering</entity> encompasses techniques for matching people with similar interests and making <entity type=\"product\">recommender system</entity> on this basis .
"""
        prompt = task_prompt + "\n\nInput: " + input_text + "\nOutput: "
        # print(prompt)
        # exit()
        
        response = llm.call(prompt=prompt)
        data[i]['response'] = response

        # 关键：在每次调用完 llm_call 后就立刻保存
        save_json(data, output_file)
    save_json(data, output_file)


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

def main_annotator(model, dataset, demo_num = 100, logits=False, name_suffix=""):
    """
    Annotator: 用Seed数据来标注unlabel数据集
        Many-Shot ICL 来标注unlabel数据集
    """
    
    # 获取测试/要标注的数据
    unlabel_data = load_json(f'./datasets/aug/{dataset}/unlabel.json')
    # NOTE: 用来测试的部分====
    # unlabel_data, labels = get_crossNER_data(dataset=dataset, subset='test', style='tagging')

    # 获取Positive数据和构建pos_retriever
    pos_data, labels = get_seed_data(dataset=dataset, subset='seed', style='tagging')
    # 生成demo数据=> randomly split 10 examples for demos and 90 for test
    # randomly select 10 examples from the positive data for demo and 90 for test
    # random.shuffle(pos_data)
    # # print(type(pos_data))
    # unlabel_data = pos_data[:90]
    # pos_data = pos_data[90:]

    pos_retriever = feed_BM25(pos_data, dataset)

    with open(f'./datasets/aug/{dataset}/annotator_prompt.txt', 'r') as f:
        task_prompt = f.read()


    # 定义LLM
    system_prompt = """You are a Named Entity Recognition (NER) annotator specialized in {}-domain texts. Your task is to identify and tag entities in the given text EXACTLY as follows:
    - Wrap each entity with `<entity type="[TYPE]">...</entity>` tags, where `[TYPE]` must be one of the provided entity types.  
    - Preserve all non-entity text unchanged.""".format(dataset)
    print(system_prompt)
    llm = LLM(model=model, system_prompt=system_prompt)
    

    # 如果输出目录不存在，则创建
    output_dir = f'./output/aug/{dataset}'
    os.makedirs(output_dir, exist_ok=True)
    if logits:
        output_file = f'{output_dir}/{model}-demo{demo_num}-ann-dev-logits{name_suffix}.json' 
    else:
        output_file = f'{output_dir}/refine-demos.json'
        # output_file = f'{output_dir}/{model}-demo{pos_num}-ann-dev{name_suffix}.json'
    print(f"===> Output file: {output_file}")
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        unlabel_data = load_json(output_file)

    for i in trange(len(unlabel_data)):
        # check if the test data has been processed
        if 'response' in unlabel_data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        # pdb.set_trace()
        # print(unlabel_data[i])
        input_text = unlabel_data[i]['text']
        # retrieve positive and negative examples
        # 如果是100-shot，直接使用100个例子，不用检索
        if demo_num == 100:
            demo_str = gen_examples(pos_data, shots=demo_num)
        else:
            topN_pos_examples = get_retrieved_examples_from_crossNER(retriever=pos_retriever, query=input_text, dataset=pos_data, topN=demo_num)
            demo_str = gen_pos_examples(topN_pos_examples)

        # 组装prompt
        prompt = task_prompt + "\n\n# Examples\n\n" + demo_str + "\n\n# Test Data:\nInput: " + input_text + "\nOutput: "
        # print(prompt) # debug时用
        # exit()

        max_retries = 3
        for attempt in range(max_retries):
            try:
                # 判断是否需要logits
                if logits:
                    response, logits = llm.call_with_logits(prompt=prompt)
                    unlabel_data[i]['response'] = response
                    unlabel_data[i]['logits'] = logits
                else:
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
        # exit() # NOTE: debug时用, 测试记得删掉
    save_json(unlabel_data, output_file)


def generate_type_error_demos(data_list):
    """
    Args:
        data_list (list of dict): each dict contains keys:
            - 'text': str, the original text
            - 'pred': list of [text_span, type] pairs (initial entity list)
            - 'errors': dict containing 'type': list of type error records
                each type error record is [text_span, wrong_type, _, correct_type]
    
    Returns:
        list of str: each string is a demo following the prompt format
    """
    demos = []

    for idx, sent in enumerate(data_list):
        text = sent['text']
        pred_ents = sent['pred'].copy()  # avoid modifying original
        type_errors = sent.get('errors', {}).get('type', [])

        # Fix type errors
        if type_errors:
            for t_error in type_errors:
                error_one = [t_error[0], t_error[1]]   # [text_span, wrong_type]
                target = [t_error[0], t_error[3]]      # [text_span, correct_type]
                for i in range(len(pred_ents)):
                    if pred_ents[i] == error_one:
                        pred_ents[i] = target

        # Format as one demo
        one_demo = []
        # one_demo.append(f"Example {idx + 1}:")
        one_demo.append(f"Text: {text}")
        one_demo.append(f"Initial Entity List: {sent['pred']}")
        one_demo.append(f"Refined Entity List: {pred_ents}")

        demos.append("\n".join(one_demo))

    return demos



def type_refiner(model, ann_data_file, dataset, demo_num=100, logits=True, suffix="", retrieval=False):
    """
    用于refine标注数据的entity types
    """
    # 获取数据
    data = load_json(ann_data_file)
    label_map = LABEL_DIC[dataset] # mapping the label to the specific domain for LLMs
    labels = [label for label in label_map.values()]
    # 获取demo数据
    demo_data = load_json(f"./datasets/aug/{dataset}/refiner-demos-errors.json")
    if retrieval:
        demo_retriver = feed_BM25(demo_data, dataset)
    # 获取type definitions
    with open(f'./datasets/aug/{dataset}/type_definition_typing.txt', 'r') as f:
        type_definitions = f.read()
    # 构建和获取输出目录
    output_dir = f'./output/aug/{dataset}/refiner/{model}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/dev-filter-spurious-missing-type-refiner.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        data = load_json(output_file)
    
    # 定义LLM
    system_prompt = TYPE_ERROR_SYSTEM_PROMPT
    llm = LLM(model=model, system_prompt=system_prompt)

    # === get refine demo_str ===
    demo_list = generate_type_error_demos(demo_data)
    demo_str = "\n\n".join(demo_list)

    for i in trange(len(data)):
        # if refine is False, then skip the test data
        if data[i]['refine'] == False:
            continue
        input_text = data[i]['text']
        # remove the "Output: " prefix from the response
        # data[i]['response'] = data[i]['response'].replace("Output: ", "")
        # data[i]['pred'] = extract_entities(data[i]['response'])
        # 组装prompt
        if retrieval:
            topN_examples = get_retrieved_examples(retriever=demo_retriver, query=input_text, dataset=demo_data, topN=demo_num)
            demo_list = generate_type_error_demos(topN_examples)
            demo_str = "\n\n".join(demo_list)
        prompt = TYPE_ERROR_PROMPT.format(
                                       labels=labels, 
                                       type_definitions=type_definitions,
                                       examples=demo_str,
                                       input_text=input_text,
                                       input_entity_list=data[i]['pred'],
                                       )

        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.call(prompt=prompt)
                data[i]['type_refined_response'] = response
                break
            except Exception as e:
                print(f"API 调用失败，重试次数: {attempt+1}/{max_retries}，错误信息: {e}")
                # 可根据需要添加 sleep 或其它逻辑
                time.sleep(1)

        # 关键：在每次调用完 llm_call 后就立刻保存
        save_json(data, output_file)
        # NOTE: Debug时用
        # print(system_prompt)
        # print("----------")
        # print(prompt)
        # print(response)
        # exit()
    save_json(data, output_file)
    return data, output_file



def build_missing_entity_example_safe(original_annotation: str, missing_errors: list[list[str]]) -> str:
  """
  Constructs a refined annotation example by adding missing entity tags,
  ensuring replacements only happen outside existing <entity> tags.

  Args:
    original_annotation: The original text string containing some annotations.
    missing_errors: A list of lists, where each inner list contains:
                      [entity_text (str): The plain text of the missing entity,
                       entity_type (str): The type to assign to the missing entity]

  Returns:
    A string representing the refined annotation with the missing entities tagged,
    without modifying text already inside <entity> tags.
  """
  original_entities_map = {}
  placeholder_index = 0

  def replacer_mask(match):
    """Replaces found entity with a placeholder and stores it."""
    nonlocal placeholder_index
    placeholder = f"__ENTITY_{placeholder_index}__"
    original_entities_map[placeholder] = match.group(0) # Store original tagged entity
    placeholder_index += 1
    return placeholder

  # Step 1: Mask existing entities
  # This regex tries to robustly find entity tags, allowing variations in whitespace
  # and handling multi-line content within tags via re.DOTALL.
  # It assumes type attribute uses double quotes. Adapt if single quotes are used.
  entity_regex = r'<entity\s+type\s*=\s*".*?"\s*>.*?</entity>'
  masked_annotation = re.sub(entity_regex, replacer_mask, original_annotation, flags=re.DOTALL)
#   print(f"Masked Annotation: {masked_annotation}") # Debugging

  # Step 2: Apply missing entity replacements to the masked string
  current_annotation = masked_annotation
  for entity_text, entity_type in missing_errors:
    # Basic check to avoid replacing if the text to find IS a placeholder itself
    # or if the entity_text contains the placeholder pattern (unlikely edge case)
    if entity_text.startswith("__ENTITY_") and entity_text.endswith("__"):
        continue
    if "__ENTITY_" in entity_text:
        print(f"Warning: Skipping replacement for '{entity_text}' as it contains the placeholder pattern.")
        continue

    # Create the new tag for the missing entity
    replacement_tag = f'<entity type="{entity_type}">{entity_text}</entity>'

    # Perform simple string replacement on the masked annotation.
    # Since existing entities are masked, this won't affect them.
    current_annotation = current_annotation.replace(entity_text, replacement_tag)

  # Step 3: Unmask entities by replacing placeholders with original content
  final_annotation = current_annotation
  # Iterate through placeholders in the order they were created
  for i in range(placeholder_index):
      placeholder_to_find = f"__ENTITY_{i}__"
      # Retrieve the original entity content using the placeholder key
      original_entity_content = original_entities_map.get(placeholder_to_find, "") # Use .get for safety
      # Replace the placeholder back with the original content
      final_annotation = final_annotation.replace(placeholder_to_find, original_entity_content)

  return final_annotation

def generate_missing_entity_demos(data_list):
    """
    构建missing entity error的demo
    """
    demos = []

    for idx, sent in enumerate(data_list):
        text = sent['text']
        missing_errors = sent['errors']['missing']

        # Format as one demo
        one_demo = []
        # one_demo.append(f"Example {idx + 1}:")
        # one_demo.append(f"Text: {text}")
        sent['response'] = sent['response'].replace("Output: ", "")
        one_demo.append(f"Input: {sent['response']}")
        # Build the refined annotation
        refined_annotation = build_missing_entity_example_safe(sent['response'], missing_errors)
        one_demo.append(f"Output: {refined_annotation}")

        demos.append("\n".join(one_demo))

    return demos

def generate_missing_entity_demos_list(data_list):
    """
    构建missing entity error的demo
    """
    demos = []

    for idx, sent in enumerate(data_list):
        text = sent['text']
        missing_errors = sent['errors']['missing']
        if len(missing_errors) == 0:
            missing_errors = None

        # Format as one demo
        one_demo = []

        one_demo.append(f"Text: {sent['text']}")
        sent['response'] = sent['response'].replace("Output: ", "")
        pred_ents = extract_entities(sent['response'])
        sent['pred'] = pred_ents
        one_demo.append(f"Entities: {pred_ents}")
        one_demo.append(f"Missing Entities: {missing_errors}")

        demos.append("\n".join(one_demo))

    return demos

def generate_missing_entity_demos_refine_list(data_list):
    """
    构建missing entity error的demo
    """
    demos = []

    for idx, sent in enumerate(data_list):
        text = sent['text']
        missing_errors = sent['errors']['missing']

        # Format as one demo
        sent['response'] = sent['response'].replace("Output: ", "")
        pred_ents = extract_entities(sent['response'])
        sent['pred'] = pred_ents
        refine_list = sent['pred'].copy()
        if len(missing_errors) > 0:
            # add each missing entity to the pred_ents
            for missing in missing_errors:
                refine_list.append(missing)
        one_demo = []
        one_demo.append(f"Text: {sent['text']}")
        one_demo.append(f"Initial Entitity List: {sent['pred']}")
        one_demo.append(f"Refined Entitity List: {refine_list}")

        demos.append("\n".join(one_demo))

    return demos

def missing_refiner(model, ann_data_file,dataset, demo_num=100, suffix="", retrieval=False):
    """
    用于refine标注数据的mising entities
    """
    # 获取数据
    data = load_json(ann_data_file)
    label_map = LABEL_DIC[dataset] # mapping the label to the specific domain for LLMs
    labels = [label for label in label_map.values()]
    # 获取demo数据
    demo_data = load_json(f"./datasets/aug/{dataset}/refiner-demos-errors.json")
    if retrieval:
        demo_retriver = feed_BM25(demo_data, dataset)
    # 获取type definitions
    with open(f'./datasets/aug/{dataset}/type_definitions.txt', 'r') as f:
        type_definitions = f.read()
    # 构建和获取输出目录
    output_dir = f'./output/aug/{dataset}/refiner/{model}'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}-demo100-dev-filter-suprious-missing-refiner.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        data = load_json(output_file)
    
    # 定义LLM
    system_prompt = MISSING_SYSTEM_PROMPT
    llm = LLM(model=model, system_prompt=system_prompt)

    # === get refine demo_str ===
    demo_list = generate_missing_entity_demos_list(demo_data) # NOTE: 新版本测试
    demo_str = "\n\n".join(demo_list)


    for i in trange(len(data)):
        # check if need to refine
        if data[i]['refine'] == False:
            continue
        # check if the test data has been processed
        if 'missing_refined_response' in data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = data[i]['text']
        if retrieval:
            topN_demos = get_retrieved_examples(retriever=demo_retriver, query=input_text, dataset=demo_data, topN=demo_num)
            demo_list = generate_missing_entity_demos_list(topN_demos)
            demo_str = "\n\n".join(demo_list)
        # remove the "Output: " prefix from the response
        data[i]['response'] = data[i]['response'].replace("Output: ", "")
        # data[i]['pred'] = extract_entities(data[i]['response'])
        pred = data[i]['pred']
        # ===== debug =====
        # print(data[i]['refine'])
        # print(extract_entities(data[i]['response']))
        # print(pred)
        # print(data[i]['spurious_refined_response'])
        # print("================")
        # 组装prompt
        prompt = MISSING_ERROR_PROMPT.format(
                                       type_definitions=type_definitions,
                                       examples=demo_str,
                                       input_text=input_text,
                                       input_entities=pred,
                                       )
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.call(prompt=prompt)
                # 清除掉多余的prefix
                response = response.replace("Output:", "")
                data[i]['missing_refined_response'] = response
                break
            except Exception as e:
                print(f"API 调用失败，重试次数: {attempt+1}/{max_retries}，错误信息: {e}")
                # 可根据需要添加 sleep 或其它逻辑
                time.sleep(1)
        # 关键：在每次调用完 llm_call 后就立刻保存
        save_json(data, output_file)

        # NOTE: Debug时用
        # print(system_prompt)
        # print("----------")
        # print(prompt)
        # print(response)
        # exit()

    save_json(data, output_file)
    return output_file


def build_span_error_example(annotation: str, span_errors: list[list[str]]) -> str:
    """
    Corrects span errors based on provided examples, applying consistent rules
    for expansion and contraction.

    Rules derived from examples provided on Apr 28, 2025:
    - Expansions consume adjacent text.
    - Contractions release removed text as adjacent plain text.
    - Applies corrections sequentially; order of errors might matter.
    - Processes only the first found instance for each error rule (count=1).

    Args:
        annotation: The original text string containing <entity> annotations.
        span_errors: A list of lists: [org_text, org_type, correct_text, correct_type]

    Returns:
        Refined annotation string.
    """
    # Make a copy to modify, ensuring original annotation string isn't changed directly
    # if it's passed as a mutable object elsewhere, although strings are immutable in Python.
    refined_annotation = annotation

    # Process errors one by one
    for error_index, error_detail in enumerate(span_errors):
        if len(error_detail) != 4:
            print(f"Warning: Skipping invalid error detail entry #{error_index}: {error_detail}")
            continue
        original_text, original_type, correct_text, correct_type = error_detail

        # Skip if no change needed
        if original_text == correct_text and original_type == correct_type:
            continue

        # Prepare new tag and regex for old tag
        new_tag = f'<entity type="{original_type}">{correct_text}</entity>' # NOTE: original_type is used for the new tag, 因为不改变type
        # Regex for the original tag structure. Using a group captures the whole tag.
        original_tag_regex = (
            r'(<entity\s+type\s*=\s*"' + re.escape(original_type) + r'"\s*>' +
            re.escape(original_text) +
            r'</entity>)'
        )

        processed = False # Flag to track if a specific case handled this error

        # --- Determine type of change ---
        is_expansion = len(correct_text) > len(original_text)
        is_contraction = len(correct_text) < len(original_text)

        # Store annotation state before attempting changes for THIS error
        annotation_before_this_error = refined_annotation

        # --- Apply specific logic based on change type ---

        # 1. Prefix Expansion (B -> A B)
        if is_expansion and correct_text.endswith(original_text):
            prefix = correct_text[:-len(original_text)]
            pattern = re.escape(prefix) + original_tag_regex # Match prefix + original tag
            replacement = new_tag
            # Use a function scope to properly handle 'processed' flag update
            def replace_func(match):
                nonlocal processed
                processed = True
                return replacement
            refined_annotation = re.sub(pattern, replace_func, refined_annotation, count=1)

        # 2. Suffix Expansion (A -> A B)
        elif is_expansion and correct_text.startswith(original_text): # Use elif assuming cases are distinct
            suffix = correct_text[len(original_text):]
            pattern = original_tag_regex + re.escape(suffix) # Match original tag + suffix
            replacement = new_tag
            def replace_func(match):
                nonlocal processed
                processed = True
                return replacement
            refined_annotation = re.sub(pattern, replace_func, refined_annotation, count=1)

        # 3. Prefix Contraction (A B -> B)
        elif is_contraction and original_text.endswith(correct_text):
            prefix = original_text[:-len(correct_text)]
            replacement = prefix + new_tag # Prefix becomes plain text before new tag
            def replace_func(match):
                nonlocal processed
                processed = True
                return replacement
            refined_annotation = re.sub(original_tag_regex, replace_func, refined_annotation, count=1)

        # 4. Suffix Contraction (A B -> A)
        elif is_contraction and original_text.startswith(correct_text):
            suffix = original_text[len(correct_text):]
            replacement = new_tag + suffix # Suffix becomes plain text after new tag
            def replace_func(match):
                nonlocal processed
                processed = True
                return replacement
            refined_annotation = re.sub(original_tag_regex, replace_func, refined_annotation, count=1)

        # 5. Fallback (Internal change, complex change, or if specific patterns failed)
        if not processed:
            # Default behavior: Replace the original tag with the new tag directly.
            replacement = new_tag
            temp_annotation = re.sub(original_tag_regex, replacement, refined_annotation, count=1)
            if temp_annotation == annotation_before_this_error:
                # Only print warning if the tag wasn't found at all
                 print(f"Warning: Could not find tag for error (or specific pattern failed) #{error_index}: {error_detail}")
            refined_annotation = temp_annotation # Update annotation even if fallback logic was used

    return refined_annotation


def generate_span_error_demos(data_list):
    """
    构建span error的demo
    """
    demos = []

    for idx, sent in enumerate(data_list):
        text = sent['text']
        span_errors = sent['errors']['span']

        # Format as one demo
        one_demo = []
        # one_demo.append(f"Example {idx + 1}:")
        # one_demo.append(f"Text: {text}")
        sent['response'] = sent['response'].replace("Output: ", "")
        one_demo.append(f"Initial Annotated Text: {sent['response']}")
        # Build the refined annotation
        refined_annotation = build_span_error_example(sent['response'], span_errors)
        one_demo.append(f"Refined Annotated Text: {refined_annotation}")

        demos.append("\n".join(one_demo))

    return demos

def span_refiner(model, dataset, demo_num=100, suffix="", retrieval=False):
    """
    用于refine标注数据的span errors
    """
    # 获取数据
    data = load_json(f"./output/aug/{dataset}/{model}-demo100-ann-dev.json")
    label_map = LABEL_DIC[dataset] # mapping the label to the specific domain for LLMs
    labels = [label for label in label_map.values()]
    # 获取demo数据
    demo_data = load_json(f"./datasets/aug/{dataset}/{model}-0-shots-ICL-errors.json")
    # 构建retrieval
    demo_retriver = feed_BM25(demo_data, dataset)
    # 获取type definitions
    with open(f'./datasets/aug/{dataset}/type_definition_typing.txt', 'r') as f:
        type_definitions = f.read()
    # 构建和获取输出目录
    output_dir = f'./output/aug/{dataset}/refiner'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}-demo100-ann-dev{suffix}-span-refiner-demo{demo_num}.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        data = load_json(output_file)
    
    # 定义LLM
    system_prompt = SPAN_SYSTEM_PROMPT
    llm = LLM(model=model, system_prompt=system_prompt)

    # === get refine demo_str ===
    demo_list = generate_span_error_demos(demo_data)
    demo_str = "\n\n".join(demo_list)


    for i in trange(len(data)):
        # check if the test data has been processed
        if 'span_refined_response' in data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = data[i]['text']
        # 构建demos，如果retrieval为True，则使用retrieval
        if retrieval:
            demos = get_retrieved_examples_from_crossNER(retriever=demo_retriver, query=input_text, dataset=demo_data, topN=demo_num)
            demo_list = generate_span_error_demos(demos)
            demo_str = "\n\n".join(demo_list)
        # remove the "Output: " prefix from the response
        data[i]['response'] = data[i]['response'].replace("Output: ", "")
        # 组装prompt
        prompt = SPAN_ERROR_PROMPT.format(
                                       examples=demo_str,
                                       input_annotated_text=data[i]['response'],
                                       )
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.call(prompt=prompt)
                # 清除可能多有的前缀
                response = response.replace("Refined Annotated Text:", "")
                # 清除掉多余的前后空格
                response = response.strip()
                data[i]['span_refined_response'] = response
                break
            except Exception as e:
                print(f"API 调用失败，重试次数: {attempt+1}/{max_retries}，错误信息: {e}")
                # 可根据需要添加 sleep 或其它逻辑
                time.sleep(1)
        # 关键：在每次调用完 llm_call 后就立刻保存
        save_json(data, output_file)
        # NOTE: Debug时用
        # print(system_prompt)
        # print(prompt)
        # print(response)
        # exit()
    save_json(data, output_file)


def generate_spurious_entity_demos(data_list):
    """
    构建spurious entity error的demo
    """
    demos = []

    for idx, sent in enumerate(data_list):
        text = sent['text']
        spurious_errors = sent['errors']['spurious']
        # 两种可能，一种是直接返回spurious_errors，一种是返回None
        if len(spurious_errors) == 0:
            spurious_errors = "None"
        sent['response'] = sent['response'].replace("Output: ", "")
        pred_ents = extract_entities(sent['response'])
        sent['pred'] = pred_ents
        # Format as one demo
        one_demo = []
        one_demo.append(f"Text: {text}")
        one_demo.append(f"Entities: {sent['pred']}")
        # Build the refined annotation
        one_demo.append(f"Spurious Entities: {spurious_errors}")
        demos.append("\n".join(one_demo))
    return demos

def spurious_refiner(model, data_file, dataset, pre_refiner="",demo_num=100, suffix="", retrieval=False):
    """
    用于refine标注数据的spurious errors
    """
    # 获取数据
    data = load_json(data_file)

    label_map = LABEL_DIC[dataset] # mapping the label to the specific domain for LLMs
    labels = [label for label in label_map.values()]
    # 获取demo数据
    demo_data = load_json(f"./datasets/aug/{dataset}/refiner-demos-errors.json")
    # 构建retrieval
    demo_retriver = feed_BM25(demo_data, dataset)
    # 获取type definitions
    with open(f'./datasets/aug/{dataset}/type_definitions.txt', 'r') as f:
        type_definitions = f.read()
    # 构建和获取输出目录
    output_dir = f'./output/aug/{dataset}/refiner'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}-dev{pre_refiner}-filter-spurious-refiner-demo{demo_num}-1.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        data = load_json(output_file)
    
    # 定义LLM
    system_prompt = SPURIOUS_SYSTEM_PROMPT
    llm = LLM(model=model, system_prompt=system_prompt)

    # === get refine demo_str ===
    demo_list = generate_spurious_entity_demos(demo_data)
    demo_str = "\n\n".join(demo_list)


    for i in trange(len(data)):
        # 如果数据的refine是False，则跳过
        if data[i]['refine'] == False:
            continue
        # check if the test data has been processed
        if 'spurious_refined_response' in data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = data[i]['text']
        # remove the "Output: " prefix from the response
        if pre_refiner:
            data[i]['type_refined_response'] = data[i]['type_refined_response'].replace("Refined Entity List: ", "")
            # remove white spaces at beginning and end
            data[i]['type_refined_response'] = data[i]['type_refined_response'].strip()
            pred_ents = ast.literal_eval(data[i]['type_refined_response'])
            data[i]['type_refineed_pred'] = pred_ents
        else:
            data[i]['response'] = data[i]['response'].replace("Output: ", "")
            pred_ents = extract_entities(data[i]['response'])
            data[i]['pred'] = pred_ents
        # 构建demos
        # if retrieval:
        #     demos = get_retrieved_examples_from_crossNER(retriever=demo_retriver, query=input_text, dataset=demo_data, topN=demo_num)
        #     demo_list = generate_spurious_entity_demos(demos)
        #     demo_str = gen_pos_examples(demo_list)
        # 组装prompt
        prompt = SPURIOUS_ERROR_PROMPT.format(
                                       type_definitions=type_definitions,
                                       examples=demo_str,
                                       input_text=input_text,
                                       input_entities=pred_ents,
                                       )
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = llm.call(prompt=prompt)
                # 清除可能多有的前缀
                response = response.replace("Spurious Entities:", "")
                # 清除掉多余的前后空格
                response = response.strip()
                data[i]['spurious_refined_response'] = response
                break
            except Exception as e:
                print(f"API 调用失败，重试次数: {attempt+1}/{max_retries}，错误信息: {e}")
                # 可根据需要添加 sleep 或其它逻辑
                time.sleep(1)
        # 关键：在每次调用完 llm_call 后就立刻保存
        save_json(data, output_file)
        # NOTE: Debug时用
        # print(system_prompt)
        # print("----------------------")
        # print(prompt)
        # print(response)
        # exit()
    save_json(data, output_file)
    return output_file


def processing_spurious_refiner(data_file):
    """
    处理spurious_refiner的输出
    """
    data = load_json(data_file)
    for i in range(len(data)):
        pred = extract_entities(data[i]['response'])
        data[i]['A1_pred'] = pred
        if data[i]['refine']:
            # 如果refine为True，则需要处理spurious_refined_response
            # remove the prefix from the response
            data[i]['spurious_refined_response'] = data[i]['spurious_refined_response'].replace("Spurious Entities:", "")
            # remove white spaces at beginning and end
            data[i]['spurious_refined_response'] = data[i]['spurious_refined_response'].strip()
            # convert to list of tuples
            if data[i]['spurious_refined_response'] != "None" and data[i]['spurious_refined_response'] != None and data[i]['spurious_refined_response'] != "" and data[i]['spurious_refined_response'] != "[]":
                data[i]['spurious_refined_response'] = parse_llm_list_string_fix_first(data[i]['spurious_refined_response'])
                spurious_ents = data[i]['spurious_refined_response']
                # update the pred, remove spurious entities from the pred list
                for spurious_ent in spurious_ents:
                    if spurious_ent in pred:
                        pred.remove(spurious_ent)
        # update the pred
        data[i]['pred'] = pred
    # save the data
    save_json(data, data_file)

def compute_confidence_scores(data_file, threshold=0.25):
    """
    计算confidence scores
    """
    data = load_json(data_file)
    for i in range(len(data)):
        seq_logits = data[i]['logits']
        entity_probs, avg_prob = compute_entity_tag_probs(seq_logits)
        data[i]['entity_probs'] = entity_probs
        data[i]['avg_prob'] = avg_prob
    # 用avg_prob对所有数据进行排序
    data.sort(key=lambda x: x['avg_prob'], reverse=True)
    # label the top 33% confidence scores as easy, then do not need refine
    for i in range(len(data)):
        if i < len(data) * threshold:
            data[i]['refine'] = False
        else:
            data[i]['refine'] = True
    # shuffle the data
    random.shuffle(data)
    # save the data
    save_json(data, data_file)
    return data

def processing_missing_refiner(data_file):
    """
    处理missing_refiner的输出
    """
    data = load_json(data_file)
    for i in range(len(data)):
        # initial prediction
        pred = extract_entities(data[i]['response'])
        data[i]['A1_pred'] = pred
        pred = data[i]['pred']
        if data[i]['refine']:
            # 如果refine为True，则需要处理spurious_refined_response
            # remove the prefix from the response
            data[i]['missing_refined_response'] = data[i]['missing_refined_response'].replace("Missing Entities:", "")
            # remove white spaces at beginning and end
            data[i]['missing_refined_response'] = data[i]['missing_refined_response'].strip()
            # convert to list of tuples
            if data[i]['missing_refined_response'] != "None" and data[i]['missing_refined_response'] != None and data[i]['missing_refined_response'] != "" and data[i]['missing_refined_response'] != "[]":
                data[i]['missing_refined_response'] = parse_llm_list_string_fix_first(data[i]['missing_refined_response'])
                missing_ents = data[i]['missing_refined_response']
                # update the pred, remove missing entities from the pred list
                for missing_ent in missing_ents:
                    if missing_ent not in pred:
                        # add the missing entity to the pred list
                        pred.append(missing_ent)
        # update the pred
        data[i]['pred'] = pred
    # save the data
    save_json(data, data_file)
    return data


def main():
    """
    Step 1. Annotator: 用Seed数据来标注unlabel数据集
    Step 2. Refiner: 用Seed数据来judge 标注的数据集
        2.1 Missing (增加了Recall)
        2.2 Spurious (增加了Precision)
        2.3 Type (增加了Precision)
    """

    # 构建个argparse
    parser = argparse.ArgumentParser(description='Run the low-resource NER with few-shot ICL.')   
    parser.add_argument('--dataset', type=str, default='ai', help='The dataset to use.')
    parser.add_argument('--model', type=str, default='deepseek-chat', help='The model to use.', choices=['deepseek-chat', 'Qwen2.5-72B', 'gpt-4o', 'gemini'])
    parser.add_argument('--demo_num', type=int, default=100, help='The number of demo examples to use.')

    args = parser.parse_args()

    dataset = args.dataset
    # model = "deepseek-chat" # "Qwen2.5-72B", "gpt-4o", "deepseek-chat", ""
    model = args.model
    demo_num = args.demo_num

    # 获取数据
    # data, labels = get_seed_data(dataset=dataset, subset='seed', style='tagging')

    # run_icl(model, dataset)


    # ===> 1. Annotator: 用Seed数据来标注unlabel数据集
    # ann_data, ann_data_file = main_annotator(model, dataset, demo_num=100, logits=True, name_suffix="")

    # ====> 1.1 计算confidence分数，给confidence分数排序，将top 33%的confidence分数直接做easy数据，不计入后续的refine
    ann_data_file = f"./output/aug/{dataset}/{model}-demo100-ann-dev-logits-errors.json"
    print(f"Computing confidence scores for {ann_data_file}")
    # ann_data = load_json(ann_data_file)
    # compute_confidence_scores(ann_data_file, 0.5)

    # ===> 2. Error-aware refinment
    # Note: 测试分别跑spurious, missing, type的refine
    # 1. 分别跑spurious, missing, type的refine
        # 1.1 跑spurious的refine
    # ann_data_file = spurious_refiner(model, ann_data_file, dataset, suffix="")
    # ann_data_file = f"output/aug/ai/refiner/{model}-dev-filter-spurious-refiner-demo100.json"
    # 处理完spurious的refine后，把pred更新一下
    print(f"Processing spurious refiner results for {ann_data_file}")
    # processing_spurious_refiner(ann_data_file)
    #     # 1.2 跑missing的refine
    # ann_data_file = missing_refiner(model, ann_data_file, dataset, suffix="")
    # 处理完missing的refine后，把pred更新一下
    ann_data_file = f"output/aug/ai/refiner/{model}/{model}-demo100-dev-filter-suprious-missing-refiner.json"
    print(f"Processing missing refiner results for {ann_data_file}")
    # ann_data_file = processing_missing_refiner(ann_data_file)
    # #     # 1.3 跑type的refine
    data = load_json(ann_data_file)
    ann_data_file = type_refiner(model, ann_data_file, dataset)

    # 2. 缝合起来跑Spurious → Missing → Type


if __name__ == '__main__':
    main()
    
    