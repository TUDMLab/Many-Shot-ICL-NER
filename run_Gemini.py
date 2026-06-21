# import google.generativeai as genai
from typing import Dict, Any, Tuple, Optional, List
from openai import OpenAI
from const import AI_CLASSS, SCIENCE_CLASS, LITERATURE_CLASSS, MUSIC_CLASSS, POLITICS_CLASSS
from prompts import ONE_STAGE_FEW_SHOT_PROMPTS, ONE_STAGE_TAG_FEW_SHOT_PROMPTS, ZERO_SHOT_TAG_PROMPT
from dataloader import get_crossNER_data, get_seed_data, feed_BM25, get_retrieved_examples
from tqdm import tqdm, trange
import os, time
from utils import extract_entities, save_json_as_csv, load_json, get_ners, save_json
import random
import argparse
import re
import ast
import pdb

from google import genai
from google.genai import types


class GeminiLLM:
    """
    A class to interact with Google Gemini models, adapted from an OpenAI-style LLM class.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash", api_key: Optional[str] = None, system_prompt: str = "", max_tokens: int = 8192):
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.client = genai.Client(api_key="YOUR_GEMINI_API_KEY") # YOUR_GEMINI_API_KEY, YOUR_GEMINI_API_KEY, YOUR_GEMINI_API_KEY

    def call(self, prompt: str) -> str:
        try:
            response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=[prompt],
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_prompt,
                        max_output_tokens=self.max_tokens,
                        temperature=1.0,
                    )
                )
            
            # Detailed error checking based on Gemini's response structure
            if not response.candidates:
                block_reason_msg = "Unknown reason"
                if response.prompt_feedback and response.prompt_feedback.block_reason:
                    block_reason_msg = response.prompt_feedback.block_reason.name
                raise ValueError(
                    f"The response was empty or blocked. Reason: {block_reason_msg}. "
                    f"Prompt feedback: {response.prompt_feedback}"
                )

            # Ensure there's text in the primary candidate
            # (Gemini typically returns one candidate for non-streaming requests)
            candidate = response.candidates[0]
            if not candidate.content or not candidate.content.parts or not candidate.content.parts[0].text:
                finish_reason_msg = candidate.finish_reason.name if candidate.finish_reason else "UNKNOWN"
                safety_ratings_msg = str(candidate.safety_ratings) if candidate.safety_ratings else "N/A"
                # Check if text is directly available (newer versions of the SDK might populate response.text directly)
                if response.text: # Prioritize response.text if available
                    return response.text

                raise ValueError(
                    f"The model generated an empty text response. "
                    f"Finish reason: {finish_reason_msg}. Safety ratings: {safety_ratings_msg}"
                )
            return response.text

        except Exception as e:
            print(f"Error during Gemini API call: {e}")
            # Re-raise the exception to be handled by the caller
            raise



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
    SPURIOUS_ERROR_PROMPT
)


# ======== LLM Class ========
API_DIC = {
    "deepseek-chat": "YOUR_API_KEY", # "YOUR_API_KEY", # "YOUR_API_KEY", "YOUR_API_KEY", YOUR_API_KEY, YOUR_API_KEY
    "Qwen2.5-72B": "EMPTY",
    "gpt-4o": "YOUR_OPENAI_API_KEY",
    "gemini": "YOUR_GEMINI_API_KEY"
}

BASE_URL = {
    "deepseek-chat": "https://api.deepseek.com",
    "Qwen2.5-72B": "http://localhost:8000/v1",
    "gpt-4o": "https://api.openai.com/v1"
}


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


def main_annotator(model, dataset, demo_num = 100, logits=False, name_suffix="", batch_num=0):
    """
    Annotator: 用Seed数据来标注unlabel数据集
        Many-Shot ICL 来标注unlabel数据集
    """
    
    # 获取测试/要标注的数据
    unlabel_data = load_json(f'./unlabeled/{dataset}/sampling/sample_{batch_num}.json')
    # NOTE: 用来测试的部分====
    # unlabel_data, labels = get_crossNER_data(dataset=dataset, subset='test', style='tagging')

    # 获取Positive数据和构建pos_retriever
    demo_data, labels = get_seed_data(dataset=dataset, subset='seed', style='tagging')
    # print(labels)
    # exit()
    retriever = feed_BM25(demo_data, dataset)

    # Read prompt
    # annotation guideline 1
    # with open(f'./datasets/aug/{dataset}/annotation_guideline_refined.txt', 'r') as f:
    #     task_prompt = f.read()
    # annotation guideline 2
    with open(f'./datasets/aug/{dataset}/annotator_prompt.txt', 'r') as f:
        task_prompt = f.read()


    # 定义LLM
    system_prompt = """You are a Named Entity Recognition (NER) annotator specialized in {} domain texts. Your task is to identify and tag entities in the given text EXACTLY as follows:
    - Wrap each entity with `<entity type="[TYPE]">...</entity>` tags, where `[TYPE]` must be one of the provided entity types.  
    - Preserve all non-entity text unchanged.""".format(dataset)
    print(system_prompt)
    # llm = LLM(model=model, system_prompt=system_prompt)
    llm = GeminiLLM(
            model_name="gemini-2.5-flash-preview-04-17", # Using a generally available model
            api_key="YOUR_GEMINI_API_KEY", # Replace with your actual API key
            system_prompt=system_prompt,
        )
    print("===> LLM initialized.")
    print("Model: ", llm.model_name)

    

    # 如果输出目录不存在，则创建
    output_dir = f'./output/augment/{dataset}/{model}'
    os.makedirs(output_dir, exist_ok=True)
    if logits:
        output_file = f'{output_dir}/batch-{batch_num}-logits{name_suffix}.json'
    else:
        output_file = f'{output_dir}/batch-{batch_num}{name_suffix}.json'
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
            demo_str = gen_examples(demo_data, shots=demo_num)
        else:
            topN_pos_examples = get_retrieved_examples(retriever=retriever, query=input_text, dataset=demo_data, topN=demo_num)
            demo_str = gen_examples(topN_pos_examples, shots=demo_num)
        
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
        # NOTE: debug时用, 测试记得删掉
        # print("-------")
        # print(prompt)
        # print(response)
        # exit()
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



def type_refiner(model, dataset, demo_num=100, logits=False, suffix="", retrieval=False):
    """
    用于refine标注数据的entity types
    """
    # 获取数据
    data = load_json(f"./output/aug/{dataset}/{model}-demo100-ann-dev.json")
    label_map = LABEL_DIC[dataset] # mapping the label to the specific domain for LLMs
    labels = [label for label in label_map.values()]
    # 获取demo数据
    demo_data = load_json(f"./datasets/aug/{dataset}/{model}-0-shots-ICL-errors.json")
    if retrieval:
        demo_retriver = feed_BM25(demo_data, dataset)
    # 获取type definitions
    with open(f'./datasets/aug/{dataset}/type_definition_typing.txt', 'r') as f:
        type_definitions = f.read()
    # 构建和获取输出目录
    output_dir = f'./output/aug/{dataset}/refiner'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}-demo100{retrieval}-ann-dev-type-refiner{suffix}.json'
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
    # demo_str = "\n\n## Examples\n" + demo_str

    for i in trange(len(data)):
        # check if the test data has been processed
        if 'type_refined_response' in data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = data[i]['text']
        # remove the "Output: " prefix from the response
        data[i]['response'] = data[i]['response'].replace("Output: ", "")
        data[i]['pred'] = extract_entities(data[i]['response'])
        # 组装prompt
        if retrieval:
            topN_examples = get_retrieved_examples(retriever=demo_retriver, query=input_text, dataset=demo_data, topN=demo_num)
            demo_list = generate_type_error_demos(topN_examples)
            demo_str = gen_pos_examples(demo_list)
        prompt = TYPE_ERROR_PROMPT.format(
                                       labels=labels, 
                                       type_definitions=type_definitions,
                                       examples=demo_str,
                                       input_text=input_text,
                                       input_entity_list=data[i]['pred'],
                                       )
        # print(prompt)
        # exit()

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
    save_json(data, output_file)



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

        # Format as one demo
        one_demo = []

        one_demo.append(f"Text: {sent['text']}")
        sent['response'] = sent['response'].replace("Output: ", "")
        pred_ents = extract_entities(sent['response'])
        sent['pred'] = pred_ents
        one_demo.append(f"Entitites: {pred_ents}")
        one_demo.append(f"Missing Entitites: {missing_errors}")

        demos.append("\n".join(one_demo))

    return demos

def missing_refiner(model, dataset, demo_num=100, suffix="", retrieval=False):
    """
    用于refine标注数据的mising entities
    """
    # 获取数据
    data = load_json(f"./output/aug/{dataset}/{model}-demo100-ann-dev.json")
    label_map = LABEL_DIC[dataset] # mapping the label to the specific domain for LLMs
    labels = [label for label in label_map.values()]
    # 获取demo数据
    demo_data = load_json(f"./datasets/aug/{dataset}/{model}-0-shots-ICL-errors.json")
    if retrieval:
        demo_retriver = feed_BM25(demo_data, dataset)
    # 获取type definitions
    with open(f'./datasets/aug/{dataset}/type_definition_typing.txt', 'r') as f:
        type_definitions = f.read()
    # 构建和获取输出目录
    output_dir = f'./output/aug/{dataset}/refiner'
    os.makedirs(output_dir, exist_ok=True)
    output_file = f'{output_dir}/{model}-demo100--ann-dev{suffix}-missing-refiner-new-{demo_num}.json'
    # 如果文件已存在，则更新test_data为已处理的部分
    if os.path.exists(output_file):
        print(f"Loading existing output file: {output_file}")
        data = load_json(output_file)
    
    # 定义LLM
    system_prompt = MISSING_SYSTEM_PROMPT
    llm = LLM(model=model, system_prompt=system_prompt)

    # === get refine demo_str ===
    demo_list = generate_missing_entity_demos(demo_data)
    # demo_list = generate_missing_entity_demos_list(demo_data) # NOTE: 新版本测试
    demo_str = "\n\n".join(demo_list)


    for i in trange(len(data)):
        # check if the test data has been processed
        if 'missing_refined_response' in data[i]:
            print(f"Skipping test data {i} as it has already been processed.")
            continue
        input_text = data[i]['text']
        if retrieval:
            topN_demos = get_retrieved_examples(retriever=demo_retriver, query=input_text, dataset=demo_data, topN=demo_num)
            demo_list = generate_missing_entity_demos(topN_demos)
            demo_str = "\n\n".join(demo_list)
        # remove the "Output: " prefix from the response
        data[i]['response'] = data[i]['response'].replace("Output: ", "")
        data[i]['pred'] = extract_entities(data[i]['response'])
        # 组装prompt
        prompt = MISSING_ERROR_PROMPT.format(
                                       labels=labels, 
                                       type_definitions=type_definitions,
                                       examples=demo_str,
                                       input_annotated_text=data[i]['response'],
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
        # print("----------")
        # print(response)
        # exit()

    save_json(data, output_file)


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
            demos = get_retrieved_examples(retriever=demo_retriver, query=input_text, dataset=demo_data, topN=demo_num)
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

def spurious_refiner(model, dataset, pre_refiner="",demo_num=100, suffix="", retrieval=False):
    """
    用于refine标注数据的spurious errors
    """
    # 获取数据
    if pre_refiner:
        print("Using pre-refiner data: ", pre_refiner)
    
        data = load_json(f"./output/aug/{dataset}/refiner/{model}-demo100-ann-dev{pre_refiner}.json")
    else:
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
    output_file = f'{output_dir}/{model}-demo100-ann-dev{pre_refiner}-spurious-refiner-demo{demo_num}.json'
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
                                       labels=labels, 
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


def main():
    # 构建个argparse
    parser = argparse.ArgumentParser(description='Run the low-resource NER with few-shot ICL.')   
    parser.add_argument('--dataset', type=str, default='ai', help='The dataset to use.')
    parser.add_argument('--model', type=str, default='gemini-2.5-flash', help='The model to use.', choices=['deepseek-chat', 'Qwen2.5-72B', 'gpt-4o', 'gemini'])
    parser.add_argument('--demo_num', type=int, default=100, help='The number of demo examples to use.')
    parser.add_argument('--batch_num', type=int, default=0, help='The batch number to use.')
    

    args = parser.parse_args()

    dataset = args.dataset
    model = args.model
    demo_num = args.demo_num
    batch_num = args.batch_num

    # ===> 1. Annotator: 用Seed数据来标注unlabel数据

    main_annotator(model, "ai", demo_num=demo_num, logits=False, name_suffix="test", batch_num=1)
    # main_annotator(model, "ai", demo_num=demo_num, logits=False, name_suffix="", batch_num=2)
    # main_annotator(model, "ai", demo_num=demo_num, logits=False, name_suffix="", batch_num=3)
    # main_annotator(model, "ai", demo_num=demo_num, logits=False, name_suffix="", batch_num=4)
    # main_annotator(model, "ai", demo_num=demo_num, logits=False, name_suffix="", batch_num=5)
    # main_annotator(model, "literature", demo_num=demo_num, logits=False, name_suffix="", batch_num=0)
    # main_annotator(model, "literature", demo_num=demo_num, logits=False, name_suffix="", batch_num=1)
    # main_annotator(model, "literature", demo_num=demo_num, logits=False, name_suffix="", batch_num=2)
    # main_annotator(model, "literature", demo_num=demo_num, logits=False, name_suffix="", batch_num=3)
    # main_annotator(model, "literature", demo_num=demo_num, logits=False, name_suffix="", batch_num=4)
    # main_annotator(model, "literature", demo_num=demo_num, logits=False, name_suffix="", batch_num=5)
    
    # main_annotator(model, "music", demo_num=demo_num, logits=False, name_suffix="", batch_num=0)
    # main_annotator(model, "music", demo_num=demo_num, logits=False, name_suffix="", batch_num=1)
    # main_annotator(model, "music", demo_num=demo_num, logits=False, name_suffix="", batch_num=2)
    # main_annotator(model, "music", demo_num=demo_num, logits=False, name_suffix="", batch_num=3)
    # main_annotator(model, "music", demo_num=demo_num, logits=False, name_suffix="", batch_num=4)
    # main_annotator(model, "music", demo_num=demo_num, logits=False, name_suffix="", batch_num=5)
    # main_annotator(model, "politics", demo_num=demo_num, logits=False, name_suffix="", batch_num=0)
    # main_annotator(model, "politics", demo_num=demo_num, logits=False, name_suffix="", batch_num=1)
    # main_annotator(model, "politics", demo_num=demo_num, logits=False, name_suffix="", batch_num=2)
    # main_annotator(model, "politics", demo_num=demo_num, logits=False, name_suffix="", batch_num=3)
    # main_annotator(model, "politics", demo_num=demo_num, logits=False, name_suffix="", batch_num=4)
    # main_annotator(model, "politics", demo_num=demo_num, logits=False, name_suffix="", batch_num=5)
    # main_annotator(model, "science", demo_num=demo_num, logits=False, name_suffix="", batch_num=0)
    # main_annotator(model, "science", demo_num=demo_num, logits=False, name_suffix="", batch_num=1)
    # main_annotator(model, "science", demo_num=demo_num, logits=False, name_suffix="", batch_num=2)
    # main_annotator(model, "science", demo_num=demo_num, logits=False, name_suffix="", batch_num=3)
    # main_annotator(model, "science", demo_num=demo_num, logits=False, name_suffix="", batch_num=4)
    # main_annotator(model, "science", demo_num=demo_num, logits=False, name_suffix="", batch_num=5)

    


    # main_annotator(model, dataset, pos_num=demo_num, logits=True, name_suffix="-logits2")

    # ====> 1.5 Confidence Score Filtering
    # confidence_score_filtering(model, dataset, demo_num=demo_num, logits=True, name_suffix="")

    # ===> 2. Judger/Refiner: 用Seed数据来judge 标注的数据集
    # NOTE: 2.1: Type Error的refine
    # type_refiner(model, dataset, demo_num=demo_num, logits=False, suffix="new-definition")

   

    # NOTE: 2.3: Span Error的refine
    # span_refiner(model, dataset, demo_num=100, suffix="-simple")
    # span_refiner(model, dataset, demo_num=50, suffix="-simple")
    # span_refiner(model, dataset, demo_num=10, suffix="-simple")



    # NOTE: 2.2: Missing Entity Error的refine
    # missing_refiner(model, dataset, demo_num=demo_num, suffix="")
    # missing_refiner(model, dataset, demo_num=50, suffix="", retrieval=True)
    # missing_refiner(model, dataset, demo_num=10, suffix="", retrieval=True)

    # NOTE: 2.4: Spurious Entity Error的refine
    # spurious_refiner(model, dataset, pre_refiner="-type-refiner",demo_num=demo_num, suffix="")



if __name__ == '__main__':
    main()

#     try:
#         gemini_llm = GeminiLLM(
#             model_name="gemini-2.5-pro-preview-05-06", # Using a generally available model
#             api_key="YOUR_GEMINI_API_KEY", # Replace with your actual API key
#             system_prompt="You are a concise and factual assistant.",
#             max_tokens=150
#         )

#         # Example 1: Standard call
#         prompt1 = "What is the capital of France?"
#         print(f"\nUser Prompt 1: {prompt1}")
#         response1 = gemini_llm.call(prompt1)
#         print(f"Gemini Response 1: {response1}")

#         # Example 2: Call that would attempt to get logits (returns None for logits part)
#         prompt2 = "Explain the concept of photosynthesis in one sentence."
#         print(f"\nUser Prompt 2: {prompt2}")
#         response2_text, logits2 = gemini_llm.call_with_logits(prompt2)
#         print(f"Gemini Response 2 (Text): {response2_text}")
#         print(f"Gemini Response 2 (Logits): {logits2}") # Will be None

#     except ValueError as ve:
#         print(f"A ValueError occurred: {ve}")
#     except Exception as e:
#         print(f"An unexpected error occurred: {e}")
#         print("Please ensure your Google API key is correctly configured and has the necessary permissions.")
#         print("You might need to set the GOOGLE_API_KEY environment variable or pass it to the GeminiLLM constructor.")


