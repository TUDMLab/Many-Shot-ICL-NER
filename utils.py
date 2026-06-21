import json
import os
import csv
import re
import collections
from typing import List, Tuple, Dict, Set, Any, Counter as CounterType
import math
import json
import os
import ast


def clean_llm_output(text):
    """清理输入字符串：移除首尾空白和常见的Markdown代码块标记。"""
    if not isinstance(text, str):
        return ""
    # 首先移除首尾空白
    text = text.strip()

    # 处理 ``` 代码块标记 (尝试更可靠地移除)
    # 情况 1: 包裹形式 ```[lang]\n...\n```
    if text.startswith('```') and text.endswith('```'):
        # 移除外层 ```
        text = text[3:-3].strip()
        # 尝试移除可选的语言标记行 (简单判断：如果第一行不像数据且只有单个词)
        if '\n' in text:
            first_line, rest = text.split('\n', 1)
            stripped_first_line = first_line.strip()
            if stripped_first_line and not stripped_first_line.startswith(('[', '{', "'", '"')) and len(stripped_first_line.split()) <= 1:
                text = rest.strip()

    # 情况 2: 只有结尾标记，如 "[]\n```"
    elif text.endswith("\n```"):
        text = text[:-4].rstrip()

    # 再次确保移除首尾空白
    return text.strip()

def parse_llm_list_string_fix_first(llm_output_str):
    """
    优先尝试修复字符串末尾缺失的']'，然后解析整个字符串。

    Args:
        llm_output_str: LLM生成的原始字符串。

    Returns:
        如果解析成功且结果是列表，则返回该列表。
        否则返回空列表 []。
    """
    if not isinstance(llm_output_str, str):
        return []

    cleaned_str = clean_llm_output(llm_output_str)
    if not cleaned_str:
        return []

    # --- 步骤 1 & 2: 检查是否需要修复 ---
    string_to_parse = cleaned_str
    # 检查条件: 以'['开头 并且 '['数量 > ']'数量
    if cleaned_str.startswith('[') and cleaned_str.count('[') > cleaned_str.count(']'):
        # 计算需要补充多少个 ']'
        open_brackets = cleaned_str.count('[')
        close_brackets = cleaned_str.count(']')
        missing_count = open_brackets - close_brackets
        # 构造修复后的字符串
        string_to_parse = cleaned_str + ']' * missing_count
        # print(f"Debug: Fixing applied: {repr(cleaned_str)} -> {repr(string_to_parse)}") # 可选的调试信息

    # else:
        # print(f"Debug: No fixing needed for: {repr(cleaned_str)}") # 可选的调试信息


    # --- 步骤 3: 尝试解析最终的字符串 (可能是原始的，也可能是修复后的) ---
    try:
        parsed_result = ast.literal_eval(string_to_parse)
        # 检查解析结果是否为列表
        if isinstance(parsed_result, list):
            return parsed_result # 成功，返回解析出的列表
        else:
            # 解析成功但不是列表 (例如解析出字符串、数字等)
            return []
    except (SyntaxError, ValueError, TypeError, MemoryError):
        # 解析失败 (原始字符串无效，或修复后仍然无效)
        # print(f"Debug: Parsing failed for: {repr(string_to_parse)}") # 可选的调试信息
        return [] # 返回空列表

def compute_entity_tag_probs(
    seq_logits: List[Dict[str, Any]]
) -> Tuple[List[float], float]:
    """
    计算每个实体标签的概率以及它们的平均概率。

    Args:
        seq_logits: 按生成顺序的 token 列表，每项是 dict，必须包含
                    - 'token': str
                    - 'logprob': float

    Returns:
        entity_probs: 各实体标签的概率列表
        avg_prob:       全部实体标签概率的算术平均
    """
    # 1. 重建全文本并记录每个 token 的字符跨度
    full_text = ""
    spans: List[Tuple[int, int]] = []
    for tok in seq_logits:
        start = len(full_text)
        full_text += tok['token']
        end = len(full_text)
        spans.append((start, end))

    # 2. 正则匹配所有实体标签
    pattern = re.compile(r'<entity[^>]*>.*?</entity>')
    matches = list(pattern.finditer(full_text))
    if not matches:
        # raise ValueError("在生成文本中未找到任何 <entity> 标签。")
        return [], 0.0

    # 3. 针对每个实体标签，累加其对应 token 的 logprob 并转为概率
    entity_probs: List[float] = []
    for m in matches:
        s_char, e_char = m.span()
        # 收集所有与该标签字符范围有重叠的 token 索引
        idxs = [
            i for i, (s, e) in enumerate(spans)
            if not (e <= s_char or s >= e_char)
        ]
        # 累加 logprob
        logp_sum = sum(seq_logits[i]['logprob'] for i in idxs)
        # 转为概率
        prob = math.exp(logp_sum)
        entity_probs.append(prob)

    # 4. 计算平均
    avg_prob = sum(entity_probs) / len(entity_probs)
    return entity_probs, avg_prob

def txt_to_json_list(data_path, save=False):
    """
    将已tokenize的txt文件转换为一个JSON文件，文件名与输入txt文件相同，仅扩展名改为.json。
    每行一个句子，每个句子转换为一个dict对象，包含id, text, tokens字段。
    :param data_path: 输入的txt文件路径，如 'unlabeled/ai/ai_tasklevel.txt'
    :return: 无，结果直接写入JSON文件
    """
    output_path = os.path.splitext(data_path)[0] + ".json"

    data = []
    with open(data_path, 'r', encoding='utf-8') as fin:
        for idx, line in enumerate(fin):
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            entry = {
                "id": idx,
                "text": line,
                "tokens": tokens
            }
            data.append(entry)

    if save:
        with open(output_path, 'w', encoding='utf-8') as fout:
            json.dump(data, fout, ensure_ascii=False, indent=2)
        print(f"Saved to {output_path}")
    return data


def analyze_ner_errors(
    llm_prediction: List[List[str]],
    ground_truth: List[List[str]]
) -> Dict[str, int]:
    """
    Analyzes NER errors for a single data sample by comparing lists of entities.

    Compares LLM predictions to Ground Truth based on exact text matches first,
    then uses substring matching to identify potential span errors for remaining
    entities.

    Error Prioritization Logic:
    1. Exact Match (Text & Type): Counts as 'correct'.
    2. Exact Text, Different Type: Counts as 'type' error.
    3. Substring Text Overlap (for remaining unmatched): Counts as 'span' error
       (this takes priority over type differences in case of overlap).
    4. Unmatched Ground Truth: Counts as 'missing'.
    5. Unmatched LLM Prediction: Counts as 'spurious'.

    Args:
        llm_prediction: A list of lists, where each inner list is
                        [entity_text, entity_type] predicted by the LLM.
        ground_truth: A list of lists, where each inner list is
                      [entity_text, entity_type] from the ground truth.

    Returns:
        A dictionary containing the counts for 'correct', 'type', 'span',
        'missing', 'spurious' errors/matches for this sample.
    """
    results: Dict[str, int] = {'correct': 0, 'type': 0, 'span': 0, 'missing': 0, 'spurious': 0}

    # Use lists directly, tracking indices is robust
    llm_entities: List[Tuple[str, str]] = [tuple(item) for item in llm_prediction]
    gt_entities: List[Tuple[str, str]] = [tuple(item) for item in ground_truth]

    num_gt = len(gt_entities)
    num_llm = len(llm_entities)

    # Keep track of indices that have been matched
    matched_llm_indices: set[int] = set()
    matched_gt_indices: set[int] = set()

    # --- Pass 1: Exact Matches (Identify Correct & Type Errors) ---
    if num_gt > 0 and num_llm > 0: # Optimization: only run if both lists have items
        for i in range(num_gt):
            gt_text, gt_type = gt_entities[i]
            for j in range(num_llm):
                # Skip if LLM entity already matched or GT already matched by an earlier LLM entity
                if j in matched_llm_indices or i in matched_gt_indices:
                    continue

                llm_text, llm_type = llm_entities[j]

                # Check for exact text match
                if gt_text == llm_text:
                    if gt_type == llm_type:
                        # Correct Match
                        results['correct'] += 1
                    else:
                        # Type Error
                        results['type'] += 1

                    # Mark both as matched and break inner loop (one-to-one mapping)
                    matched_gt_indices.add(i)
                    matched_llm_indices.add(j)
                    break

    # --- Pass 2: Approximate/Span Matches (Substring Method for REMAINING entities) ---
    if len(matched_gt_indices) < num_gt and len(matched_llm_indices) < num_llm: # Optimization
        for i in range(num_gt):
            # Skip if already matched exactly
            if i in matched_gt_indices:
                continue
            gt_text, gt_type = gt_entities[i] # Get text/type only if needed

            for j in range(num_llm):
                # Skip if already matched (either exactly or as a span error for a previous GT entity)
                if j in matched_llm_indices:
                    continue

                llm_text, llm_type = llm_entities[j] # Get text/type only if needed

                # --- Span Check Logic (Substring) ---
                # Ensure texts are not empty and not identical (identical handled in Pass 1)
                if gt_text and llm_text and (gt_text in llm_text or llm_text in gt_text):
                    # Found overlapping text for unmatched entities. Classify as Span Error.
                    results['span'] += 1
                    matched_gt_indices.add(i)
                    matched_llm_indices.add(j)
                    # Break inner loop: Assume one span match per GT entity in this pass
                    break

    # --- Pass 3: Identify Missing and Spurious (from final set of unmatched entities) ---
    # Count unmatched GT entities as Missing
    for i in range(num_gt):
        if i not in matched_gt_indices:
            results['missing'] += 1

    # Count unmatched LLM entities as Spurious
    for j in range(num_llm):
        if j not in matched_llm_indices:
            results['spurious'] += 1

    return results



def get_present_ner_error_types(
    llm_prediction: List[List[str]],
    ground_truth: List[List[str]]
) -> Set[str]:
    """
    Determines the types of NER errors present in a single data sample.

    Compares LLM predictions to Ground Truth based on exact text matches first,
    then uses substring matching to identify potential span errors for remaining
    entities. Returns a set containing the names of all error/match types
    found in the comparison.

    Error Prioritization Logic (used for classification before adding to set):
    1. Exact Match (Text & Type): Identifies 'correct'.
    2. Exact Text, Different Type: Identifies 'type' error.
    3. Substring Text Overlap (for remaining unmatched): Identifies 'span' error
       (this takes priority over type differences in case of overlap).
    4. Unmatched Ground Truth: Identifies 'missing'.
    5. Unmatched LLM Prediction: Identifies 'spurious'.

    Args:
        llm_prediction: A list of lists, where each inner list is
                        [entity_text, entity_type] predicted by the LLM.
        ground_truth: A list of lists, where each inner list is
                      [entity_text, entity_type] from the ground truth.

    Returns:
        A set containing strings representing the types of errors/matches
        found (e.g., {'correct', 'missing', 'span'}). Possible values in the
        set are: 'correct', 'type', 'span', 'missing', 'spurious'.
    """
    # Initialize a set to store the types of errors/matches found
    present_error_types: Set[str] = set()

    # Use lists of tuples for easier comparison
    llm_entities: List[Tuple[str, str]] = [tuple(item) for item in llm_prediction]
    gt_entities: List[Tuple[str, str]] = [tuple(item) for item in ground_truth]

    num_gt = len(gt_entities)
    num_llm = len(llm_entities)

    # Keep track of indices that have been matched
    matched_llm_indices: set[int] = set()
    matched_gt_indices: set[int] = set()

    # --- Pass 1: Exact Matches (Identify Correct & Type Errors) ---
    if num_gt > 0 and num_llm > 0: # Optimization
        for i in range(num_gt):
            gt_text, gt_type = gt_entities[i]
            for j in range(num_llm):
                # Skip if LLM entity already matched or GT already matched by an earlier LLM entity
                if j in matched_llm_indices or i in matched_gt_indices:
                    continue

                llm_text, llm_type = llm_entities[j]

                # Check for exact text match
                if gt_text == llm_text:
                    if gt_type == llm_type:
                        # Found a 'correct' match
                        present_error_types.add('correct')
                    else:
                        # Found a 'type' error
                        present_error_types.add('type')

                    # Mark both as matched and break inner loop (one-to-one mapping)
                    matched_gt_indices.add(i)
                    matched_llm_indices.add(j)
                    break # Move to the next GT entity

    # --- Pass 2: Approximate/Span Matches (Substring Method for REMAINING entities) ---
    # Only run if there are unmatched entities remaining on both sides
    if len(matched_gt_indices) < num_gt and len(matched_llm_indices) < num_llm:
        for i in range(num_gt):
            # Skip if already matched exactly
            if i in matched_gt_indices:
                continue
            gt_text, gt_type = gt_entities[i] # Get text/type only if needed

            for j in range(num_llm):
                # Skip if already matched (either exactly or as a span error for a previous GT entity)
                if j in matched_llm_indices:
                    continue

                llm_text, llm_type = llm_entities[j] # Get text/type only if needed

                # --- Span Check Logic (Substring) ---
                # Ensure texts are not empty and not identical (identical handled in Pass 1)
                if gt_text and llm_text and (gt_text in llm_text or llm_text in gt_text):
                    # Found overlapping text for unmatched entities. Classify as Span Error.
                    present_error_types.add('span')
                    matched_gt_indices.add(i)
                    matched_llm_indices.add(j)
                    # Break inner loop: Assume one span match per GT entity in this pass
                    break # Move to the next GT entity

    # --- Pass 3: Identify Missing and Spurious (from final set of unmatched entities) ---
    # Check for any unmatched GT entities (Missing)
    found_missing = False
    for i in range(num_gt):
        if i not in matched_gt_indices:
            present_error_types.add('missing')
            found_missing = True
            # Optimization: We only need to find one instance to add 'missing' to the set
            # break # Uncomment if you ONLY want to know IF 'missing' exists at all,
                  # but the current structure correctly handles adding it once.

    # Check for any unmatched LLM entities (Spurious)
    found_spurious = False
    for j in range(num_llm):
        if j not in matched_llm_indices:
            present_error_types.add('spurious')
            found_spurious = True
            # Optimization: Similar to 'missing'
            # break # Uncomment if you ONLY want to know IF 'spurious' exists at all.

    return present_error_types



def extract_entities(annotated_text: str):
    """
    Extracts the mention and type from <entity type="some_type">Mention text</entity> annotations.

    Args:
        annotated_text (str): A string containing annotations in the format:
                             <entity type="some_type">entity mention</entity>

    Returns:
        list: A list of [mention, type] pairs.
    """
    # The regex pattern captures two groups:
    # 1. The value of the type attribute (group 1)
    # 2. The text inside the <entity>...</entity> (group 2)
    pattern = r'<entity type="([^"]+)">(.*?)</entity>'

    # re.findall(pattern, text) will return a list of tuples like:
    # [("algorithm", "naive Bayes classifier"), ("algorithm", "Gaussian mixture model"), ...]
    matches = re.findall(pattern, annotated_text)

    # Convert each tuple from (etype, mention) to [mention, etype]
    entities = [[mention, etype] for etype, mention in matches]
    return entities

def save_json_as_csv(json_data, csv_file_path):
    """
    Saves JSON data as a CSV file.

    Args:
    json_data (str or list): JSON string or list of dictionaries.
    csv_file_path (str): Path to the output CSV file.
    """
    # If the input is a JSON string, parse it to a Python list
    if isinstance(json_data, str):
        json_data = json.loads(json_data)

    # Ensure the data is a list of dictionaries
    if not isinstance(json_data, list) or not all(isinstance(item, dict) for item in json_data):
        raise ValueError("Input JSON data should be a list of dictionaries.")

    # Get the header from the keys of the first dictionary
    header = json_data[0].keys()

    # Write the data to a CSV file
    with open(csv_file_path, 'w', newline='', encoding='utf-8') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=header)
        writer.writeheader()
        writer.writerows(json_data)

def save_jsonl(data, save_path):
    with open(save_path, 'w') as f:
        for line in data:
            json.dump(line, f)
            f.write('\n')
def load_jsonl(jsonl_path):
    data = []
    with open(jsonl_path) as f:
        for line in f:
            data.append(json.loads(line))
    return data
def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json(data, output_file):
    with open(output_file, 'w') as file:
        json.dump(data, file, indent=4)

def read_conll(file_path):
    sentences = []
    sentence = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.strip() == "":
                if sentence:
                    sentences.append(sentence)
                    sentence = []
            else:
                token = line.strip().split()
                sentence.append({
                    "str_words": token[0],
                    "tags_ner": token[1]
                })

    if sentence:
        sentences.append(sentence)

    return sentences


### process data
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






# ----- performance -----
def safe_div(num, denom):
    if denom > 0:
        return num / denom
    else:
        return 0


def compute_f1(predicted, gold, matched):
    # F1 score.
    precision = safe_div(matched, predicted)
    recall = safe_div(matched, gold)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return dict(precision=precision, recall=recall, f1=f1)


def evaluate_sent(gt_ner, pred_ner,counts):
    # correct_ner = set()
    # Entities.
    counts["ner_gold"] += len(gt_ner)
    counts["ner_predicted"] += len(pred_ner)
    for prediction in pred_ner:
        if any([prediction == actual for actual in gt_ner]):
            counts["ner_matched"] += 1
            # correct_ner.add(prediction[0])
    return counts