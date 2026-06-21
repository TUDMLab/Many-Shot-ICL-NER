from openai import OpenAI
from typing import List, Dict, Any
from const import AI_CLASSS, CLIMATE_CHAIN_PROMPTS_ZERO, CLIMATE_CHAIN_PROMPTS_FEW, CoT_CLIMATE_CHAIN_PROMPTS_ZERO
from tqdm import tqdm
import os, time
from utils import extract_entities, save_json_as_csv, load_json, get_ners, save_json


def llm_call(user_prompt: str, system_prompt: str = "", model="gpt-4o") -> str:
    """
    Calls the model with the given prompt and returns the response.

    Args:
        prompt (str): The user prompt to send to the model.
        system_prompt (str, optional): The system prompt to send to the model. Defaults to "".
        model (str, optional): The model to use for the call. Defaults to "claude-3-5-sonnet-20241022".

    Returns:
        str: The response from the language model.
    """
    # OpenAI YOUR_OPENAI_API_KEY
    # client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.deepseek.com/v1") # TODO: Change the API key
    client = OpenAI(api_key="YOUR_OPENAI_API_KEY") # TODO: Change the API key
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        # max_tokens=4096, # not for o3-mini
        stream=False
    )
    return response.choices[0].message.content

# def chain(input: str, prompts: List[str]) -> str:
#     """Chain multiple LLM calls sequentially, passing results between steps."""
#     response = input
#     result = []
#     result.append(response)
#     for i, prompt in enumerate(prompts, 1):
#         # print(response)
#         response = llm_call(f"{prompt}\nInput: {response}")
#         result.append(response)
#     return result

def chain(model: str, input: str, prompts: List[str]) -> List[str]:
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
                current_response = llm_call(user_prompt=f"{prompt}\n\nInput: {response}", model=model)
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

def run_tag_workflow_few_shot(model: str = "gpt-4o", setting: str = "few-shot"):
    """
    Runs the few-shot workflow, saving results incrementally for each chunk and file.
    """
    # check setting
    if setting == "few-shot":
        setting = "few_shot"
        user_prompt = CLIMATE_CHAIN_PROMPTS_FEW
        print("Running **few-shot**")
    elif setting == "zero-shot":
        setting = "zero_shot"
        user_prompt = CLIMATE_CHAIN_PROMPTS_ZERO
        print("Running **zero-shot**")
    elif setting == "zero-shot-CoT":
        setting = "zero-shot-CoT"
        user_prompt = CoT_CLIMATE_CHAIN_PROMPTS_ZERO
    print(f"Model: {model}")
    print(f"Setting: {setting}")


    
    # Paths
    doc_files = os.listdir('./datasets/climate/')
    output_dir = f'./output/baselines/{model}_{setting}/'
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
                output = chain(model, text, user_prompt)
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

if __name__ == '__main__':
    # run_GliNER()
    # run_tag_workflow()
    run_tag_workflow_few_shot(model="gpt-4o", setting="zero-shot")
    # run_one_stage()
    # out_path = os.listdir('./output/baselines/llm_definition/')
    # print(out_path)