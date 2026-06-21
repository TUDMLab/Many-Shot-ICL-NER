from openai import OpenAI
from typing import List, Dict, Any
from const import AI_CLASSS, CLIMATE_CHAIN_PROMPTS_ZERO, CLIMATE_CHAIN_PROMPTS_FEW
from tqdm import tqdm
import os, time
from utils import extract_entities, save_json_as_csv, load_json, get_ners, save_json


ENTITY_TYPING_PROMPT_ZERO = """Classify all entity mentions in brackets into the corresponding types based on the context. The interested types and their definitions are:
    - 'project': A project refers to the scientific program, field campaign, or project from which the data were collected.
    - 'location': A location is a place on Earth, a location within Earth, a vertical location, or a location outside of the Earth.
    - 'model': A model is a sophisticated computer simulation that integrate physical, chemical, biological, and dynamical processes to represent and predict Earth's climate system.
    - 'experiment': An experiment is a structured simulation designed to test specific hypotheses, investigate climate processes, or assess the impact of various forcings on the climate system.
    - 'platform': A platform refers to a system, theory, or phenomenon that accounts for its known or inferred properties and may be used for further study of its characteristics.
    - 'instrument': A instrument is a device used to measure, observe, or calculate.
    - 'provider': A provider is an organization, an academic institution or a commercial company.
    - 'variable': A variable is a quantity or a characteristic that can be measured or observed in climate experiments.
    - 'weather event': A weather event refers to a specific atmospheric phenomenon or condition, such as storms, hurricanes, droughts, or heatwaves, occurring over a short period of time and often having measurable impacts on the environment or society.
    - 'natural hazard': A natural hazard is a potentially damaging physical event, such as earthquakes, volcanic eruptions, floods, or landslides, that arises from natural processes and may pose risks to human life, property, or the environment.
    - 'teleconnection': A teleconnection is a statistical relationship or linkage between climate anomalies in different geographic regions, often driven by large-scale atmospheric or oceanic patterns, such as El Niño or the North Atlantic Oscillation.
    - 'ocean circulation': Ocean circulation refers to the large-scale movement of water masses within the oceans, driven by factors such as wind, salinity, and temperature gradients, and playing a critical role in regulating Earth’s climate system.
    - other  (use this if the mention does not fit any of the above)
    Return only a JSON list of objects with keys "entity" and "type". Do not add explanations.

    Examples:
    Context: The CMIP6 experiments were run by the UK Met Office and included the simulation of front patterns in the North Atlantic.
    Mentioned entities: ['CMIP6', 'UK Met Office', 'North Atlantic']
    Output: [{"entity": "CMIP6", "type": "experiment"}, {"entity": "UK Met Office", "type": "provider"}, {"entity": "North Atlantic", "type": "location"}]\n\n
    """

def llm_call(prompt: str, system_prompt: str = "", model="gpt-4o") -> str:
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
            {"role": "user", "content": prompt},
        ],
        max_tokens=4096,
        stream=False
    )
    return response.choices[0].message.content

# Main script
prompt = ENTITY_TYPING_PROMPT_ZERO
ann_file_path = "./datasets/climate/"
files = os.listdir(ann_file_path)
result_dir = "./entity_typing_results/"

# Ensure result directory exists
os.makedirs(result_dir, exist_ok=True)

# Process files
for file in files:
    print(f"Processing {file}")
    data = load_json(os.path.join(ann_file_path, file))
    doc_res = []
    result_file = os.path.join(result_dir, file)

    # Check if file has been partially processed
    if os.path.exists(result_file):
        print(f"Loading existing results for {file}")
        doc_res = load_json(result_file)

    processed_chunks = {chunk["text"] for chunk in doc_res}  # Avoid re-processing completed chunks

    for chunk in tqdm(data):
        if chunk["text"] in processed_chunks:
            continue  # Skip already processed chunks

        # Prepare input for the LLM
        entities = chunk['entities']
        mentioned_entities = [ent['substring'] for ent in entities]
        labels = [ent['label'] for ent in entities]
        try:
            current_response = llm_call(f"{prompt}\nContext: {chunk['text']}\nMentioned entities: {mentioned_entities}\nOutput:")
        except Exception as e:
            print(f"Error processing chunk: {e}")
            continue

        # Save chunk result
        chunk_res = {
            "text": chunk["text"],
            "entities": entities,
            "response": current_response,
            "labels": labels,
        }
        doc_res.append(chunk_res)

        # Save intermediate results after each chunk
        save_json(doc_res, result_file)

    print(f"Finished processing {file}")

# Save final aggregated results
save_json({"files": files}, "./entity_typing_summary.json")


# # def run_et()
# prompt = ENTITY_TYPING_PROMPT_ZERO
# # scan the entir folder
# ann_file_path = "./datasets/climate/"
# files = os.listdir(ann_file_path)
# result_dir = "./entity_typing_results/"
# # Ensure result directory exists
# os.makedirs(result_dir, exist_ok=True)

# res = {}
# for file in files:
#     print(f"Processing {file}")
#     data = load_json(f'./datasets/climate/{file}')
#     doc_res = []
#     for chunk in tqdm(data):
#         entities = chunk['entities']
#         mentioned_entities = [ ent['substring'] for ent in entities]
#         labels = [ ent['label'] for ent in entities]
#         current_response = llm_call(f"{prompt}\nContext: {chunk['text']}\nMentioned entities: {mentioned_entities}\nOutput:")
#         chunk_res = {}
#         chunk_res['text'] = chunk['text']
#         chunk_res['entities'] = entities
#         chunk_res['response'] = current_response
#         chunk_res['labels'] = labels
#         doc_res.append(chunk_res)
#         break
#     # save one result as a jsonline file
#     save_json(doc_res, f'./entity_typing_results/{file}')
#     res[file] = doc_res

# # save
# save_json(res, './entity_typing.json')