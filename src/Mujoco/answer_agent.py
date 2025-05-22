import os
import json
import sys
import random
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
from utils import *
from functions.functions import *
from tools.mujoco_all import *

# Ensure TOKENIZERS_PARALLELISM is set to false to avoid tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def convert_dict_values(d):
    new_dict = {}
    for key, value in d.items():
        if key == "simend":
            new_dict[key] = int(value)
        else:
            new_dict[key] = float(value)
    return new_dict

def generate_function(typem):
    useful_tools = [typem]
    functions = [functions_mujoco[name] for name in useful_tools]
    _functions = []
    for value in functions:
        if "parameters" in value and "properties" in value["parameters"]:
            value["parameters"]["properties"]["thought"] = {
                "type": "string",
                "description": "Your internal reasoning and thoughts of why you call this function.",
            }
        _functions.append({"type": "function", "function": value})
    functions = _functions + function_answer
    return functions

system_prompt = """
You are a physicist. You are going to answer a multi-choice question. You should use given tools to help you answer the question. You can call tools for many turns, but you should call only one tool each time. When you have used the tool to get enough information, you should use answer_question to choose one answer from A/B/C/D.

Now Begin!
"""

def func_chain(messages, functions, llama3):
    tried = 0
    while True:
        tried += 1
        if tried > 10:
            continue
        func_call = llama3.generate(messages, functions)
        if not func_call:
            return None
        messages.append(
            {"role": "assistant", "content": None, "function_call": func_call}
        )
        func_name = func_call["name"]
        func_para = func_call["parameters"]
        
        try:
            if func_name == "answer_question":
                return messages
            else:
                func_para.pop("thought", None)
                func_para = convert_dict_values(func_para)
                back_content, _ = globals()[func_name](**func_para)
        except Exception as e:
            print(e)
            back_content = f"Error: {e}"
            
        print(back_content)
        messages.append(
            {"role": "function", "name": func_name, "content": back_content}
        )
        if len(messages) > 10:
            return None

def process_chunk(questions_subset, gpu_id, chunk_id, output_file):
    llama3 = llama(device=f"cuda:{gpu_id}", model_path = "")

    for question in tqdm(questions_subset, desc=f"GPU {gpu_id} - Chunk {chunk_id}"):
        if "type" not in question:
            question["type"] = "sphere_collision"
        problem_text = f"Question: {question['Question']}\n\nOptions: A. {question['Options'][0]}\nB. {question['Options'][1]}\nC. {question['Options'][2]}\nD. {question['Options'][3]}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text},
        ]
        functions = generate_function(question["type"] + "_simulation")
        question[""] = func_chain(messages, functions, llama3)
        print(question["Correct"])
        
        # Write the processed question to the JSONL file
        with open(output_file, 'a') as f:
            json.dump(question, f)
            f.write('\n')

def split_data(data, num_chunks):
    chunk_size = len(data) // num_chunks
    return [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

def load_and_update_questions(original_json_file, processed_jsonl_file):
    # Load original JSON data
    with open(original_json_file, "r") as f:
        original_questions = json.load(f)

    # Create a dictionary of original questions for easy lookup
    original_dict = {q['Question']: q for q in original_questions}

    # Load and process JSONL data
    updated_questions = []
    if os.path.exists(processed_jsonl_file):
        with open(processed_jsonl_file, "r") as f:
            for line in f:
                processed_q = json.loads(line.strip())
                if processed_q['Question'] in original_dict:
                    updated_questions.append(processed_q)
                    original_dict.pop(processed_q['Question'])
                else:
                    updated_questions.append(processed_q)

    # Add remaining questions from the original data
    updated_questions.extend(original_dict.values())

    return updated_questions  # Limit to 2000 questions as in the original code

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')

    original_json_file = "mujoco_test.json"
    processed_jsonl_file = "temp_mujoco_processed.jsonl"

    with open(original_json_file, "r") as f:
        questions = json.load(f)
    
    random.shuffle(questions)
    
    num_gpus = 7
    num_chunks = 7 
    chunks = split_data(questions, num_chunks)

    temp_output_file = "temp_mujoco_processed.jsonl"
    
    open(temp_output_file, 'w').close()

    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        futures = [executor.submit(process_chunk, chunk, i % num_gpus, i, temp_output_file) for i, chunk in enumerate(chunks)]
        
        for future in as_completed(futures):
            future.result()

    processed_data = []
    with open(temp_output_file, 'r') as f:
        for line in f:
            processed_data.append(json.loads(line))

    with open(original_json_file, 'w') as f:
        json.dump(processed_data, f, indent=2)

    os.remove(temp_output_file)