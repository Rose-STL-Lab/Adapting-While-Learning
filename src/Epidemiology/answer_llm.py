import os
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), 'emulators/pandemic'))
import random
import torch
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils import *
from tools.emulators import *
from functions.functions import *
import multiprocessing

# Ensure TOKENIZERS_PARALLELISM is set to false to avoid tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Prepare global variables
useful_tools = ["emulate"]
functions = [functions_pandemic[name] for name in useful_tools]
_functions = []
for value in functions:
    if "parameters" in value and "properties" in value["parameters"]:
        value["parameters"]["properties"]["thought"] = {
            "type": "string",
            "description": "Your internal reasoning and thoughts of why you call this function.",
        }
    _functions.append({"type": "function", "function": value})
functions = _functions + function_answer

# System prompt for the LLM
system_prompt = """
You are an epidemiologist. You are going to answer a multi-choice question. Use the given tools to help you answer the question. Call only one tool at a time.
When you've used the `emulate` function to gather enough information, use the `answer_question` function to choose one answer from A/B/C/D.
"""

# Function to process a chunk of questions
def process_chunk(questions_subset, gpu_id, chunk_id):
    llama3 = llama(device=f"cuda:{gpu_id}")
    processed_questions = []

    for question in tqdm(questions_subset, desc=f"GPU {gpu_id} - Chunk {chunk_id}"):
        
        problem_text = f"Question: {question['question']}\n\nOptions: A. {question['options'][0]}\nB. {question['options'][1]}\nC. {question['options'][2]}\nD. {question['options'][3]}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text},
        ]
        
        question[""] = func_chain(messages, question['scenario'], llama3)
        processed_questions.append(question)
        
        save_progress(processed_questions, chunk_id)
    
    return processed_questions

# Function to save chunk progress
def save_progress(processed_data, chunk_id):
    filename = f"progress_chunk_{chunk_id}.json"
    with open(filename, "w") as f:
        json.dump(processed_data, f, indent=4)

# Modified func_chain to work in multiprocessing
def func_chain(messages, scenario, llama3):
    global functions
    tried = 0
    while True:
        tried += 1
        if tried > 10:
            continue
        func_call = llama3.generate(messages, functions)
        if not func_call:
            return
        messages.append({"role": "assistant", "content": func_call})
        func_name = func_call["name"]
        func_para = func_call["parameters"]
        
        try:
            if func_name == "answer_question":
                return messages
            else:
                func_para.pop("thought", None)
                func_para["scenario"] = scenario
                back_content = globals()[func_name](**func_para)

            print(back_content)
        except Exception as e:
            print(e)
            back_content = f"Error: {e}"
        messages.append({"role": "tool", "name": func_name, "content": back_content})

        if len(messages) > 20:
            return None

# Function to split data into chunks for parallel processing
def split_data(data, num_chunks):
    random.shuffle(data)
    chunk_size = len(data) // num_chunks + 1
    return [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

if __name__ == "__main__":
    # Initialize multiprocessing
    multiprocessing.set_start_method('spawn')

    # Load dataset
    with open("qa_pairs_final_test.json", "r") as f:
        todo = json.load(f)
    
    # Define number of chunks and workers
    num_chunks = 8
    chunks = split_data(todo, num_chunks)

    # Process data in parallel using ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        futures = [executor.submit(process_chunk, chunk, i % torch.cuda.device_count(), i) for i, chunk in enumerate(chunks)]
        
        processed_data = []
        for future in as_completed(futures):
            processed_data.extend(future.result())

    # Combine results from all chunks
    final_processed_data = processed_data

    # Save final dataset
    with open("qa_pairs_final_test.json", "w") as f:
        json.dump(final_processed_data, f, indent=4)