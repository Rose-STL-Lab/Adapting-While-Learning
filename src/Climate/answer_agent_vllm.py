# answer_agent_vllm.py
import sys
import os

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(src_dir)

from utils.utils import *
from functions.functions import *

import json
import requests
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import signal
from tqdm import tqdm
import torch
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

# Define the API endpoint
API_ENDPOINT = "http://localhost:5000"

# Define the useful tools
useful_tools = [
    "location_summary",
    "history_temperature",
    "future_temperature",
    "query_lat_and_lon",
    "diy_greenhouse",
]

# Function to call the API
def call_tool_api(func_name, params):
    url = f"{API_ENDPOINT}/{func_name}"
    try:
        response = requests.post(url, json=params, timeout=30)  # Add timeout
        if response.status_code == 200:
            return response.json()
        else:
            error_message = f"API call failed with status {response.status_code}"
            try:
                # Try to get more detailed error from response
                error_data = response.json()
                if "text" in error_data:
                    error_message = error_data["text"]
            except:
                pass
            return {"result": None, "text": f"Error: {error_message}"}
    except requests.RequestException as e:
        return {"result": None, "text": f"Error connecting to API: {str(e)}"}

# Parse the model output to extract function call information
def parse_function_call(output_text):
    # Clean up the output
    output_text = (
        output_text.replace("<|python_tag|>", "")
        .replace("<|eom_id|>", "")
        .replace("<|eot_id|>", "")
        .replace("None<|end_header_id|><|start_header_id|>assistant<|end_header_id|>", "")
        .replace("None<|end_header_id|><|start_header_id|>function<|end_header_id|>", "")
        .strip()
    )
    
    # Print the cleaned output for debugging
    sys.stdout.print_colored(output_text, "yellow")
    
    # Process the output based on format
    if "\n" in output_text:
        output_text = output_text.split("\n")[0]
    
    try:
        if ";" not in output_text:
            # Single function call format
            json_data = json.loads(output_text.replace("'", '"'))
        else:
            # Multiple function calls format (take first one)
            output_parts = output_text.replace("'", '"').split(";")
            json_data = json.loads(output_parts[0].strip())
        
        # Extract function name and parameters
        if "name" in json_data and "parameters" in json_data:
            return {
                "name": json_data["name"],
                "parameters": json_data["parameters"]
            }
        else:
            # Try alternative format
            return {
                "name": list(json_data.keys())[0],
                "parameters": json_data[list(json_data.keys())[0]]
            }
    except Exception as e:
        sys.stdout.print_colored(f"Error parsing function call: {e}", "red")
        return None

# Define the function chains
def func_chain(messages, llm, sampling_params, tokenizer, functions):
    while len(messages) < 20:
        try:
            # Convert messages to prompt format
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                tools=functions,
            )
            
            # Generate response
            outputs = llm.generate(prompt_token_ids=prompt, sampling_params=sampling_params, use_tqdm=False)
            output_text = outputs[0].outputs[0].text
            
            # Parse the function call
            func_call = parse_function_call(output_text)

            print(f"Function Call: {func_call}")
            
            if func_call is None:
                # If parsing fails, try again with a simpler prompt
                messages.append({"role": "user", "content": "Please call a function in the correct JSON format."})
                continue
                
            func_name = func_call["name"]
            func_para = func_call["parameters"]
            
            # Add the function call to messages
            messages.append({"role": "assistant", "content": output_text})
            
            if func_name == "answer_question":
                return messages
            else:
                # Remove thought parameter before API call
                thought = func_para.pop("thought", None)
                
                # Call the API
                api_response = call_tool_api(func_name, func_para)
                back_content = api_response["text"]
                
                print(back_content)
                
        except Exception as e:
            print(f"Error: {e}")
            back_content = f"Error: {e}"

def process_chunk(questions_subset, gpu_id, model_id, functions):
    print(f"GPU {gpu_id} - Processing {len(questions_subset)} questions")
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id + 1)
    
    # Initialize vLLM
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.8,
        max_model_len=8192
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=1024,
        stop=["</tool_call>", "<|im_end|>"]
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    results = []
    
    system_prompt = """
    You are a climate scientist. You are going to answer a multi-choice question. You should use given tools to help you answer the question.

    You must firstly use `query_lat_and_lon` to get the latitude and longitude of the given place, even if you think you know the latitude and longitude of the place.
    """
    
    with open("few_shot.txt", "r") as f:
        few_shot = f.read()
    
    for question in tqdm(questions_subset, desc=f"GPU {gpu_id}"):
        problem_text = f"Question: {question['Question']}\nOptions:\nA. {question['Options'][0]}\nB. {question['Options'][1]}\nC. {question['Options'][2]}\nD. {question['Options'][3]}"
        print(problem_text)
        print(question["Correct"])
        
        messages = [
            {
                "role": "system",
                "content": system_prompt + few_shot,
            },
            {"role": "user", "content": problem_text},
        ]
        
        question[model_id] = func_chain(messages, llm, sampling_params, tokenizer, functions)
        results.append(question)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Process climate questions with vLLM')
    parser.add_argument('--model', type=str, default="/home/test/test12/bohan/models/Meta-Llama-3.1-8B-Instruct", 
                        help='Path to the model')
    parser.add_argument('--num_gpus', type=int, default=7, help='Number of GPUs to use')
    args = parser.parse_args()
    
    # Process functions for vLLM format
    _functions = []
    for name in useful_tools:
        value = functions_climate[name].copy()
        if "parameters" in value and "properties" in value["parameters"]:
            value["parameters"]["properties"]["thought"] = {
                "type": "string",
                "description": "Your internal reasoning and thoughts of why you call this function.",
            }
        _functions.append({"type": "function", "function": value})
    
    functions = _functions + function_answer
    
    # Load questions
    with open("climate_train.json", "r") as f:
        questions = json.load(f)
    
    # Distribute questions across GPUs
    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    chunk_size = len(questions) // num_gpus + (1 if len(questions) % num_gpus != 0 else 0)
    chunks = [questions[i:i+chunk_size] for i in range(0, len(questions), chunk_size)]
    
    # Process in parallel
    if num_gpus > 1:
        with ProcessPoolExecutor(max_workers=num_gpus) as executor:
            futures = [
                executor.submit(process_chunk, chunk, gpu_id, args.model, functions) 
                for gpu_id, chunk in enumerate(chunks)
            ]
            
            all_results = []
            for future in as_completed(futures):
                all_results.extend(future.result())
    else:
        all_results = process_chunk(questions, 0, args.model, functions)
    
    # Save results
    with open("test.json", "w") as f:
        json.dump(all_results, f, indent=4)

if __name__ == "__main__":
    main()