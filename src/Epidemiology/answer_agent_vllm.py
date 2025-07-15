import random
import json
import sys
import os
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import torch
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(src_dir)

from utils.utils import *
from functions.functions import *

sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
from tools.inference import *

# Define the useful tools
useful_tools = ["emulate"]

# Parse function call from model output
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
    
    try:
        if "\n" in output_text:
            output_text = output_text.split("\n")[0]
        
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
        print(f"Error parsing function call: {e}")
        return None

# Function chain for vLLM
def func_chain(messages, scenario, llm, sampling_params, tokenizer, functions):
    tries = 0
    while len(messages) < 20 and tries < 10:
        tries += 1
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
            
            if func_call is None:
                continue
                
            func_name = func_call["name"]
            func_para = func_call["parameters"]
            
            # Add the function call to messages
            messages.append({"role": "assistant", "content": json.dumps(func_call)})
            
            try:
                if func_name == "answer_question":
                    return messages
                else:
                    # Remove thought parameter before function call
                    func_para.pop("thought", None)
                    func_para["scenario"] = scenario
                    back_content = globals()[func_name](**func_para)

                print(back_content)
                
            except Exception as e:
                print(e)
                back_content = f"Error: {e}"
                
            messages.append({"role": "tool", "name": func_name, "content": back_content})
            
        except Exception as e:
            print(f"Error: {e}")
            back_content = f"Error: {e}"
            
        if tries >= 10:
            return None
            
    return messages

def process_chunk(questions_subset, gpu_id, model_id, functions):
    print(f"GPU {gpu_id} - Processing {len(questions_subset)} questions")
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Initialize vLLM
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
        max_model_len=20000
    )
    
    sampling_params = SamplingParams(
        temperature=0.1,
        max_tokens=1024,
        stop=["</tool_call>", "<|im_end|>"]
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    results = []
    
    system_prompt = """
    You are a epidemiologist. You are going to answer a multi-choice question. You should use given tools to help you answer the question. You can call tools for many turns, but you should call only one tool each time. When you have used `emulate` function to get enough information, you should use answer_question to choose one answer from A/B/C/D.
    """
    
    for question in tqdm(questions_subset, desc=f"GPU {gpu_id}"):
        problem_text = f"Question: {question['question']}\n\nOptions: A. {question['options'][0]}\nB. {question['options'][1]}\nC. {question['options'][2]}\nD. {question['options'][3]}"
        print("...\n" + problem_text[-500:])
        print(question["correct_option"])
        
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {"role": "user", "content": problem_text},
        ]
        
        question[model_id] = func_chain(messages, question['scenario'], llm, sampling_params, tokenizer, functions)
        results.append(question)
    
    return results

def main():
    parser = argparse.ArgumentParser(description='Process epidemiology questions with vLLM')
    parser.add_argument('--model', type=str, default="meta-llama/Llama-3.1-8B-Instruct", 
                        help='Path to the model')
    parser.add_argument('--num_gpus', type=int, default=torch.cuda.device_count(), help='Number of GPUs to use')
    args = parser.parse_args()
    
    # Process functions for vLLM format
    functions_list = [functions_pandemic[name] for name in useful_tools]
    _functions = []
    for value in functions_list:
        if "parameters" in value and "properties" in value["parameters"]:
            value["parameters"]["properties"]["thought"] = {
                "type": "string",
                "description": "Your internal reasoning and thoughts of why you call this function.",
            }
        _functions.append({"type": "function", "function": value})
    functions = _functions + function_answer
    
    print(json.dumps(functions, indent=4))
    
    # Load questions
    with open("test.json", "r") as f:
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
