import random
import re
import os
import transformers
import torch
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams

def process_chunk(chunk, gpu_id, model_id, instruction):
    gpu_id = gpu_id % torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        ""
    )
    
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.95,
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=4096,
        stop=None
    )

    processed_chunk = []
    all_prompts = []
    
    for item in chunk:
        problem_text = f"Question: {item['question']}"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": problem_text}
        ]
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        all_prompts.extend([prompt] * 5)
        item[model_id] = []
        processed_chunk.append(item)

    print(f"Processing {len(all_prompts)} prompts on GPU {gpu_id}")
    outputs = llm.generate(all_prompts, sampling_params)
    
    for output, item in zip(outputs, [item for item in processed_chunk for _ in range(5)]):
        item[model_id].append(output.outputs[0].text)

    return processed_chunk

def main():
    model_ids = []
    
    with open("", "r") as f:
        raw_data = json.load(f)

    print(len(raw_data))

    for model_id in model_ids:
        instruction = """
You are a climate scientist. You are going to answer a given question. You should answer in the following format:\n\nSolution: <solution>\nAnswer: <answer, the list of locations (lon_i, lat_i)s>\nA Demo Answer: Solution: \n1. Identify the starting point and endpoint of the route, which are Shenzhen (lon: 114.0596, lat: 22.5415) and London (lon: -0.1275, lat: 51.5072) respectively.\n2. Determine the significant waypoints that would increase SO2 levels along the route by 20.0%. \n   - Relevant data: The average temperature increase in 2058 under ssp126 is 6.468113164307739.\n   - Logical reasoning: To minimize the global average temperature increase, we should avoid areas with high temperature increases. \n3. Analyze the temperature increase along different routes to find the most suitable route.\n   - Relevant data: The first route resulted in a slightly lower temperature increase compared to the second one.\n   - Logical reasoning: The first route is preferable as it minimizes the global average temperature increase.\n4. Propose a new maritime route that meets the requirements and includes at least the starting point, endpoint, and any significant waypoints.\n   - Relevant data: The route will be automatically connected based on the nodes provided.\n   - Logical reasoning: The route should include the following key points: \n     - (114.0596, 22.5415) - Shenzhen\n     - (113.8068, 23.0863) - A point in the sea to avoid high temperature increases\n     - (-0.1275, 51.5072) - London\n\nAnswer: [(114.0596, 22.5415), (113.8068, 23.0863), (-0.1275, 51.5072)]\n\nBegin!
"""
        num_chunks = 7
        chunk_size = len(raw_data) // num_chunks + 1
        chunks = [raw_data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks-1)]
        chunks.append(raw_data[(num_chunks-1)*chunk_size:])
        
        print(len(chunks))
        
        with ProcessPoolExecutor(max_workers=num_chunks) as executor:
            futures = [
                executor.submit(process_chunk, chunk, i, model_id, instruction) 
                for i, chunk in enumerate(chunks)
            ]
            
            processed_data = []
            for future in as_completed(futures):
                processed_data.extend(future.result())
        
        with open(f"", "w") as f:
            json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    main()