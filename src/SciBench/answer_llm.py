import random
import re
import os
import json
import torch
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

instruction = """
Please answer the following given question. You should first give the Solution then give the answer. You should think step by step. The final answer shoud be presented in latex like '\boxed{\frac{1}{2}}'. Your answer should be in the following format:
Solution: <Your solution>
Answer: <Your answer, a number>
"""

def process_chunk(chunk, gpu_id, chunk_id, model_id, instruction):
    print(f"GPU {gpu_id} - Chunk {chunk_id} - Processing {len(chunk)} items")
    
    # Set single GPU for process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id + 1)
    
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.7,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    llm = LLM(
        model=model_id,
        tensor_parallel_size=1,  # Changed to 1 as we're using 1 GPU per process
        max_model_len=8000,
        gpu_memory_utilization=0.95
    )

    processed_chunk = []
    
    prompts = []
    for item in chunk:
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": f"Problem: {item['problem']}"},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        prompts.append(prompt)
    
    # Batch inference
    outputs = llm.generate(prompts, sampling_params)
    
    for item, output in zip(chunk, outputs):
        item[model_id] = output.outputs[0].text
        processed_chunk.append(item)

    return processed_chunk

def main():
    multiprocessing.set_start_method("spawn")
    
    model_ids = []
    
    test = []
    with open("sci_final.json", "r") as f:
        raw_data = json.load(f)

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    # Configure for 8 processes, each using 1 GPU
    num_processes = 6
    
    for model_id in model_ids:
        # Create exactly 8 chunks
        chunk_size = len(raw_data) // num_processes + 1
        chunks = [
            raw_data[i * chunk_size : (i + 1) * chunk_size]
            for i in range(num_processes - 1)
        ]
        chunks.append(
            raw_data[(num_processes - 1) * chunk_size :]
        )

        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            futures = [
                executor.submit(process_chunk, chunk, i, i, model_id, instruction)
                for i, chunk in enumerate(chunks)
            ]

            processed_data = []
            for future in as_completed(futures):
                processed_data.extend(future.result())

        raw_data = processed_data
        
        with open(f"sci_final.json", "w") as f:
            json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    main()