import random
import re
import os
import transformers
import torch
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import random

instruction = """
Please answer the following question. Your answer should be in the following format:
Solution: <your solution>
Answer: <your answer, one of A/B/C/D>
"""

def process_chunk(chunk, gpu_id, model_id, instruction):
    gpu_id = gpu_id % torch.cuda.device_count()
    tokenizer = transformers.AutoTokenizer.from_pretrained("")
    pipeline = transformers.pipeline(
        "text-generation",
        tokenizer=tokenizer,
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=f"cuda:{gpu_id}",
    )

    processed_chunk = []

    for item in tqdm(chunk):
        problem_text = f"Question: {item['Question']}\nOptions:\nA. {item['Options'][0]}\nB. {item['Options'][1]}\nC. {item['Options'][2]}\nD. {item['Options'][3]}"
        llama_messages = [{"role": "system", "content": instruction},
                            {"role": "user", "content": problem_text}]
        
        response = pipeline(llama_messages, max_new_tokens=1024)
        print(response[0]["generated_text"][-1])
        item[model_id] = response[0]["generated_text"][-1]["content"]

        processed_chunk.append(item)

    return processed_chunk

def main():
    model_ids = ["/"]
    with open("mujoco_test.json", "r") as f:
        raw_data = json.load(f)

    # Create exactly 8 chunks
    num_chunks = 6
    chunk_size = len(raw_data) // num_chunks + 1
    chunks = [raw_data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks-1)]
    chunks.append(raw_data[(num_chunks-1)*chunk_size:])  # Add the remaining items to the last chunk
    
    print(len(chunks))

    for model_id in model_ids:
        # Process chunks in parallel
        with ProcessPoolExecutor(max_workers=6) as executor:
            futures = [executor.submit(process_chunk, chunk, i, model_id, instruction) for i, chunk in enumerate(chunks)]
            
            processed_data = []
            for future in as_completed(futures):
                processed_data.extend(future.result())

        with open(f"mujoco_test.json", "w") as f:
            json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    main()