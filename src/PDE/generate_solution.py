import random
import re
import os
import transformers
import torch
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

instruction = """
You are a physicist specialized in solving PDE problems. Your task is to provide a detailed solution process for a given question. The question was answered with a PDE solver and logical reasoning, but you shouldn't mention these tools using procedures in your answer.

You should list detailed data from the PDE simulator (but you shouldn't mention the emulator, you should directly list relevant data). Analysis of this data. Logical reasoning connecting the data to the answer.

If you think the answer is wrong or the solving can't solve the question, you should answer 'The answer is wrong' and explain why, or give the correct answer in A/B/C/D.

Your answer should be in the following format:
Solution: <solution>
Answer: <answer, one of A/B/C/D>
"""

def process_chunk(chunk, gpu_id, model_id, instruction):
    gpu_id = gpu_id % torch.cuda.device_count()
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=f"cuda:{gpu_id}",
    )

    processed_chunk = []

    for item in tqdm(chunk):
        if "llama" not in item:
            continue
        if not item["llama"]:
            continue
        if "cot" in item and "answer:" in item["cot"].lower():
            continue
        print(json.dumps(item['llama'][-1], indent = 4))
        prompt = "Following is the interaction for answering the question.\n\n" + "\n".join([str(i) for i in item['llama'][-3:]]) + f"(Got Answer: {item['llama'][-1]['function_call']['parameters']['answer']})\n"
        
        prompt += "Following is the question that you should write solution for.\n\n"
       
        problem_text = f"Question: {item['question']}\n\nOptions: A. {item['options'][0]}\nB. {item['options'][1]}\nC. {item['options'][2]}\nD. {item['options'][3]}"
        messages = [
            {
                "role": "system",
                "content": instruction,
            },
            {"role": "user", "content": prompt + problem_text},
        ]
        
        response = pipeline(messages, max_new_tokens=2048)
        print(response[0]["generated_text"][-1])
        item["cot"] = response[0]["generated_text"][-1]["content"]

        processed_chunk.append(item)

    return processed_chunk

def main():
    model_id = ""
    with open("diffusion_questions.json", "r") as f:
        raw_data = json.load(f)
    # Create exactly 8 chunks
    num_chunks = 8
    chunk_size = len(raw_data) // num_chunks + 1
    chunks = [raw_data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks-1)]
    chunks.append(raw_data[(num_chunks-1)*chunk_size:])  # Add the remaining items to the last chunk
    
    print(len(chunks))

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_chunk, chunk, i, model_id, instruction) for i, chunk in enumerate(chunks)]
        
        processed_data = []
        for future in as_completed(futures):
            processed_data.extend(future.result())

    with open("diffusion_questions.json", "w") as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    main()