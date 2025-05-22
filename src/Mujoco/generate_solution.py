import random
import re
import os
import transformers
import torch
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

instruction = r"""
You are a physicist. Your task is to provide a solution process for a given questions. The question was answered with relevant emulator, but you shouldn't mention the emulator using procedures in your answer.

You should list detailed data from the physics engine (but you shouldn't mention the emulator, you should directly list relevant data). Analysis of this data. Logical reasoning connecting the data to the answer.

If you think the answer is wrong, you should answer 'The answer is wrong' and explain why, or give the correct answer in A/B/C/D.

Your answer should be in the following format:
Solution: <solution>
Answer: <answer, one of A/B/C/D>
"""

def load_already_processed(output_file):
    processed_ids = set()
    if os.path.exists(output_file):
        with open(output_file, "r") as f:
            for line in f:
                item = json.loads(line)
                if "Question" in item:
                    processed_ids.add(item["Question"])
    return processed_ids

def process_chunk(chunk, gpu_id, model_id, instruction, output_file, processed_ids):
    gpu_id = gpu_id % torch.cuda.device_count()
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=f"cuda:{gpu_id}",
    )

    for item in tqdm(chunk):
        if "llama" not in item:
            continue
        if not item["llama"]:
            continue
        if "cot" in item and "answer:" in item["cot"].lower():
            continue
        if item["Question"] in processed_ids:
            # Skip already processed items
            continue

        prompt = "Following is the interaction for answering the question.\n\n" + "\n".join([str(i) for i in item['llama'][-3:]]) + f"(Got Answer: {item['llama'][-1]['function_call']['parameters']['answer']})\n"
        
        prompt += "Following is the question that you should write solution for.\n\n"
       
        problem_text = f"Question: {item['Question']}\n\nOptions: A. {item['Options'][0]}\nB. {item['Options'][1]}\nC. {item['Options'][2]}\nD. {item['Options'][3]}"
        messages = [
            {
                "role": "system",
                "content": instruction,
            },
            {"role": "user", "content": prompt + problem_text},
        ]
        
        response = pipeline(messages, max_new_tokens=1024)
        print(response[0]["generated_text"][-1])
        item["cot"] = response[0]["generated_text"][-1]["content"]

        # Save the processed item immediately
        with open(output_file, "a") as f:
            json.dump(item, f)
            f.write("\n")

def main():
    
    model_id = ""
    input_file = "mujoco.json"
    output_file = "mujoco.jsonl"

    # Load already processed items
    processed_ids = load_already_processed(output_file)

    # Load raw data
    with open(input_file, "r") as f:
        raw_data = json.load(f)
    
    # Create exactly 8 chunks
    num_chunks = 8
    chunk_size = len(raw_data) // num_chunks + 1
    chunks = [raw_data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks-1)]
    chunks.append(raw_data[(num_chunks-1)*chunk_size:])  # Add the remaining items to the last chunk
    
    print(f"Total chunks: {len(chunks)}")

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(process_chunk, chunk, i, model_id, instruction, output_file, processed_ids) for i, chunk in enumerate(chunks)]
        
        for future in as_completed(futures):
            future.result()

    # Combine all processed items into a single list
    processed_data = []
    with open(output_file, "r") as f:
        for line in f:
            processed_data.append(json.loads(line))

    # Save the final combined data
    with open("mujoco.json", "w") as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    main()