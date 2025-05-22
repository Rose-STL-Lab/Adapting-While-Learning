import random
import re
import os
import transformers
import torch
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

instruction = r"""
You are a epidemiologist. Your task is to provide a solution process for a given questions. The question was answered with relevant emulator, but you shouldn't mention the emulator using procedures in your answer.

You should list detailed data from the pandemic simulator (but you shouldn't mention the emulator, you should directly list relevant data). Analysis of this data. Logical reasoning connecting the data to the answer.
- You should list number from the emulator output. But you should not list a lot of numbers in the question.

If you think the answer is wrong, you should answer 'The answer is wrong' and explain why, or give the correct answer in A/B/C/D.

Your answer should be in the following format:
Solution: <solution>
Answer: <answer, one of A/B/C/D>
"""

def process_chunk(chunk, gpu_id, model_id, instruction, output_file):
    gpu_id = gpu_id % torch.cuda.device_count()
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=f"cuda:{gpu_id}",
    )

    for item in tqdm(chunk):
        if "gpt4" not in item:
            continue
        if not item["gpt4"]:
            continue
        if "cot" in item and "answer:" in item["cot"].lower():
            continue
        
        prompt = "Following is the interaction for answering the question.\n\n" + "\n".join([str(i) for i in item['gpt4'][-3:]]) + f"(Got Answer: {item['gpt4'][-1]['content']['parameters']['answer']})\n"
        
        prompt += "Following is the question that you should write solution for.\n\n"
       
        problem_text = f"Question: {item['question']}\n\nOptions: A. {item['options'][0]}\nB. {item['options'][1]}\nC. {item['options'][2]}\nD. {item['options'][3]}"
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
    os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
    
    model_id = ""
    input_file = ""
    output_file = ""

    open(output_file, 'w').close()

    with open(input_file, "r") as f:
        raw_data = json.load(f)[-120:]

    # Create exactly 8 chunks
    num_chunks = 4
    chunk_size = len(raw_data) // num_chunks + 1
    chunks = [raw_data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks-1)]
    chunks.append(raw_data[(num_chunks-1)*chunk_size:])  # Add the remaining items to the last chunk
    
    print(len(chunks))

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(process_chunk, chunk, i, model_id, instruction, output_file) for i, chunk in enumerate(chunks)]
        
        for future in as_completed(futures):
            future.result()

    # Combine all processed items into a single list
    processed_data = []
    with open(output_file, "r") as f:
        for line in f:
            processed_data.append(json.loads(line))

    # Save the final combined data
    with open("qa_pairs_final1.json", "w") as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    main()