import json
import os
import random
import re
from concurrent.futures import ProcessPoolExecutor, as_completed

import torch
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

instruction = """
You are a climate scientist. Your task is to provide solution process for a given questions about climate issues. The question was answered with relevant tools and logical reasoning, but you shouldn't mention these tools using procedures in your answer.

You should provide a solution process that includes:
- Relevant data from the climate simulator (but you shouldn't mention the simulations and simulators, you should directly list relevant data)
- Logical reasoning connecting the data to the question

If you think the answer is wrong, you should answer 'The answer is wrong' and explain why. If you think the question is correct, you should answer in the following format:

Solution: <solution>
Answer: <answer, the list of locations>
"""

gpu_ids = [3, 5, 6, 7]

def process_chunk(chunk, gpu_id, model_id, instruction):
    gpu_id = gpu_id % torch.cuda.device_count()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[gpu_id])
    
    llm = LLM(model=model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    sampling_params = SamplingParams(
        max_tokens=2048,
        temperature=0.7,
    )

    processed_chunk = []
    prompts = []
    valid_items = []

    for item in tqdm(chunk, desc=f"GPU {gpu_id}"):
        if "" not in item or not item[""] or len(item[""]) <= 3:
            continue

        prompt = (
            "Following is the interaction for answering the question.\n\n"
            + "\n".join([str(i) for i in item[""][-3:]])
            + f"(Got Answer: {item[''][-1]['function_call']['parameters']['answer']})\n"
        )

        prompt += "Following is the question that you should write solution for.\n\n"
        problem_text = f"Question: {item['question']}"

        messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": prompt 
                + problem_text 
                + "Now begin to write solution for this question based on the question and interaction.",
            },
        ]
        
        formatted_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        prompts.append(formatted_prompt)
        valid_items.append(item)

    # Batch inference
    outputs = llm.generate(prompts, sampling_params)
    
    for item, output in zip(valid_items, outputs):
        item["cot"] = output.outputs[0].text
        processed_chunk.append(item)

    return processed_chunk

def main():
    model_id = ""
    raw_data = []
    for file in ["open_train0.json", "open_train1.json", "open_train2.json", "open_train3.json"]:
        with open(file, "r") as f:
            raw_data.extend(json.load(f))
    
    # Filter data
    raw_data = [d for d in raw_data if 
        "" in d 
        and d[""] 
        and isinstance(d[""], list)
        and len(d[""]) > 3 
        and isinstance(d[""][-1], dict)
        and "function_call" in d[""][-1]
        and isinstance(d[""][-1]["function_call"], dict)
        and "parameters" in d[""][-1]["function_call"]
        and isinstance(d[""][-1]["function_call"]["parameters"], dict)
        and "answer" in d[""][-1]["function_call"]["parameters"]
    ]

    # Create chunks
    num_gpus = 4
    chunk_size = len(raw_data) // num_gpus
    chunks = [raw_data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus - 1)]
    chunks.append(raw_data[(num_gpus - 1) * chunk_size :])

    print(f"Number of chunks: {len(chunks)}")

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=num_gpus) as executor:
        futures = [
            executor.submit(process_chunk, chunk, i, model_id, instruction)
            for i, chunk in enumerate(chunks)
        ]

        processed_data = []
        for future in as_completed(futures):
            processed_data.extend(future.result())

    with open("open_all.json", "w") as f:
        json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    print(f"Available GPUs: {torch.cuda.device_count()}")
    main()