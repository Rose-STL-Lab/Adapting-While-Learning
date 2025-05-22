import random
import re
import os
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def process_chunk(chunk, gpu_id, model_id, instruction):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id + 7)
    
    sampling_params = SamplingParams(
        max_tokens=4096,
        temperature=0.7,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    llm = LLM(
        model=model_id
    )

    processed_chunk = []
    
    prompts = []
    for item in chunk:
        problem_text = f"Question: {item['question']}\nOptions:\nA. {item['options'][0]}\nB. {item['options'][1]}\nC. {item['options'][2]}\nD. {item['options'][3]}"
        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": problem_text},
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
    model_ids = [
    ]
    
    with open("pde_test.json", "r") as f:
        raw_data = json.load(f)

    print(len(raw_data))

    if "id" not in raw_data[0]:
        for i, item in enumerate(raw_data):
            item["id"] = i

    for model_id in model_ids:
        if model_id == "":
            instruction = """
You are a physicist. Answer the following question. Your answer should be in the following format:
Solution: <Your solution>
Answer: <Your answer, one of A/B/C/D>
"""
        else:
            instruction = """
    Please answer the following question. Your answer should end with 'the answer is A/B/C/D'.
    """
        
        num_chunks = 1
        chunk_size = len(raw_data) // num_chunks + 1
        chunks = [raw_data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks - 1)]
        chunks.append(raw_data[(num_chunks - 1) * chunk_size :])

        print(len(chunks))
        
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(process_chunk, chunk, i, model_id, instruction)
                for i, chunk in enumerate(chunks)
            ]

            processed_data = []
            for future in as_completed(futures):
                processed_data.extend(future.result())

        processed_data.sort(key=lambda x: x["id"])
        raw_data = processed_data

        if model_id == "":
            for d in raw_data:
                d[model_id] = d[model_id].replace("Answer:", "the answer is")

        with open(f"pde_test.json", "w") as f:
            json.dump(raw_data, f, indent=4)

if __name__ == "__main__":
    main()