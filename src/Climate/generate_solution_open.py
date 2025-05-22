import random
import re
import os
import transformers
import torch
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

instruction = """
You are a climate scientist. Your task is to provide solution process for a given questions about climate issues. The question was answered with relevant tools and logical reasoning, but you shouldn't mention these tools using procedures in your answer.

You should provide a solution process that includes:
- Relevant data from the climate simulator (but you shouldn't mention the simulations and simulators, you should directly list relevant data)
- Logical reasoning connecting the data to the question

If you think the answer is wrong, you should answer 'The answer is wrong' and explain why. If you think the question is correct, you should answer in the following format:

Solution: <solution>
Answer: <answer, the list of locations>
"""


def process_chunk(chunk, gpu_id, model_id, instruction):
    pipeline = transformers.pipeline(
        "text-generation",
        model=model_id,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map=f"cuda:{gpu_id}",
    )

    processed_chunk = []

    for item in tqdm(chunk, desc=f"GPU {gpu_id}"):
        prompt = (
            "Following is the interaction for answering the question.\n\n"
            + "\n".join([str(i) for i in item["llama"][-3:]])
            + f"(Got Answer: {item['llama'][-1]['function_call']['parameters']['answer']})\n"
        )

        prompt += "Following is the question that you should write solution for.\n\n"

        problem_text = f"Question: {item['question']}"

        llama_messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": prompt
                + problem_text
                + "Now begin to write solution for this question based on the question and interaction.",
            },
        ]

        response = pipeline(llama_messages, max_new_tokens=2048)
        print(response[0]["generated_text"][-1])
        item["cot"] = response[0]["generated_text"][-1]["content"]

        processed_chunk.append(item)

    return processed_chunk


def main():
    model_id = "t"
    raw_data = []
    for file in []:
        with open(file, "r") as f:
            raw_data.extend(json.load(f))
    raw_data = [
        d for d in raw_data if "llama" in d and d["llama"] and len(d["llama"]) > 3
    ]

    with open("", "r") as f:
        data = json.load(f)

    existing_questions = {item.get("question") for item in data}

    for item in raw_data:
        if "question" in item and item["question"] not in existing_questions:
            data.append(item)
            existing_questions.add(item["question"])

    raw_data = data

    # Create exactly 8 chunks
    num_gpus = 8
    chunk_size = len(raw_data) // num_gpus
    chunks = [
        raw_data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_gpus - 1)
    ]
    chunks.append(raw_data[(num_gpus - 1) * chunk_size :])

    print(len(chunks))

    # Process chunks in parallel
    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_chunk, chunk, i, model_id, instruction)
            for i, chunk in enumerate(chunks)
        ]

        processed_data = []
        for future in as_completed(futures):
            processed_data.extend(future.result())

    with open("", "w") as f:
        json.dump(processed_data, f, indent=4)


if __name__ == "__main__":
    print(f"Available GPUs: {torch.cuda.device_count()}")
    main()
