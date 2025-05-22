import random
import re
import os
import transformers
import torch
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

instruction = """
You are a teacher specialized in climate science and data analysis. Your task is to provide a detailed solution process for a given questions about climate issues. The question was answered with relevant tools and logical reasoning, but you shouldn't mention these tools using procedures in your answer. The solution should end with 'the answer is A/B/C/D'.

You should provide a solution process that includes:
- Relevant data from the climate simulator (but you shouldn't mention the simulations and simulators, you should directly list relevant data)
- Detailed analysis of this data
- Logical reasoning connecting the data to the question

If you think the answer is wrong, you should answer 'The answer is wrong' and explain why.

Example:
Question: How much will the temperature of Domoni in 2098 under ssp370 change if the emission of CH4 is increased by 60.0%?

Options:
A. 0.02633057
B. 0.2995891571044922
C. 0.14374233
D. 0.07292475

Call function `query_lat_and_lon` with {'city_name': 'Domoni'}.
Feedback from `query_lat_and_lon`: latitude: -12.2586, longitude: 44.5303.
Call function `diy_greenhouse` with {'delta_CH4': '0', 'year': '2098', 'latitude': '-12.2586', 'setting': 'ssp370', 'delta_CO2': '0', 'longitude': '44.5303'}.
Feedback from `diy_greenhouse`: The temperature is 29.585771560668945.
Call function `answer_question` with {'thought': 'To calculate the change in temperature, we need to first calculate the new concentration of CH4 by increasing the original concentration by 60.0%. Then, we use the diy_greenhouse function to predict the temperature in 2098 under the ssp370 scenario with the new concentration of CH4.', 'answer': 'B'}.

Answer:
To calculate the temperature change in Domoni by 2098 under the ssp370 scenario with a 60.0% increase in CH4 emissions, we first identify Domoni's location at latitude -12.2586 and longitude 44.5303. The baseline temperature for 2098 is determined to be 29.585771560668945°C. After accounting for the 60.0% increase in CH4 emissions, we calculate the new temperature and find the difference. This difference is 0.2995891571044922°C, which matches option B from the given choices. Therefore, the answer is B.

Now Begin!
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
        messages = item["llama"]
        prompt = "Following is the interaction for answering the question.\n\n"
        for message in messages:
            if message["role"] == "user":
                prompt += message["content"]
            elif message["role"] == "assistant":
                prompt += f"\nCall function `{message['content']['name']}` with {message['content']['parameters']}."
            elif message["role"] == "function":
                prompt += f"\nFeedback from `{message['name']}`: {message['content']}."

        prompt += "Following is the question that you should write solution for.\n\n"

        problem_text = f"Question: {item['Question']}\nOptions:\nA. {item['Options'][0]}\nB. {item['Options'][1]}\nC. {item['Options'][2]}\nD. {item['Options'][3]}"
        llama_messages = [
            {"role": "system", "content": instruction},
            {
                "role": "user",
                "content": prompt
                + problem_text
                + "Now begin to write solution for this question based on the data, question and answer.",
            },
        ]

        response = pipeline(llama_messages, max_new_tokens=2048)
        print(response[0]["generated_text"][-1])
        item["cot"] = response[0]["generated_text"][-1]["content"]

        processed_chunk.append(item)

    return processed_chunk


def main():
    model_id = ""
    with open("", "r") as f:
        raw_data = json.load(f)
    # Create exactly 8 chunks
    num_chunks = 8
    chunk_size = len(raw_data) // num_chunks + 1
    chunks = [
        raw_data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks - 1)
    ]
    chunks.append(
        raw_data[(num_chunks - 1) * chunk_size :]
    )  # Add the remaining items to the last chunk

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
    main()
