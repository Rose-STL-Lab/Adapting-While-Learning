from jload import jload, jsave
from vlllm import generate
import random

instruction = """
You are a epidemiologist. Please answer the following question. Your answer should end with 'the answer is A/B/C/D'.
"""

def process_entry(item):
    return  f"Question: {item['question']}\nOptions:\nA. {item['options'][0]}\nB. {item['options'][1]}\nC. {item['options'][2]}\nD. {item['options'][3]}"

models = ["meta-llama/Llama-3.1-8B-Instruct"]


if __name__ == "__main__":
    for MODEL_ID in models:

        data = jload("test.json")

        random.shuffle(data)

        for d in data:
            d["input"] = process_entry(d)
        
        print(len(data[0]["input"]))

        processed_data = generate(
            model_id=MODEL_ID,
            data=[item.copy() for item in data if item["input"]],
            system=instruction,
            message_key="input",
            tp=2,
            pp=1,
            n=1,
            worker_num=4,
            temperature=0.1,
            use_sample=False,
            result_key=MODEL_ID + "pn",
            max_model_len=24000,
            max_output_len=4000,
            chunk_size=None,
            gpu_assignments=None,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            dtype="auto"
        )

        jsave(processed_data, "test.json")
