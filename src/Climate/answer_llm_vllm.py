from jload import jload, jsave
from vlllm import generate
import random

instruction = """
You are a climate scientist. Answer the following question. Your answer should end with 'the answer is A/B/C/D'.
"""

def process_entry(item):
    return  f"Question: {item['Question']}\nOptions:\nA. {item['Options'][0]}\nB. {item['Options'][1]}\nC. {item['Options'][2]}\nD. {item['Options'][3]}"

models = ["meta-llama/Llama-3.1-8B-Instruct"]


if __name__ == "__main__":
    for MODEL_ID in models:

        data = jload("test.json")

        random.shuffle(data)

        for d in data:
            d["input"] = process_entry(d)

        processed_data = generate(
            model_id=MODEL_ID,
            data=[item.copy() for item in data if item["input"]],
            system=instruction,
            message_key="input",
            tp=2,
            pp=1,
            n=1,
            worker_num=1,
            temperature=0.1,
            use_sample=False,
            result_key=MODEL_ID + "pn",
            max_model_len=8000,
            max_output_len=4000,
            chunk_size=None,
            gpu_assignments=None,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            dtype="auto"
        )

        jsave(processed_data, "test.json")