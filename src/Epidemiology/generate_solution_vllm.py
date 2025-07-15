from jload import jload, jsave
from vlllm import generate
import random

instruction = """
You are a epidemiologist. Your task is to provide a detailed solution process for a given questions. The question was answered with emulator, but you shouldn't mention the emulator using procedures in your answer. The solution should end with 'the answer is A/B/C/D'.

You should provide a solution process that includes:
- Relevant data from the emulator (but you shouldn't mention the simulations and simulators, you should directly list relevant data)
- Detailed analysis of this data
- Logical reasoning connecting the data to the question

If you think the answer is wrong, you should answer 'The answer is wrong' and explain why.
"""

def process_entry(item):
    """
    Process a single entry to extract the prompt and question.
    """
    if "/home/test/test12/bohan/models/Meta-Llama-3.1-8B-Instruct" not in item or not item["/home/test/test12/bohan/models/Meta-Llama-3.1-8B-Instruct"]:
        return None
    messages = item["/home/test/test12/bohan/models/Meta-Llama-3.1-8B-Instruct"]
    prompt = "Following is the problem and the interaction for answering the question.\n\n"
    for message in messages:
        if message["role"] == "user":
            prompt += message["content"]
        elif message["role"] == "assistant":
            prompt += f"\nAssistant calling function: {message['content']}."
        elif message["role"] == "tool":
            prompt += f"\nFeedback from `{message['name']}`: {message['content']}."

    # prompt += "Following is the question that you should write solution for.\n\n"

    problem_text = f"Question: {item['question']}\nOptions:\nA. {item['options'][0]}\nB. {item['options'][1]}\nC. {item['options'][2]}\nD. {item['options'][3]}"

    return prompt + "\n\nNow begin to write solution for this question based on the data, question and answer."



if __name__ == "__main__":
    data = jload("train.json")

    random.shuffle(data)

    for d in data:
        d["input"] = process_entry(d)
    
    print(data[0]["input"])

    processed_data = generate(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        data=[item.copy() for item in data if item["input"]],
        system=instruction,
        message_key="input",
        tp=2,
        pp=1,
        n=1,
        worker_num=4,
        temperature=0.1,
        use_sample=False,
        result_key="solution",
        max_model_len=28000,
        max_output_len=4000,
        chunk_size=None,
        gpu_assignments=None,
        trust_remote_code=True,
        gpu_memory_utilization=0.95,
        dtype="auto"
    )

    jsave(processed_data, "train.json")