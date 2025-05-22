import json
import math
from fractions import Fraction
from utils.math_utils import *
from tqdm import tqdm

import random

instruction = """
Please answer the following given question. You should first give the Solution then give the answer. Your answer should be in the following format:
Solution: <Your solution>
Answer: <Your answer, a number>
"""

instruction_tool = """Please answer the following given question. You should first give the Solution then give the answer. You can write a code to solve the problem or give the answer directly. Your answer should be in the following format:
Solution: <Your solution>
Answer: <Your answer, a number>

Each time, you should whether write a code or answer the question.

If you want to write a code, you answer should be in the following format:

Thought: <Your thought>
Action: write_and_run_code
Code:
```python
<Your code>
```

If you want to answer the question, you should answer in the following format:

Thought: <Your thought>
Action: answer_question
Solution: <Your solution>
Answer: <Your answer>
"""

with open("matht1.json", "r") as f:
    data = json.load(f)

test = []
with open("test.jsonl", "r") as f:
    for line in f:
        test.append(json.loads(line))

test_id = set()
for t in test:
    test_id.add(t["unique_id"])

cnt = 0
cnt_tool = 0
done = 0

train_dataset = []

print(len(test_id))

ccc = 0

for item in tqdm(data):
    if item["unique_id"] in test_id:
        continue
    if "_int" not in item or not item["_int"]:
        continue
    if "solution" not in item["_int"][-1]["content"].lower():
        continue
    if "answer" not in item["_int"][-1]["content"].lower():
        continue
    if len(item["_int"][-1]["content"].split("Answer:")) < 2:
        continue
    if len(item["_int"]) <= 3:
        continue
    answer = item[""].split("Answer:")[-1].strip()
    try:
        solution = item["cot"] 
        if "wrong" in solution.lower():
            continue
        if "answer:" not in solution.lower():
            continue
        if "solution:" not in solution.lower():
            continue
        train_dataset.append({"messages": [
            {
                "role": "system",
                "content": instruction
            },
            {
                "role": "user",
                "content": f"Problem: {item['problem']}",
            },
            {
                "role": "assistant",
                "content": solution
            }
        ]})
        if not equiv(answer, item["answer"]):
            train_dataset.append({"messages": [{
                "role": "system",
                "content": instruction_tool
            }] + [item["_int"][1]] + item["_int"][-3:]})
        else:
            train_dataset.append({"messages": [{
                "role": "system",
                "content": instruction_tool
            },{
                "role": "user",
                "content": f"Problem: {item['problem']}",
            },{
                "role": "assistant",
                "content": f"Thought: I can answer the problem directly.\nAction: answer_question\n{solution}"
            }]})
            cnt += 1
        cnt_tool += 1
    except Exception as e:
        print(f"Error in second loop: {e}")
        print(answer, item["answer"])

print(f"Count for second model: {cnt_tool}")
print(f"Total items: {len(data)}")
print(f"Accuracy for second model: {cnt / cnt_tool}")
print(f"Accuracy for cc: {ccc / cnt_tool}")

print(len(train_dataset))

with open("train_dataset_math.json", "w") as f:
    json.dump(train_dataset, f, indent=4)
    print("Saved train_dataset_math.json")
