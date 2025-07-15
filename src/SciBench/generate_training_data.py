import json
from math_utils import *
from tqdm import tqdm

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

with open("scibench.json", "r") as f:
    data = json.load(f)

with open("sci_final.json", "r") as f:
    test = json.load(f)

test_questions = [i["problem_text"] for i in test]

cnt = 0
cnt_tool = 0
done = 0

MODEL_ID = ""
train_dataset = []
ut = []
nt = []

for item in tqdm(data):
    is_test = False
    for i in test_questions:
        if i in item["problem"]:
            is_test = True
    if is_test:
        continue
    if f"{MODEL_ID}_int" not in item or not item[f"{MODEL_ID}_int"]:
        continue
    if "solution" not in item[f"{MODEL_ID}_int"][-1]["content"].lower():
        continue
    if "answer" not in item[f"{MODEL_ID}_int"][-1]["content"].lower():
        continue
    if len(item[f"{MODEL_ID}_int"][-1]["content"].split("Answer:")) < 2:
        continue
    if len(item[f"{MODEL_ID}_int"]) <= 3:
        continue
    answer = item[MODEL_ID].split("Answer:")[-1].strip()
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
            ut.append({"messages": [{
                    "role": "system",
                    "content": instruction_tool
                }] + [item[f"{MODEL_ID}_int"][1]] + item[f"{MODEL_ID}_int"][-3:]})
        else:
            nt.append({"messages": [{
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
        pass

balance = True

if balance:
    if len(ut) > len(nt):
        longer_list = ut
        shorter_list = nt
    else:
        longer_list = nt
        shorter_list = ut
    
    if len(shorter_list) > 0:
        multiplication_factor = len(longer_list) // len(shorter_list)
        
        if shorter_list == ut:
            balanced_data = train_dataset + ut * multiplication_factor + nt
        else:
            balanced_data = train_dataset + ut + nt * multiplication_factor
    else:
        balanced_data = train_dataset + longer_list
else:
    balanced_data = train_dataset + ut + nt

with open("train_dataset_sci.json", "w") as f:
    json.dump(balanced_data, f, indent=4)