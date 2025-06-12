import random
import json
import sys
import os

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(src_dir)

from utils.utils import *
from functions.functions import *

sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
from tools.inference import *

useful_tools = ["emulate"]
functions = [functions_pandemic[name] for name in useful_tools]
_functions = []
for value in functions:
    if "parameters" in value and "properties" in value["parameters"]:
        value["parameters"]["properties"]["thought"] = {
            "type": "string",
            "description": "Your internal reasoning and thoughts of why you call this function.",
        }
    _functions.append({"type": "function", "function": value})
functions = _functions + function_answer

print(json.dumps(functions, indent = 4))

model_id = f"/home/test/test12/bohan/models/Meta-Llama-3.1-8B-Instruct"

llama3 = llama(device=f"cuda:0", model_path=model_id)

def func_chain(messages, scenario):
    global functions
    tried = 0
    while True:
        tried += 1
        func_call = llama3.generate(messages, functions)
        if not func_call:
            continue
        messages.append(
            {"role": "assistant", "content": func_call}
        )
        func_name = func_call["name"]
        func_para = func_call["parameters"]
        try:
            if func_name == "answer_question":
                if len(messages) > 3:
                    return messages
                else:
                    back_content = "You should use more tools to help you answer the question."
            else:
                func_para.pop("thought", None)
                func_para["scenario"] = scenario
                back_content = globals()[func_name](**func_para)

            print(back_content)
            
        except Exception as e:
            print(e)
            back_content = f"Error: {e}"
        messages.append(
            {"role": "tool", "name": func_name, "content": back_content}
        )
        if len(messages) > 20 or tried > 10:
            return None

system_prompt = """
You are a epidemiologist. You are going to answer a multi-choice question. You should use given tools to help you answer the question. You can call tools for many turns, but you should call only one tool each time. When you have used `emulate` function to get enough information, you should use answer_question to choose one answer from A/B/C/D.
"""

with open("../../test_set/epidemiology.json", "r") as f:
    questions = json.load(f)

for question in questions:
    problem_text = f"Question: {question['question']}\n\nOptions: A. {question['options'][0]}\nB. {question['options'][1]}\nC. {question['options'][2]}\nD. {question['options'][3]}"
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": problem_text},
    ]
    question["gpt4"] = func_chain(messages, question['scenario'])
    print(problem_text[-500:])
    print(question["correct_option"])
    with open("test.json", "w") as f:
        json.dump(questions, f, indent=4)




