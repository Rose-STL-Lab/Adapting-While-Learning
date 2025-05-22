from utils.utils import *
import random
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "tools/Climate_online"))
from functions.functions import *
from tools.emulators import *
import argparse

useful_tools = [
    "location_summary",
    "history_temperature",
    "future_temperature",
    "query_lat_and_lon",
    "diy_greenhouse",
]
functions = [functions_climate[name] for name in useful_tools]
_functions = []
for value in functions:
    if "parameters" in value and "properties" in value["parameters"]:
        value["parameters"]["properties"]["thought"] = {
            "type": "string",
            "description": "Your internal reasoning and thoughts of why you call this function.",
        }
    _functions.append({"type": "function", "function": value})
functions = _functions + function_answer


def func_chain(messages):
    global functions
    while True:
        func_call = llama3.generate(messages, functions)
        messages.append({"role": "assistant", "content": func_call})
        func_name = func_call["name"]
        func_para = func_call["parameters"]
        try:
            if func_name == "answer_question":
                return messages
            else:
                func_para.pop("thought", None)
                _, back_content = globals()[func_name](**func_para)

            print(back_content)

        except Exception as e:
            print(e)
            back_content = f"Error: {e}"
        messages.append({"role": "tool", "name": func_name, "content": back_content})
        if len(messages) > 20:
            return None


system_prompt = """
You are a climate scientist. You are going to answer a multi-choice question. You should use given tools to help you answer the question.
"""

model_id = f""

llama3 = llama(device=f"cuda:0", model_path=model_id)

with open("", "r") as f:
    questions = json.load(f)

for question in questions:
    problem_text = f"Question: {question['Question']}\nOptions:\nA. {question['Options'][0]}\nB. {question['Options'][1]}\nC. {question['Options'][2]}\nD. {question['Options'][3]}"
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": problem_text},
    ]
    question[model_id] = func_chain(messages)

    with open(f"", "w") as f:
        json.dump(questions, f, indent=4)
