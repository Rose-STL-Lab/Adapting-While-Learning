import sys
import os

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(src_dir)

from utils.utils import *
from functions.functions import *
from utils.make_problem_utils import *
import pandas as pd
import argparse
import json
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "tools/Climate_online"))
from tools.emulators import *

import random
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

# You are a climate scientist. You are going to answer a multi-choice question. You should use given tools to help you answer the question, or you can also answer the problem directly. You can call tools for many turns, but you should call only one tool each time. When you have gathered enough information, you should use answer_question to choose one answer from A/B/C/D.

system_prompt = """
You are a climate scientist. You are going to answer a multi-choice question. You should use given tools to help you answer the question.
"""

model_id = f"/home/test/test12/bohan/models/Meta-Llama-3.1-8B-Instruct"

llama3 = llama(device=f"cuda:0", model_path=model_id)

with open("/home/test/test12/bohan/PGLLM-2/src/Climate/climate_train.json", "r") as f:
    questions = json.load(f)

with open("few_shot.txt", "r") as f:
    few_shot = f.read()

for question in questions:
    problem_text = f"Question: {question['Question']}\nOptions:\nA. {question['Options'][0]}\nB. {question['Options'][1]}\nC. {question['Options'][2]}\nD. {question['Options'][3]}"
    print(problem_text)
    print(question["Correct"])
    messages = [
        {
            "role": "system",
            "content": system_prompt + few_shot,
        },
        {"role": "user", "content": problem_text + "\n\nYou must firstly use `query_lat_and_lon` to get the latitude and longitude of the given place, even if you think you know the latitude and longitude of the place."},
    ]
    question[model_id] = func_chain(messages)

    with open(f"test.json", "w") as f:
        json.dump(questions, f, indent=4)
