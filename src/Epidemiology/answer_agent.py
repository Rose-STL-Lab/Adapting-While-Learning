from utils import *
import random
import json
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'tools/pandemic'))
from functions.functions import *
from tools.emulators import *

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

def func_chain(messages, scenario):
    global functions
    while True:
        func_call = gpt4_functions(messages, functions)
        if not func_call:
            return None
        messages.append(
            {"role": "assistant", "content": None, "function_call": func_call}
        )
        func_name = func_call["name"]
        func_para = json.loads(func_call["arguments"])
        
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
        except Exception as e:
            print(e)
            back_content = f"Error: {e}"
            
        print(back_content)
        messages.append(
            {"role": "function", "name": func_name, "content": back_content}
        )
        if len(messages) > 10:
            return None

system_prompt = """
You are a epidemiologist. You are going to answer a multi-choice question. You should use given tools to help you answer the question. You can call tools for many turns, but you should call only one tool each time. When you have used `emulate` function to get enough information, you should use answer_question to choose one answer from A/B/C/D.
"""

with open("", "r") as f:
    questions = json.load(f)

for question in questions:
    if "gpt4" in question and question["gpt4"] and json.loads(question["gpt4"][-1]["function_call"]["arguments"])["answer"] == question["correct_option"]:
        continue
    problem_text = f"Question: {question['question']}\n\nOptions: A. {question['options'][0]}\nB. {question['options'][1]}\nC. {question['options'][2]}\nD. {question['options'][3]}"
    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {"role": "user", "content": problem_text},
    ]
    question["gpt4"] = func_chain(messages, question['scenario'])
    print(question["correct_option"])
    with open("", "w") as f:
        json.dump(questions, f, indent=4)
        print("saved!")




