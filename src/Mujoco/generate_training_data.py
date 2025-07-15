import random
import json
from functions.functions import *

tools = """
Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

{function}

If you don't know the answer, you can use the tool to help you. If you can answer the problem without the tool, use `answer_question` to answer the problem directly.
"""

def generate_function(typem):
    useful_tools = [typem]
    functions = [functions_mujoco[name] for name in useful_tools]
    _functions = []
    for value in functions:
        if "parameters" in value and "properties" in value["parameters"]:
            value["parameters"]["properties"]["thought"] = {
                "type": "string",
                "description": "Your internal reasoning and thoughts of why you call this function.",
            }
        _functions.append({"type": "function", "function": value})
    functions = _functions + function_answer
    return functions

with open("mujoco.json", "r") as f:
    questions = json.load(f)

random.shuffle(questions)

with open("mujoco_test.json", "w") as f:
    f.write(json.dumps(questions[-280:], indent=4))

for question in questions:
    for i in question["llama"]:
        if i["role"] == "tool":
            i.pop("name", None)
            i["role"] = "user"
        elif i["role"] == "assistant":
            i["content"] = json.dumps(i["function_call"])
            i.pop("function_call", None)
        elif i["role"] == "function":
            i["role"] = "user"
            i["content"] = f"Feedback from {i['name']}: {i['content']}"
            i.pop("name", None)

train_questions = questions

ut = []
nt = []
standard_examples = []

for question in train_questions:
    problem_text = f"Question: {question['Question']}\nOptions:\nA. {question['Options'][0]}\nB. {question['Options'][1]}\nC. {question['Options'][2]}\nD. {question['Options'][3]}"
    
    standard_examples.append({
        "messages": [
            {
                "role": "system",
                "content": "Please answer the following question. Your answer should end with 'the answer is A/B/C/D'.",
            },
            {
                "role": "user",
                "content": problem_text,
            },
            {
                "role": "assistant",
                "content": question["cot"],
            }
        ]
    })
    
    if f"answer: {question['Correct']}".lower() not in question[""].lower():
        ut.append({
            "messages": [
                {
                    "role": "system",
                    "content": "When you receive a tool call response, use the output to format an answer to the orginal user question.\nYou are a helpful assistant with tool calling capabilities.",
                },
                {
                    "role": "user",
                    "content": tools.replace("{function}", json.dumps(generate_function(question["type"] + "_simulation"))) + problem_text,
                }] + question["llama"][-3:]
        })
    else:
        nt.append({
            "messages": [
                {
                    "role": "system",
                    "content": "When you receive a tool call response, use the output to format an answer to the orginal user question.\nYou are a helpful assistant with tool calling capabilities.",
                },
                {
                    "role": "user",
                    "content": tools.replace("{function}", json.dumps(generate_function(question["type"] + "_simulation"))) + problem_text,
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "name": "answer_question",
                        "parameters": {
                            "thought": "I can answer the problem directly",
                            "answer": question['Correct'],
                            "sloving_process": question["cot"]
                        }
                    })
                }
            ]
        })

balance = True

if balance:
    if len(ut) > len(nt):
        longer_list = ut
        shorter_list = nt
    else:
        longer_list = nt
        shorter_list = ut

    multiplication_factor = len(longer_list) // len(shorter_list) if len(shorter_list) > 0 else 1
    
    if shorter_list == ut:
        balanced_data = standard_examples + ut * multiplication_factor + nt
    else:
        balanced_data = standard_examples + ut + nt * multiplication_factor
else:
    balanced_data = standard_examples + ut + nt

with open("mujoco_train.json", "w") as f:
    f.write(json.dumps(balanced_data, indent=4))