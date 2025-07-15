import json
import random

tools = """
Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

[
    {
        "type": "function",
        "function": {
            "name": "emulate",
            "description": "Emulates the pandemic by giving the output feature that you want. This fuction will give you the value of the feature in following days.",
            "parameters": {
                "type": "object",
                "properties": {
                    "out": {
                        "type": "integer",
                        "description": "The output name.",
                        "enum": [
                            "prevalence_CA_state_total_Latent",
                            "prevalence_CA_state_total_Infectious_symptomatic",
                            "prevalence_CA_state_total_Infectious_asymptomatic",
                            "prevalence_CA_state_total_Hospitalized",
                            "prevalence_CA_state_total_ICU",
                            "prevalence_CA_state_total_Removed_asymptomatic",
                            "prevalence_CA_state_total_Removed_symptomatic",
                            "prevalence_CA_state_total_Home_asymptomatic",
                            "prevalence_CA_state_total_Home_mild",
                            "prevalence_CA_state_total_Home_severe",
                            "prevalence_CA_state_total_Removed_hospitalized",
                            "prevalence_CA_state_total_Deaths_hospitalized",
                            "incidence_CA_state_total_Latent",
                            "incidence_CA_state_total_Infectious_symptomatic",
                            "incidence_CA_state_total_Infectious_asymptomatic",
                            "incidence_CA_state_total_Hospitalized",
                            "incidence_CA_state_total_ICU",
                            "incidence_CA_state_total_Removed_asymptomatic",
                            "incidence_CA_state_total_Removed_symptomatic",
                            "incidence_CA_state_total_Home_asymptomatic",
                            "incidence_CA_state_total_Home_mild",
                            "incidence_CA_state_total_Home_severe",
                            "incidence_CA_state_total_Removed_hospitalized",
                            "incidence_CA_state_total_Deaths_hospitalized"
                        ]
                    },
                    "thought": {
                        "type": "string",
                        "description": "Your internal reasoning and thoughts of why you call this function."
                    }
                },
                "required": [
                    "out"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "answer_question",
            "description": "Answer the Question.",
            "parameters": {
                "type": "object",
                "properties": {
                    "thought": {
                        "type": "string",
                        "description": "Internal reasoning and thoughts of why you call this function."
                    },
                    "sloving_process": {
                        "type": "string",
                        "description": "Detailed list how do you solve this question, step by step. If you wrote code and got result from it, you should write how the problem was solved based on the output of the code, but don't mention your coding here."
                    },
                    "answer": {
                        "type": "string",
                        "enum": [
                            "A",
                            "B",
                            "C",
                            "D"
                        ],
                        "description": "Your answer to this question. If you have multiple answers, you can write them all. If none of the answers are correct, you can give your answer as well."
                    }
                },
                "required": [
                    "thought",
                    "sloving_process",
                    "answer"
                ]
            }
        }
    }
]
"""

data = []
ut = []
nt = []

with open("train.json", "r") as f:
    questions = json.load(f)

with open("/home/test/test12/bohan/PGLLM-2/test_set/epidemiology.json", "r") as f:
    tests = json.load(f)

test_questions = [i["question"] for i in tests]

for question in questions:
    for i in question["meta-llama/Llama-3.1-8B-Instruct"]:
        if i["role"] == "tool":
            i.pop("name", None)
            i["role"] = "user"

for question in questions:
    if question["question"] in test_questions:
        continue
    if f"the answer is".lower() not in question["solution"].lower():
        continue
    problem_text = f"Question: {question['question']}\nOptions:\nA. {question['options'][0]}\nB. {question['options'][1]}\nC. {question['options'][2]}\nD. {question['options'][3]}"
    if f"the answer is {question['correct_option']}".lower() not in question["/home/test/test12/bohan/ws/LLaMA-Factory/pan1pn"].lower():
        ut.append({
            "messages": [
                {
                    "role": "system",
                    "content": "When you receive a tool call response, use the output to format an answer to the orginal user question.\nYou are a helpful assistant with tool calling capabilities.",
                },
                {
                    "role": "user",
                    "content": tools + problem_text,
                }] + question["meta-llama/Llama-3.1-8B-Instruct"][2:]})
    else:
        nt.append(
            {
                "messages": [
                    {
                        "role": "system",
                        "content": "When you receive a tool call response, use the output to format an answer to the orginal user question.\nYou are a helpful assistant with tool calling capabilities.",
                    },
                    {
                        "role": "user",
                        "content": tools + problem_text,
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "name": "answer_question",
                                "parameters": {
                                    "thought": "I can answer the problem directly",
                                    "sloving_process": question["solution"],
                                    "answer": question["correct_option"],
                                },
                            }
                        ),
                    },
                ]
            }
        )
    data.append({
        "messages": [
            {
                "role": "system",
                "content": "You are a epidemiologist. Please answer the following question. Your answer should end with 'the answer is A/B/C/D'.",
            },
            {
                "role": "user",
                "content": problem_text,
            },
            {
                "role": "assistant",
                "content": question["solution"],
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
    
    multiplication_factor = len(longer_list) // len(shorter_list)
    
    if shorter_list == ut:
        balanced_data = data + ut * multiplication_factor + nt
    else:
        balanced_data = data + ut + nt * multiplication_factor

else:
    balanced_data = data + ut + nt

with open("train2.json", "w") as f:
    f.write(json.dumps(balanced_data, indent=4))