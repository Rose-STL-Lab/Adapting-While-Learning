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
                    "answer"
                ]
            }
        }
    }
]
"""

data = []

with open("qa_pairs_final.json", "r") as f:
    questions = json.load(f)[:1800]

for question in questions:
    for i in question["gpt4"]:
        if i["role"] == "tool":
            i.pop("name", None)
            i["role"] = "user"
        elif i["role"] == "assistant":
            i["content"] = json.dumps(i["content"])

for question in questions:
    if f"Answer:" not in question["cot"] and f"the answer is" not in question["cot"].lower():
        continue
    if f"Answer: {question["correct_option"]}" not in question["cot"] and f"the answer is {question["correct_option"].lower()}" not in question["cot"].lower() and random.random()<0.5:
        print(question["cot"])
        continue
    if "wrong" in question["cot"]:
        continue
    problem_text = f"Question: {question['question']}\nOptions:\nA. {question['options'][0]}\nB. {question['options'][1]}\nC. {question['options'][2]}\nD. {question['options'][3]}"
    data.append({
        "messages": [
            {
                "role": "system",
                "content": "When you receive a tool call response, use the output to format an answer to the orginal user question.\nYou are a helpful assistant with tool calling capabilities.",
            },
            {
                "role": "user",
                "content": tools + problem_text,
            }] + question["gpt4"][2:]})
    data.append({
        "messages": [
            {
                "role": "system",
                "content": "You are a epidemiologist. Please answer the following question. Your answer should be in the following format:\nSolution: <your solution>\nAnswer: <your answer, one of A/B/C/D>",
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

print(len(data))

with open("pan_train.json", "w") as f:
    f.write(json.dumps(data, indent = 4))