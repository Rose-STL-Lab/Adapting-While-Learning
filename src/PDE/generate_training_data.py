import random
import json

tools = """
Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

[
    {
        "type": "function",
        "function": {
            "name": "setup_and_solve",
            "description": "Set up the PDE problem parameters and solve the diffusion equation.",
            "properties": {
                "type": "object",
                "properties": {
                    "dimension": {
                        "type": "string",
                        "enum": [
                            "1D",
                            "2D"
                        ],
                        "description": "Dimension of the PDE problem. Can be '1D' or '2D'."
                    },
                    "D": {
                        "type": "number",
                        "description": "Diffusion coefficient, a measure of how fast diffusion occurs."
                    },
                    "T": {
                        "type": "number",
                        "description": "Final time of the simulation."
                    },
                    "L": {
                        "type": "number",
                        "description": "Length of the domain in the x-direction (only for 1D dimension)."
                    },
                    "Lx": {
                        "type": "number",
                        "description": "Length of the domain in the x-direction (only for 2D diffusion)."
                    },
                    "Ly": {
                        "type": "number",
                        "description": "Length of the domain in the y-direction (only for 2D diffusion)."
                    },
                    "ic_type": {
                        "type": "string",
                        "enum": [
                            "constant",
                            "step",
                            "checkerboard"
                        ],
                        "description": "Type of the initial condition."
                    },
                    "ic_params": {
                        "type": "object",
                        "description": "Parameters for the initial condition, specified as a JSON object. The content depends on the type of initial condition chosen."
                    },
                    "bc_type": {
                        "type": "string",
                        "enum": [
                            "dirichlet",
                            "neumann",
                            "periodic"
                        ],
                        "description": "Type of the boundary condition."
                    },
                    "bc_params": {
                        "type": "object",
                        "description": "Parameters for the boundary condition, specified as a JSON object. The content depends on the type of boundary condition chosen."
                    },
                    "solution_type": {
                        "type": "string",
                        "enum": [
                            "get_max_value",
                            "get_min_value",
                            "get_mean_value",
                            "get_value_at_point",
                            "get_max_gradient"
                        ],
                        "description": "Type of solution to retrieve after solving the diffusion equation."
                    },
                    "solution_x": {
                        "type": "number",
                        "description": "The x-coordinate of the point to get the value"
                    },
                    "solution_y": {
                        "type": "number",
                        "description": "The y-coordinate of the point to get the value (only for 2D diffusion)."
                    }
                },
                "required": [
                    "dimension",
                    "D",
                    "T",
                    "ic_type",
                    "ic_params",
                    "bc_type",
                    "bc_params",
                    "solution_type"
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
                    },
                    "sloving_process": {
                        "type": "string",
                        "description": "Detailed list how do you solve this question, step by step. If you wrote code and got result from it, you should write how the problem was solved based on the output of the code, but don't mention your coding here."
                    }
                },
                "required": [
                    "thought",
                    "answer",
                    "sloving_process"
                ]
            }
        }
    }
]

If you don't know the answer, you can use the tool to help you. If you can answer the problem without the tool, use `answer_question` to answer the problem directly.
"""

MODEL_ID = ""

data = []
ut = []
nt = []
standard_examples = []

with open("diffusion_questions.json", "r") as f:
    questions = json.load(f)

cnt = 0


questions = [d for d in questions if "wrong" not in d["cot"].lower() and "answer" in d["cot"].lower()]

correct = []
incorrect = []

for d in questions:
    if f"Answer: {d['correct']}".lower() in d["cot"].lower():
        cnt += 1
        correct.append(d)
    else:
        incorrect.append(d)


random.shuffle(questions)

with open("pde_test.json", "w") as f:
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
            i.pop("name", None)

train_questions = questions

for question in train_questions:
    problem_text = f"Question: {question['question']}\nOptions:\nA. {question['options'][0]}\nB. {question['options'][1]}\nC. {question['options'][2]}\nD. {question['options'][3]}"

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
                "content": question[MODEL_ID],
            }
        ]
    })
    
    if f"answer: {question['correct']}".lower() not in question[MODEL_ID].lower():
        ut.append({
            "messages": [
                {
                    "role": "system",
                    "content": "When you receive a tool call response, use the output to format an answer to the orginal user question.\nYou are a helpful assistant with tool calling capabilities.",
                },
                {
                    "role": "user",
                    "content": tools + problem_text,
                }
            ] + question["llama"][-3:]
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
                    "content": tools + problem_text,
                },
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "name": "answer_question",
                        "parameters": {
                            "thought": "I can answer the problem directly",
                            "answer": question[MODEL_ID].split("Answer:")[-1].strip(),
                            "sloving_process": question[MODEL_ID],
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
    
    if len(shorter_list) > 0:
        multiplication_factor = len(longer_list) // len(shorter_list)
        
        if shorter_list == ut:
            balanced_data = standard_examples + ut * multiplication_factor + nt
        else:
            balanced_data = standard_examples + ut + nt * multiplication_factor
    else:
        balanced_data = standard_examples + longer_list
    
else:
    balanced_data = standard_examples + ut + nt

with open("pde_train_aba1.json", "w") as f:
    f.write(json.dumps(balanced_data, indent=4))