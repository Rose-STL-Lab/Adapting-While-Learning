import json
from itertools import combinations

tools = """
Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

[
    {
        "type": "function",
        "function": {
            "name": "diy_aerosol_mean",
            "description": "Predict the average temperature of the world in the future under a specific climate scenario with DIY change of SO2 and BC based on the original setting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "setting": {
                        "type": "string",
                        "enum": [
                            "ssp126",
                            "ssp245",
                            "ssp370",
                            "ssp585"
                        ],
                        "description": "Future climate scenarios, a string from ssp126, ssp245, ssp370, ssp585."
                    },
                    "year": {
                        "type": "number",
                        "description": "The year you would check the temperature for, an integer from 2015 to 2100."
                    },
                    "delta_SO2": {
                        "type": "number",
                        "description": "The change of SO2 you would like to make, a float. SO2_after = SO2_before * (1 + delta_SO2)."
                    },
                    "delta_BC": {
                        "type": "number",
                        "description": "The change of BC you would like to make, a float. BC_after = BC_before * (1 + delta_BC)."
                    },
                    "modify_points": {
                        "type": "number",
                        "description": "Points along the line or curve to modify the grid, specified as longitude-latitude pairs. For a single point, use the format '[(longitude, latitude)]'. For multiple points, use '[(longitude1, latitude1), (longitude2, latitude2), ...]'. Longitude ranges from -180 to 180, and latitude from -90 to 90."
                    },
                    "thought": {
                        "type": "string",
                        "description": "Your internal reasoning and thoughts of why you call this function."
                    }
                },
                "required": [
                    "setting",
                    "year",
                    "delta_SO2",
                    "delta_BC",
                    "modification_method",
                    "modify_points"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "is_land_or_sea",
            "description": "Query whether a place is on land or sea with latitude and longitude.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lat": {
                        "type": "number",
                        "description": "The latitude of the place you would check, a float from -90 to 90."
                    },
                    "lon": {
                        "type": "number",
                        "description": "The longitude of the place you would check, a float from -180 to 180."
                    },
                    "thought": {
                        "type": "string",
                        "description": "Your internal reasoning and thoughts of why you call this function."
                    }
                },
                "required": [
                    "lat",
                    "lon"
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
                        "description": "Your answer to this question. It should be a list of locations. If you are going to propose a maritime route, it starts from the given start point and ends at the given ending point. If you are going to propose a transfer station, if is a 3 element list, the first element is the start point, the second element is the transfer station, and the third element is the ending point."
                    },
                    "sloving_process": {
                        "type": "string",
                        "description": "Detailed list how do you solve this question, step by step. If you use tools and got result from it, you should write how the problem was solved based on the output of the code, but don't mention your tool using here."
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
"""

MODEL_ID = ""

def create_datasets(input_file, ft_output_file, rl_output_file):
    with open(input_file, "r") as f:
        data = json.load(f)

    ft_data = []
    rl_data = []

    for question in data:
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

    for question in data:
        valid_answers = [
            ans
            for ans, valid in zip(
                question[MODEL_ID],
                question["valid"],
            )
            if valid
        ]

        if not valid_answers:
            continue

        answers = question[MODEL_ID]
        valids = question["valid"]
        temps = question["temp"]

        # Sort answers by validity (valid first) and then by temperature
        sorted_answers = sorted(
            zip(answers, valids, temps), key=lambda x: (-x[1], x[2])
        )

        sorted_answers_no_temp = [
            (answer, valid) for answer, valid, _ in sorted_answers
        ]

        # Fine-tuning data (best answer: valid if possible, then lowest temp)
        best_answer = sorted_answers[0][0]
        ft_entry = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a climate scientist. You should answer in the following format:\n\nSolution: <solution>\nAnswer: <answer, the list of locations>",
                },
                {"role": "user", "content": f"Question: {question['question']}"},
                {"role": "assistant", "content": best_answer},
            ]
        }
        ft_data.append(ft_entry)

        tool_entry = {
            "messages": [
                {
                    "role": "system",
                    "content": "When you receive a tool call response, use the output to format an answer to the orginal user question.\nYou are a helpful assistant with tool calling capabilities.",
                },
                {
                    "role": "user",
                    "content": tools + f"Question: {question['question']}",
                },
            ]
            + question["llama"][2:]
        }
        ft_data.append(tool_entry)

        # Reinforcement learning data
        for (chosen, chosen_valid), (rejected, rejected_valid) in combinations(
            sorted_answers_no_temp, 2
        ):
            if chosen_valid > rejected_valid or (
                chosen_valid == rejected_valid
                and sorted_answers_no_temp.index((chosen, chosen_valid))
                < sorted_answers_no_temp.index((rejected, rejected_valid))
            ):
                rl_entry = {
                    "conversations": [
                        {
                            "from": "system",
                            "value": "You are a climate scientist. You should answer in the following format:\n\nSolution: <solution>\nAnswer: <answer, the list of locations>",
                        },
                        {"from": "human", "value": f"Question: {question['question']}"},
                    ],
                    "chosen": {"from": "gpt", "value": chosen},
                    "rejected": {"from": "gpt", "value": rejected},
                }
                rl_data.append(rl_entry)

    print(len(ft_data), len(rl_data))

    with open(ft_output_file, "w") as f:
        json.dump(ft_data, f, indent=2)

    with open(rl_output_file, "w") as f:
        json.dump(rl_data, f, indent=2)