import json
import random
import argparse

tools = """
Given the following functions, please respond with a JSON for a function call with its proper arguments that best answers the given prompt.

Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. Do not use variables.

[
    {
        "type": "function",
        "function": {
            "name": "location_summary",
            "description": "Retrieve the temperature of a place in 1850, 1900, 1950, 2000, and predicted temperature under difference scenarios in 2050 and 2100.",
            "parameters": {
                "type": "object",
                "properties": {
                    "longitude": {
                        "type": "number",
                        "description": "The longitude of the place you would check the temperature for, a float from -180 to 180."
                    },
                    "latitude": {
                        "type": "number",
                        "description": "The latitude of the place you would check the temperature for, a float from -90 to 90."
                    },
                    "thought": {
                        "type": "string",
                        "description": "Your internal reasoning and thoughts of why you call this function."
                    }
                },
                "required": [
                    "longitude",
                    "latitude"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "history_temperature",
            "description": "Retrieve the temperature of a place from 1850 to 2014 with longitude and latitude.",
            "parameters": {
                "type": "object",
                "properties": {
                    "longitude": {
                        "type": "number",
                        "description": "The longitude of the place you would check the temperature for, a float from -180 to 180."
                    },
                    "latitude": {
                        "type": "number",
                        "description": "The latitude of the place you would check the temperature for, a float from -90 to 90."
                    },
                    "year": {
                        "type": "number",
                        "description": "The year you would check the temperature for, an integer from 1850 to 2014."
                    },
                    "thought": {
                        "type": "string",
                        "description": "Your internal reasoning and thoughts of why you call this function."
                    }
                },
                "required": [
                    "longitude",
                    "latitude",
                    "year"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "future_temperature",
            "description": "Retrieve the temperature of a place from 2015 to 2100 under different climate scenarios with longitude and latitude.",
            "parameters": {
                "type": "object",
                "properties": {
                    "longitude": {
                        "type": "number",
                        "description": "The longitude of the place you would check the temperature for, a float from -180 to 180."
                    },
                    "latitude": {
                        "type": "number",
                        "description": "The latitude of the place you would check the temperature for, a float from -90 to 90."
                    },
                    "year": {
                        "type": "number",
                        "description": "The year you would check the temperature for, an integer from 2015 to 2100."
                    },
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
                    "thought": {
                        "type": "string",
                        "description": "Your internal reasoning and thoughts of why you call this function."
                    }
                },
                "required": [
                    "longitude",
                    "latitude",
                    "year",
                    "setting"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "query_lat_and_lon",
            "description": "Retrieve the latitude and longitude of a place with the name.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city_name": {
                        "type": "string",
                        "description": "The name of the place you would check the latitude and longitude for, a string."
                    },
                    "thought": {
                        "type": "string",
                        "description": "Your internal reasoning and thoughts of why you call this function."
                    }
                },
                "required": [
                    "city_name"
                ]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "diy_greenhouse",
            "description": "Predict the temperature of a place in the future under a specific climate scenario with DIY change of CO2 and CH4 based on the original setting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "longitude": {
                        "type": "number",
                        "description": "The longitude of the place you would check the temperature for, a float from -180 to 180."
                    },
                    "latitude": {
                        "type": "number",
                        "description": "The latitude of the place you would check the temperature for, a float from -90 to 90."
                    },
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
                    "delta_CO2": {
                        "type": "number",
                        "description": "The change of CO2 you would like to make, a float. CO2_after = CO2_before * (1 + delta_CO2)."
                    },
                    "delta_CH4": {
                        "type": "number",
                        "description": "The change of CH4 you would like to make, a float. CH4_after = CH4_before * (1 + delta_CH4)."
                    },
                    "thought": {
                        "type": "string",
                        "description": "Your internal reasoning and thoughts of why you call this function."
                    }
                },
                "required": [
                    "longitude",
                    "latitude",
                    "setting",
                    "year",
                    "delta_CO2",
                    "delta_CH4"
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

def make_data(model_id):
    with open("climate_train.json", "r") as f:
        questions = json.load(f)

    for question in questions:
        for i in question["meta-llama/Llama-3.1-8B-Instruct"]:
            if i["role"] == "tool":
                i.pop("name", None)
                i["role"] = "user"
            # elif i["role"] == "assistant":
            #     i["content"] = json.dumps(i["content"])

    for question in questions:
        problem_text = f"Question: {question['Question']}\nOptions:\nA. {question['Options'][0]}\nB. {question['Options'][1]}\nC. {question['Options'][2]}\nD. {question['Options'][3]}"
        if (
            f"the answer is {question['Correct']}".lower()
            not in question[model_id].lower()
        ):
            ut.append(
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
                    ]
                    + question["meta-llama/Llama-3.1-8B-Instruct"][2:]
                }
            )
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
                                        "answer": question["Correct"],
                                    },
                                }
                            ),
                        },
                    ]
                }
            )

    for question in questions:
        if "the answer is" not in question["solution"].lower():
            continue

        problem_text = f"Question: {question['Question']}\nOptions:\nA. {question['Options'][0]}\nB. {question['Options'][1]}\nC. {question['Options'][2]}\nD. {question['Options'][3]}"
        data.append(
            {
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
                        "content": question["solution"],
                    },
                ]
            }
        )

if __name__ == "__main__":
    balance = True
    make_data("sft1_base")

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
    
    with open("train4.json", "w") as f:
        f.write(json.dumps(balanced_data, indent=4))
