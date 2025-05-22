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
PF_KEY = ""

def preprocess_conversation(messages):
    processed = []
    
    if messages[0]["role"] == "system":
        processed.append({
            "role": "system",
            "content": "When you receive a tool call response, use the output to format an answer to the orginal user question.\nYou are a helpful assistant with tool calling capabilities."
        })
    
    initial_user_message = next(msg for msg in messages if msg["role"] == "user")
    processed.append(initial_user_message)
    
    assistant_responses = []
    i = 0
    while i < len(messages):
        current = messages[i]
        
        if current["role"] == "assistant":
            content = current.get("function_call")
            if content:
                content = json.dumps({"name": content["name"], "parameters": content["parameters"]})
            else:
                content = current.get("content")
                
            if content and content not in assistant_responses:
                processed.append({
                    "role": "assistant",
                    "content": content
                })
                assistant_responses.append(content)
                
                if i + 1 < len(messages) and messages[i + 1]["role"] in ["user", "function"]:
                    processed.append({
                        "role": "user",
                        "content": messages[i + 1]["content"]
                    })
                    i += 1
        i += 1
    
    return processed

cnt1 = 0
with open("open_all1.json", "r") as f:
    data = json.load(f)
with open("open_all.json", "r") as f:
    data += json.load(f)

ft_data = []

for question in data:
    messages = question[PF_KEY]
    processed_messages = preprocess_conversation(messages)
    question[PF_KEY] = processed_messages

for question in data:
    answer = question["cot"]
    answer_valid = question["cot_eval"][0]["valid"]
    answer_temperature = question["cot_eval"][0]["temperature"]
            
    if question["cot_eval"][0]["valid"]:
        ft_entry = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a climate scientist. You should answer in the following format:\n\nSolution: <solution>\nAnswer: <answer, the list of locations>",
                },
                {"role": "user", "content": f"Question: {question['question']}"},
                {"role": "assistant", "content": answer},
            ]
        }
        ft_data.append(ft_entry)
    a = question[MODEL_ID + "_eval"]
    if sum(1 for x in a if x["temperature"] is not None and x["valid"] is not None and x["temperature"] < 0.01 and x["valid"]) > 3:
        if question["cot_eval"][0]["valid"]:
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
                    {
                        "role": "assistant",
                        "content": json.dumps(
                            {
                                "name": "answer_question",
                                "parameters": {
                                    "thought": "I can answer the problem directly",
                                    "answer": answer.split("Answer: ")[1],
                                    "sloving_process": answer,
                                },
                            }
                        ),
                    },
                ]
            }
            ft_data.append(tool_entry)
        cnt1 += 1
    else:
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
            + question[PF_KEY][2:]
        }
        ft_data.append(tool_entry)

print(len(ft_data), cnt1)

# Write fine-tuning data
with open("open_sft_2.json", "w") as f:
    json.dump(ft_data, f, indent=2)
