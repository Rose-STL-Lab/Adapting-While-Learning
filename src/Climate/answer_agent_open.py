import json
import os
import random
import sys
import traceback

import torch.multiprocessing as mp
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(src_dir)
from utils.utils import *
from functions.functions import *

sys.path.append(os.path.join(os.path.dirname(__file__), "tools/Climate_online"))
from tools.emulators import *

functions = functions_climate
_functions = []
for key, value in functions.items():
    if "parameters" in value and "properties" in value["parameters"]:
        value["parameters"]["properties"]["thought"] = {
            "type": "string",
            "description": "Your internal reasoning and thoughts of why you call this function.",
        }
    _functions.append({"type": "function", "function": value})
functions = _functions[-2:] + function_answer_open

print(functions)

system_prompt = """
You are a climate scientist. You are going to answer a given question. You can use given tools to help you answer the question. You should pass a list of (lon_i, lat_i)s, start from the given start point and end at the given ending point, to `modify_points` of function `diy_aerosol_mean` along the proposed route. You can simulate many times and get the average temperature increase in the future. You should choose the best proposal as the final answer. You should also use `is_land_or_sea` to check whether a point is on land or sea. You don't need to check the condition of the start point and end point. If the question ask you to propose a maritime route, you need to check points in the middle way of the route to make sure they are in the sea. If the question ask you to propose a transfer station, you need to make sure it is on the land. You should use `answer_question` to answer the question. You should use `is_land_or_sea` for no more than 5 times and use `diy_aerosol_mean` for no more than 3 times. You should use `diy_aerosol_mean` for at least 2 times before you give the final answer. You can call tools for many turns, but you should call only one tool each time. You can use tools to solve the problem or answer it directly.

Examples:

Question: Design a new maritime route between Jeddah (lon: 39.1728, lat: 21.5433) and Singapore (lon: 103.8, lat: 1.3) that would increase BC levels along the route by 20.0%. Propose a route that would minimize the global average temperature increase in 2039 under ssp245. Present your answer as a list of coordinates (longitude, latitude) representing key points along the route. Format your response as follows: [(longitude_1, latitude_1), (longitude_2, latitude_2), ..., (longitude_n, latitude_n)]. Include at least the starting point, endpoint, and any significant waypoints. Ensure that the distance between any two consecutive points in your list is no less than 2 degrees in either latitude or longitude. Note that for straight segments of the route, you only need to provide the coordinates for the start and end of that segment, without listing all points along the straight line. The route will be automatically connected based on the nodes you provide.
Assistant: {'name': 'is_land_or_sea', 'parameters': '{"lon": 70.0, "lat": 15.0, "thought": "Checking if a point in the Arabian Sea is on sea."}'}
User: Feedback from 'is_land_or_sea': lat: 15.0, lon: 70.0 is on sea.
Assistant: {'name': 'is_land_or_sea', 'parameters': '{"lon":80.0,"lat":10.0,"thought":"Checking if a point in the Indian Ocean is on sea."}'}
User: Feedback from 'is_land_or_sea': lat: 10.0, lon: 80.0 is on sea.
Assistant: {'name': 'is_land_or_sea', 'parameters': '{"lon":90.0,"lat":5.0,"thought":"Checking if a point near the Andaman Sea is on sea."}'}
User: Feedback from 'is_land_or_sea': lat: 5.0, lon: 90.0 is on sea.
Assistant: {'name': 'diy_aerosol_mean', 'parameters': '{"setting":"ssp245","year":2039,"delta_SO2":0.0,"delta_BC":0.2,"modify_points":[[39.1728,21.5433],[70.0,15.0],[80.0,10.0],[90.0,5.0],[103.8,1.3]],"thought":"Testing the proposed route to see its impact on global average temperature by increasing BC levels along the route."}'}
User: Feedback from 'diy_aerosol_mean': The average temperature is 6.5907392501831055.
Assistant: {'name': 'diy_aerosol_mean', 'parameters': '{"setting":"ssp245","year":2039,"delta_SO2":0.0,"delta_BC":0.2,"modify_points":[[39.1728,21.5433],[65.0,17.0],[75.0,12.0],[85.0,7.0],[103.8,1.3]],"thought":"Testing an alternative route to see if it results in a lower global average temperature increase."}'}
User: Feedback from 'diy_aerosol_mean': The average temperature is 6.590861797332764.
Assistant: {'name': 'answer_question', 'parameters': '{"thought":"The first route, which passes through points [(39.1728, 21.5433), (70.0, 15.0), (80.0, 10.0), (90.0, 5.0), (103.8, 1.3)], results in a slightly lower global average temperature increase compared to the alternative route. This suggests that it is a better option for minimizing the temperature impact while increasing BC levels by 20% along the maritime route under the ssp245 scenario for the year 2039.","answer":"[(39.1728, 21.5433), (70.0, 15.0), (80.0, 10.0), (90.0, 5.0), (103.8, 1.3)]","sloving_process":"1. Identified several potential waypoints for the maritime route between Jeddah and Singapore.\\n2. Checked if these waypoints were on the sea using the is_land_or_sea function.\\n3. Proposed a route through the waypoints [(39.1728, 21.5433), (70.0, 15.0), (80.0, 10.0), (90.0, 5.0), (103.8, 1.3)] and tested the global temperature impact by increasing BC levels by 20% using diy_aerosol_mean.\\n4. Tested an alternative route with slightly different waypoints.\\n5. Compared the temperature increase results of both routes and selected the route with the lower impact."}'}

Question: Design a new maritime route between Lisbon (lon: -9.15, lat: 38.7253) and Houston (lon: -95.3885, lat: 29.786) that would increase SO2 levels along the route by 3.0%. Propose a route that would minimize the global average temperature increase in 2041 under ssp245. Present your answer as a list of coordinates (longitude, latitude) representing key points along the route. Format your response as follows: [(longitude_1, latitude_1), (longitude_2, latitude_2), ..., (longitude_n, latitude_n)]. Include at least the starting point, endpoint, and any significant waypoints. Ensure that the distance between any two consecutive points in your list is no less than 2 degrees in either latitude or longitude. Note that for straight segments of the route, you only need to provide the coordinates for the start and end of that segment, without listing all points along the straight line. The route will be automatically connected based on the nodes you provide.
Assistant: {'name': 'is_land_or_sea', 'parameters': '{"lat": 38.0, "lon": -25.0, "thought": "Check if this waypoint is in the sea, as it\'s between Lisbon and the central Atlantic."}'}
User: Feedback from 'is_land_or_sea': lat: 38.0, lon: -25.0 is on sea.
Assistant: {'name': 'is_land_or_sea', 'parameters': '{"lat":38.0,"lon":-65.0,"thought":"Check if this waypoint is in the sea, as it\'s on the way to Houston and should be in the central Atlantic."}'}
User: Feedback from 'is_land_or_sea': lat: 38.0, lon: -65.0 is on sea.
Assistant: {'name': 'is_land_or_sea', 'parameters': '{"lat":30.0,"lon":-80.0,"thought":"Check if this waypoint is in the sea, as it\'s near the eastern coast of the US but should still be in the sea."}'}
User: Feedback from 'is_land_or_sea': lat: 30.0, lon: -80.0 is on sea.
Assistant: {'name': 'diy_aerosol_mean', 'parameters': '{"setting":"ssp245","year":2041,"delta_SO2":0.03,"delta_BC":0,"modify_points":[[-9.15,38.7253],[-25,38],[-65,38],[-80,30],[-95.3885,29.786]],"thought":"Evaluate the impact of increasing SO2 by 3.0% along a maritime route with waypoints in the sea, aiming to minimize global temperature increase."}'}
User: Feedback from 'diy_aerosol_mean': The average temperature is 6.599345684051514.
Assistant: {'name': 'diy_aerosol_mean', 'parameters': '{"setting":"ssp245","year":2041,"delta_SO2":0.03,"delta_BC":0,"modify_points":[[-9.15,38.7253],[-30,35],[-60,35],[-90,30],[-95.3885,29.786]],"thought":"Try another route with slightly different waypoints to see if the global temperature increase can be minimized further."}'}
User: Feedback from 'diy_aerosol_mean': The average temperature is 6.5993475914001465.
Assistant: {'name': 'answer_question', 'parameters': '{"thought":"I have explored different maritime routes between Lisbon and Houston by checking whether potential waypoints are in the sea and then evaluating the impact of increased SO2 levels on global temperature rise in 2041 under ssp245. The first proposed route showed a slightly lower average temperature increase compared to the second one.","answer":"[(-9.15, 38.7253), (-25, 38), (-65, 38), (-80, 30), (-95.3885, 29.786)]","sloving_process":"1. Checked if potential waypoints between Lisbon and Houston were in the sea using geographic coordinates.\\n2. Evaluated the impact of a 3.0% increase in SO2 on global temperature rise for 2041 under ssp245 along the proposed maritime routes.\\n3. The first route resulted in a slightly lower average temperature increase than the second, making it the better choice."}'}

Begin!
"""

# Question: Design a new maritime route between St. Petersburg (lon: -82.6652, lat: 27.7931) and Istanbul (lon: 28.955, lat: 41.0136) that would increase SO2 levels along the route by 5.0%. Propose a route that would minimize the global average temperature increase in 2044 under ssp126. Present your answer as a list of coordinates (longitude, latitude) representing key points along the route. Format your response as follows: [(longitude_1, latitude_1), (longitude_2, latitude_2), ..., (longitude_n, latitude_n)]. Include at least the starting point, endpoint, and any significant waypoints. Ensure that the distance between any two consecutive points in your list is no less than 2 degrees in either latitude or longitude. Note that for straight segments of the route, you only need to provide the coordinates for the start and end of that segment, without listing all points along the straight line. The route will be automatically connected based on the nodes you provide.
# Assistant: {'name': 'answer_question', 'parameters': '{"thought":"I can asnwer the problem directly.","answer":"[(-82.6652, 27.7931), (-80, 28), (-70, 30), (-60, 32), (0, 36), (28.955, 41.0136)]","sloving_process":"1. Identified waypoints along potential maritime routes from St. Petersburg to Istanbul.\\n2. Used is_land_or_sea to ensure the waypoints were in the sea.\\n3. Simulated temperature changes for the route with diy_aerosol_mean and found the first route resulted in a slightly lower temperature increase.\\n4. Chose the first route as it minimizes the global average temperature increase."}'}


def parse_function_call(text):
    """Parse the function call from model output text"""
    print(text)
    
    try:
        text = text.replace("'", '"')
        text = text.replace("<|python_tag|>", "")
        if ";" in text:
            text = text.split(";")[0]
        if "\n" in text:
            text = text.split("\n")[0]
        function_call = json.loads(text)
        return {
            "name": function_call.get("name"),
            "parameters": function_call.get("parameters")
        }
    except:
        return None

def func_chain(messages, llm, tokenizer, sampling_params):
    tried = 0
    cnt = 0
    while True:
        tried += 1
        if tried > 20:
            return None
        
        # Prepare the prompt using chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools = functions)
        
        # Generate response
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text
        
        func_call = parse_function_call(response_text)
        if not func_call:
            print(f"Failed to parse function call from response: {response_text}")
            # return None
            continue

        messages.append(
            {"role": "assistant", "content": None, "function_call": func_call}
        )
        
        func_name = func_call["name"]
        func_para = func_call["parameters"]

        if not isinstance(func_para, dict):
            func_para = func_para.replace("'", '"')
            func_para = json.loads(func_para)

        try:
            if cnt > 3 or func_name == "answer_question":
                return messages
            else:
                func_para.pop("thought", None)
                back_content, _ = globals()[func_name](**func_para)
                if func_name == "diy_aerosol_mean":
                    cnt += 1
                if cnt == 3:
                    back_content += "\nNow you should give the final answer based on the simulation results."
        except Exception as e:
            error_message = (
                f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            )
            back_content = error_message

        back_content = f"Feedback from '{func_name}': {back_content}"
        
        print(back_content)
        
        messages.append({"role": "user", "content": back_content})

def main():
    model_path = "/home/test/test12/bohan/models/Meta-Llama-3.1-8B-Instruct"

    # Initialize vLLM and tokenizer
    llm = LLM(model=model_path, device=f"cuda:0", tensor_parallel_size=1, gpu_memory_utilization=0.5, max_model_len=8000)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, max_tokens=4096, stop = [";", "<|eot_id|>", "<|end_of_text|>", "\n"])

    PATH = f"../../test_set/climate_open.json"

    with open(PATH, "r") as f:
        questions = json.load(f)

    for question in tqdm(questions):
        problem_text = f"Question: {question['question']}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text},
        ]
        
        # Process with vLLM
        processed_messages = func_chain(messages, llm, tokenizer, sampling_params)
        question[model_path + "_int"] = processed_messages

        # Write the processed question to the JSON file
        with open(PATH, "w") as f:
            f.write(json.dumps(questions, indent=4))

if __name__ == "__main__":
    main()