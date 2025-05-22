import ast
import json
import sys
import os
import re
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "tools/Climate_online"))
from tools.emulators import *

class Cache:
    def __init__(self, cache_file="evaluation_cache.json"):
        self.cache_file = cache_file
        self.cache = self._load_cache()

    def _load_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, 'r') as f:
                return json.load(f)
        return {}

    def save(self):
        with open(self.cache_file, 'w') as f:
            json.dump(self.cache, f, indent=4)

    def get(self, key):
        return self.cache.get(key)

    def set(self, key, value):
        self.cache[key] = value
        self.save()

def extract_info_from_question(question):
    scenario = r"(maritime route|transfer station)"
    gas_pattern = r"(SO2|BC)"
    setting_pattern = r"(ssp126|ssp245|ssp370|ssp585)"
    increase_pattern = r"(\d{1,2}(?:\.\d{1,2})?)%"
    year_pattern = r"\b(20[2-9][0-9])\b"

    scenario_match = re.search(scenario, question)
    gas_match = re.search(gas_pattern, question)
    setting_match = re.search(setting_pattern, question)
    increase_match = re.search(increase_pattern, question)
    year_match = re.search(year_pattern, question)

    scenario = scenario_match.group(0) if scenario_match else None
    gas = gas_match.group(0) if gas_match else None
    setting = setting_match.group(0) if setting_match else None
    increase = float(increase_match.group(1)) if increase_match else None
    year = int(year_match.group(1)) if year_match else None

    return scenario, gas, setting, increase, year

def route_valid(route):
    for i in route[1:-1]:
        if is_land_or_sea(i[0], i[1])[1]:
            return False
    return True

def evaluate_response(response, question, cache):
    cache_key = f"{question}_{response}"
    
    cached_result = cache.get(cache_key)
    if cached_result:
        return cached_result["temperature"], cached_result["valid"]
    
    try:
        scenario, gas, setting, increase, year = extract_info_from_question(question)
        route = ast.literal_eval(response)
        
        temp = float(diff_diy_aerosol_mean(
            setting,
            year,
            increase if gas == "SO2" else 0,
            increase if gas == "BC" else 0,
            response
        )[1])
        
        valid = route_valid(route)
        
        cache.set(cache_key, {
            "temperature": temp,
            "valid": valid
        })
        
        return temp, valid
    except Exception as e:
        print(f"Error in evaluation: {e}")
        return None, None

def main():
    with open("", "r") as f:
        data = json.load(f)

    model_id = ""
    cache = Cache()
    
    results = []
    for item in tqdm(data):
        item_results = []

        responses = item[model_id] if isinstance(item[model_id], list) else [item[model_id]]
        
        for response in responses:
            try:
                answer = response.split("Answer: ")[1].strip() if "Answer: " in response else response.strip()
                temp, valid = evaluate_response(answer, item["question"], cache)
                item_results.append({
                    "temperature": temp,
                    "valid": valid,
                    "answer": answer
                })
            except Exception as e:
                print(f"Error processing response: {e}")
                item_results.append({
                    "temperature": None,
                    "valid": None,
                    "answer": response
                })
            
        item[model_id + "_eval"] = item_results
        results.append(item)
    
    with open("", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()