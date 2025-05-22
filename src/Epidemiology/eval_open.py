import ast
import json
import sys
import os
import re
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "Epidemic"))
from tools.inference_open import *

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

def simulate(scenario, output, tgt_feature, category, input_feature, absolute_budget, max_adjustment_per_county):
    try:
        # Safely extract the answer part
        parts = output.split("Answer:")
        if len(parts) < 2:
            print("parts < 2")
            return False, False
            
        output = parts[1].replace("`", "").replace("json", "").strip()
        adjs = process_model_response(output)
        
        total = sum(adjs.values())
        if total > absolute_budget:
            print("abs bgt")
            return False, False
            
        if any(v > max_adjustment_per_county for v in adjs.values()):
            print("sgl bgt")
            return False, False

        result = test_single_scenario(scenario, output, input_feature, tgt_feature)
        
        if category == "peak":
            return result['feature_comparison']['peak_comparison']['modified_peak'], True
        else:
            return result['feature_comparison']['tail_comparison']['modified_tail'], True
            
    except Exception as e:
        print(f"Simulation error: {str(e)}")
        return False, False

def evaluate_response(response, question, cache):
    
    try:
        # Safely extract parameters from question dictionary
        scenario = question.get("scenario")
        tgt_feature = question.get("target_feature")
        category = question.get("category")
        input_feature = question.get("input_feature")
        absolute_budget = question.get("absolute_budget")
        max_adjustment_per_county = question.get("max_adjustment_per_county")
            
        value, valid = simulate(scenario, response, tgt_feature, category, input_feature, 
                              absolute_budget, max_adjustment_per_county)
        
        return value, valid
        
    except Exception as e:
        print(f"Error in evaluation: {str(e)}")
        return None, None

def main():
    with open("open_test.json", "r") as f:
        data = json.load(f)

    model_id = ""
    
    results = []
    for item in tqdm(data):
        item_results = []
        
        # Handle both single response and list of responses
        responses = item[model_id] if isinstance(item[model_id], list) else [item[model_id]]
        
        for response in responses:
            try:
                answer = response
                
                temp, valid = evaluate_response(answer, item, None)
                item_results.append({
                    "temperature": temp,
                    "valid": valid,
                    "answer": answer
                })
            except Exception as e:
                print(f"Error processing response: {str(e)}")
                item_results.append({
                    "temperature": None,
                    "valid": None,
                    "answer": response
                })
                
        item[model_id + "_eval"] = item_results
        results.append(item)
    
    with open("open_test.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()