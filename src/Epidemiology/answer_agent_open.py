import random
import json
import sys
import os
import traceback
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(src_dir)

# from utils.utils import *
from functions.functions import *

sys.path.append(os.path.join(os.path.dirname(__file__), 'tools'))
from tools.inference_open import *

system_prompt = """
You are the leader of California, tasked with managing the state during a pandemic. Your role is to formulate effective policies to mitigate its impact by allocating resources and making key decisions. You will be provided with specific details of the pandemic, including its severity, affected regions, and other relevant data. Based on this information, you will be asked to take action.

Before submitting your final answer, you are allowed to perform up to three simulations to test your decisions and refine your strategy based on feedback. Use these simulations wisely to optimize your resource allocation.

Your responses should follow the following format. Note that whether you choose to simulate or provide a final answer, you must provide your proposal in JSON format.

Action: [One of {Simulate/Answer}. Choose Simulate to test your decisions and receive feedback from the simulator. Choose Answer to provide your final resource allocation for the crisis.]
Thought: [Clearly explain your reasoning for the decisions you make.]
Answer: Present your proposal in JSON format, specifying the amount allocated to each county. For example:  
{
    "county_name_1": x.xx,
    "county_name_2": x.xx,
    ...
}

Following are 2 examples of how you can structure your response:

Example Response 1:

Action: Simulate  
Thought: Los Angeles is the most densely populated and hardest-hit area, so it will receive the highest allocation. San Francisco also has a high population density and significant cases, so it will receive the second-highest share. Fresno, while less dense, has seen a concerning rise in cases and will need sufficient resources. I will allocate the remaining resources to smaller counties to ensure statewide coverage.  
Answer:  
{
    "Los Angeles": 4.5,
    "San Francisco": 2.5,
    "Fresno": 1.5
    
}


Example Response 2:

Action: Answer  
Thought: Based on the simulator feedback, Los Angeles requires more resources due to its continued surge in cases. San Francisco's curve is flattening, so its allocation is reduced slightly. Fresno is stable but still needs support. Other counties will receive minimal funding to address smaller-scale outbreaks.  
Answer:  
{
    "Los Angeles": 5.0,
    "San Francisco": 2.0,
    "Fresno": 2.0,
    "Santa Barbara": 1.0,
    "Tulare": 0.5
}
"""

JSONL_PATH = "open_questions.jsonl"

def simulate(scenario, output, tgt_feature, category, input_feature, absolute_budget, max_adjustment_per_county):
    try:
        # print("-----------------")
        # print(output)
        # print("-----------------")
        output = output.split("Answer:")[1].replace("`", "").replace("json", "").strip()
        adjs = process_model_response(output)
    except Exception as e:
        print(e)
        return """Your answer should be in the following format.

Action: Simulate/Answer
Thought: Describe your reasoning for allocating the budget.
Answer:
{
    "county_name_1": x.xx,
    "county_name_2": x.xx,
    ...
}"""

    total = 0
    for k, v in adjs.items():
        total += v
    if total > absolute_budget:
        return "The total budget exceeds the absolute budget. Please adjust your allocations accordingly."
    if any(v > max_adjustment_per_county for v in adjs.values()):
        return "The adjustment for at least one county exceeds the maximum allowed per county. Please adjust your allocations accordingly."

    result = test_single_scenario(scenario, output, input_feature, tgt_feature)
    
    if category == "peak":
        return f"The peak value of '{tgt_feature}' is {result['feature_comparison']['peak_comparison']['modified_peak']}."
    else:
        return f"The final value of '{tgt_feature}' is {result['feature_comparison']['tail_comparison']['modified_tail']}."
    

def func_chain(messages, llm, tokenizer, sampling_params, question):
    tried = 0
    cnt = 0
    while True:
        tried += 1
        if tried > 20:
            return None
        
        # Tokenize and prepare the prompt using the chat template
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        func_call = outputs[0].outputs[0].text
        
        print(func_call)

        messages.append(
            {"role": "assistant", "content": func_call}
        )
    
        try:
            if cnt >3 or "Action: Answer" in func_call:
                back_content = simulate(question["scenario"], func_call, question["target_feature"], question["category"], question["input_feature"], question["absolute_budget"], question["max_adjustment_per_county"])
                messages.append({"role": "user", "content": back_content})
                return messages
            else:
                back_content = simulate(question["scenario"], func_call, question["target_feature"], question["category"], question["input_feature"], question["absolute_budget"], question["max_adjustment_per_county"])
                cnt += 1
                if cnt == 3:
                    back_content += "\nNow you should give the final answer based on the simulation results."
        except Exception as e:
            error_message = (
                f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            )
            back_content = error_message

        print(back_content)
        back_content = f"Feedback from the simulator: {back_content}"
        messages.append({"role": "user", "content": back_content})

def main():
    i = 5
    model_path = f"/home/test/test12/bohan/models/Meta-Llama-3.1-8B-Instruct"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(i)
    
    llm = LLM(model=model_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(temperature=0.7, max_tokens = 4096)

    with open("../../test_set/epidemiology_open.json", "r") as f:
        questions = json.load(f)

    for question in tqdm(questions):
        problem_text = f"Question: {question['question']}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text},
        ]

        returned_messages = func_chain(messages, llm, tokenizer, sampling_params, question)
        
        print("===")
        print(returned_messages)
        print("===")
        
        question[model_path + "_int"] = returned_messages
        # Write the processed question to the JSONL file
        with open("test.json", "w") as f:
            f.write(json.dumps(questions, indent=4))
        
        print(question)
        
        with open(JSONL_PATH, "a") as f:
            f.write(json.dumps(question) + "\n")


if __name__ == "__main__":
    main()