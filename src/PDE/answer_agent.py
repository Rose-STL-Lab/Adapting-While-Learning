import json
import os
import random
import sys
import traceback
from utils.make_problem_utils import *
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm
sys.path.append('..')
from tools.diffusion import DiffusionSolver
from functions.answer import function_answer
from functions.pdes import function_pdes

# Ensure TOKENIZERS_PARALLELISM is set to false to avoid tokenizer parallelism issues
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def setup_functions():
    useful_tools = list(function_pdes.keys())
    functions = [function_pdes[name] for name in useful_tools]
    _functions = []
    for value in functions:
        if "parameters" in value and "properties" in value["parameters"]:
            value["parameters"]["properties"]["thought"] = {
                "type": "string",
                "description": "Your internal reasoning and thoughts of why you call this function.",
            }
        _functions.append({"type": "function", "function": value})
    return _functions + function_answer

functions = setup_functions()

def call_method(obj, method_name, **kwargs):
    method = getattr(obj, method_name)
    return method(**kwargs)

system_prompt = """
You are a physics scientist. You are going to answer a multi-choice question. You should use given tools to help you answer the question. You should first set the dimension, DLT, initial condition and boundry condition with necessary parameters, then solve and get the number you want. You can call tools for many turns, but you should call only one tool each time. You should also give the solution_type based on what kind of answer the problem requires.

Below are examples for how to set ic_params and bc_params:
                    
**1D Initial Condition - Constant:**
{
    "value": 100
}
This sets a constant initial value of 100 across the domain.

**1D Initial Condition - Step:**
{
    "left_value": 100,
    "right_value": 0,
}
This creates a step function with 100 on the left half and 0 on the right half of the domain.

**2D Initial Condition - Constant:**
{
    "value": 100
}
This sets a constant initial value of 100 across the 2D domain.

**2D Initial Condition - Checkerboard:**
{
    "value1": 100,
    "value2": 0,
    "cell_size_x": 0.5,
    "cell_size_y": 0.5
}
This sets up a checkerboard pattern with alternating values of 100 and 0, where each cell in the checkerboard has dimensions `0.5 x 0.5`.

**1D Boundary Condition - Dirichlet:**
{
    "left_value": 100,
    "right_value": 50
}
This sets Dirichlet boundary conditions with a value of 100 at the left boundary (`x=0`) and 50 at the right boundary (`x=L`).

**1D Boundary Condition - Neumann:**
{}
This sets Neumann boundary conditions with zero flux at both boundaries (i.e., insulated or reflecting boundaries).

**1D Boundary Condition - Periodic:**
{}
This sets periodic boundary conditions. No additional parameters are needed.

**2D Boundary Condition - Dirichlet:**
{
    "left_value": 100,
    "right_value": 0,
    "bottom_value": 50,
    "top_value": 75
}
This sets Dirichlet boundary conditions with values of 100, 0, 50, and 75 at the left, right, bottom, and top boundaries of the 2D domain, respectively.

**2D Boundary Condition - Neumann:**
{}
This sets Neumann boundary conditions with zero flux at all boundaries of the 2D domain.

**2D Boundary Condition - Periodic:**
{}
This sets periodic boundary conditions for the 2D domain. No additional parameters are needed.

Examples:

Question: In a 1D heat conduction process with constant initial temperature of 15 °C and Dirichlet boundary conditions with 33 °C at x=0 and 87 °C at x=L, what is the temperature (°C) at x=36 cm after 241 seconds? (D=0.18, L=93 cm)
Options:
A. 10.5
B. 12.0
C. 13.5
D. 15.0
Assistant: {'name': 'setup_and_solve', 'parameters': '{"dimension":"1D","D":0.18,"T":241,"L":93,"ic_type":"constant","ic_params":{"value":15},"bc_type":"dirichlet","bc_params":{"left_value":33,"right_value":87},"solution_type":"get_value_at_point","solution_x":36}'}
Feedback from 'setup_and_solve': Simulation Result:
[33.         20.1129984  15.8995674  15.11285897 15.0142828  15.04435316
 15.45059527 18.598222   35.45199124 87.        ]
The result for solution: 15.065160821102706
Assistant: {'name': 'answer_question', 'parameters': '{"thought":"After solving the 1D heat conduction problem, the temperature at x=36 cm after 241 seconds is approximately 15.065 °C. This value is closest to option D, 15.0 °C.","answer":"D","sloving_process":"1. Identified the problem as a 1D heat conduction problem.\\n2. Set the parameters: diffusion coefficient D=0.18, total time T=241 seconds, length L=93 cm.\\n3. Used constant initial temperature of 15 °C and Dirichlet boundary conditions with 33 °C at x=0 and 87 °C at x=L.\\n4. Solved the PDE to find the temperature at x=36 cm.\\n5. Found the temperature to be approximately 15.065 °C, which matches option D (15.0 °C)."}'}

Question: In a 1D chemical diffusion process with step function with 8 mol/L for x < L/2 and 74 mol/L for x ≥ L/2 and Dirichlet boundary conditions with 53 mol/L at x=0 and 93 mol/L at x=L, what is the minimum concentration (mol/L) after 686 seconds? (D=0.0006, L=3 cm)
Options:
A. 27.622
B. 31.568
C. 35.514
D. 39.46
Assistant: {'name': 'setup_and_solve', 'parameters': '{"dimension":"1D","D":0.0006,"T":686,"L":3.0,"ic_type":"step","ic_params":{"left_value":8.0,"right_value":74.0},"bc_type":"dirichlet","bc_params":{"left_value":53.0,"right_value":93.0},"solution_type":"get_min_value"}'}
Feedback from 'setup_and_solve': Simulation Result:
[53.         45.02834872 39.92147009 39.46174977 43.82646755 51.84663884
 61.82227765 72.36314131 82.78392977 93.        ]
The result for solution: 39.46174977174152
Assistant: {'name': 'answer_question', 'parameters': '{"thought":"I have solved the diffusion equation using the given parameters. The minimum concentration after 686 seconds is approximately 39.46 mol/L. This matches option D.","answer":"D","sloving_process":"1. Set the problem to a 1D diffusion equation with given parameters: diffusion coefficient D = 0.0006, length L = 3 cm, and time T = 686 seconds.\\n2. Set the initial condition as a step function with 8 mol/L for x < L/2 and 74 mol/L for x ≥ L/2.\\n3. Set Dirichlet boundary conditions with 53 mol/L at x=0 and 93 mol/L at x=L.\\n4. Solve the diffusion equation to find the minimum concentration over the domain after 686 seconds.\\n5. The simulation result shows the minimum concentration is approximately 39.46 mol/L, which corresponds to option D."}'}

Question: In a 2D population spread process with checkerboard pattern with alternating 100 individuals/km² and 0 individuals/km² and Dirichlet boundary conditions with 40, 55, 10, and 97 individuals/km² at left, right, bottom, and top boundaries, what is the maximum gradient of population density (individuals/km²) per km after 7 years? (D=0.44, Lx=72 km, Ly=5 km)
Options:
A. 91.2
B. 102.6
C. 114.0
D. 125.4
Assistant: {'name': 'setup_and_solve', 'parameters': '{"dimension":"2D","D":0.44,"T":7.0,"Lx":72.0,"Ly":5.0,"ic_type":"checkerboard","ic_params":{"value1":100,"value2":0,"cell_size_x":36.0,"cell_size_y":2.5},"bc_type":"dirichlet","bc_params":{"left_value":40,"right_value":55,"bottom_value":10,"top_value":97},"solution_type":"get_max_gradient"}'}
Feedback from 'setup_and_solve': Simulation Result:
[[40.         10.         10.         10.         10.         10.
  10.         10.         10.         55.        ]
 [40.         19.04698679 18.9870911  18.98847405 19.02232274 19.72358612
  19.75743771 19.76023771 19.98672407 55.        ]
 [40.         28.13951612 28.13105633 28.13409633 28.18620385 29.26552629
  29.3176303  29.32204808 29.61833575 55.        ]
 [40.         37.41548122 37.53494247 37.53977842 37.58622972 38.54784503
  38.59427412 38.59876928 38.86069926 55.        ]
 [40.         46.93165015 47.21658759 47.22303366 47.24272279 47.64916657
  47.66880682 47.67208748 47.8381163  55.        ]
 [40.         56.66898941 57.1129399  57.12042507 57.10485539 56.78015416
  56.76451043 56.76591475 56.80903241 55.        ]
 [40.         66.56854291 67.11642303 67.12399217 67.0811654  66.19145723
  66.14854387 66.14818587 66.0731101  55.        ]
 [40.         76.58058426 77.12539848 77.13176979 77.08235479 76.05643102
  76.00693791 76.00561364 75.85257841 55.        ]
 [40.         86.70424714 87.08500648 87.08875598 87.05634108 86.38350327
  86.35104158 86.34990154 86.20358211 55.        ]
 [40.         97.         97.         97.         97.         97.
  97.         97.         97.         55.        ]]
The result for solution: 114.0
Assistant: {'name': 'answer_question', 'parameters': '{"thought":"I\'ve determined the maximum gradient of population density after solving the 2D diffusion equation with the given parameters. The maximum gradient is 114.0 individuals/km² per km. This matches option C.","answer":"C","sloving_process":"1. I set up a 2D diffusion equation with a checkerboard initial condition, alternating between 100 and 0 individuals/km².\\n2. The domain dimensions Lx and Ly were set to 72 km and 5 km, respectively, and the diffusion coefficient D was set to 0.44.\\n3. Dirichlet boundary conditions were applied with values of 40, 55, 10, and 97 individuals/km².\\n4. I solved the equation for a time period of 7 years.\\n5. After solving, I extracted the maximum gradient of the population density, which was found to be 114.0 individuals/km² per km.\\n6. I compared this result with the given options and found that it matches option C."}'}

Question: In a 2D chemical diffusion process with checkerboard pattern with alternating 100 mol/L and 0 mol/L and Dirichlet boundary conditions with 4, 68, 89, and 9 mol/L at left, right, bottom, and top boundaries, what is the maximum gradient of concentration (mol/L) per cm after 676 seconds? (D=0.0004, Lx=1 cm, Ly=8 cm)
Options:
A. [28.45245304897628, 401.9839963684302)
B. [401.9839963684302, 607.1165071186451)
C. [607.1165071186451, 1042.2697596200178)
D. [1042.2697596200178, 1493.7920949383997)
Assistant: {'name': 'setup_and_solve', 'parameters': '{"dimension":"2D","D":0.0004,"T":676,"Lx":1.0,"Ly":8.0,"ic_type":"checkerboard","ic_params":{"value1":100,"value2":0,"cell_size_x":0.5,"cell_size_y":0.5},"bc_type":"dirichlet","bc_params":{"left_value":4,"right_value":68,"bottom_value":89,"top_value":9},"solution_type":"get_max_gradient"}'}
Feedback from 'setup_and_solve': Simulation Result:
[[ 4.         89.         89.         89.         89.         89.
  89.         89.         89.         68.        ]
 [ 4.         14.55445798 23.95956822 32.36340214 39.89225251 46.65749582
  52.7603388  58.29457019 63.34758142 68.        ]
 [ 4.         11.75490236 19.42119093 26.93236113 34.24557219 41.34251867
  48.22906078 54.93340579 61.50293721 68.        ]
 [ 4.         11.56216111 19.06912124 26.47276947 33.73802122 40.84691882
  47.80048962 54.61827699 61.33574966 68.        ]
 [ 4.         11.54618808 19.03947622 26.43354369 33.69434073 40.80415799
  47.76360079 54.59130603 61.32153394 68.        ]
 [ 4.         11.53849104 19.02554871 26.41588938 33.67581235 40.78727483
  47.75011235 54.58211576 61.31692891 68.        ]
 [ 4.         11.52748018 19.00491856 26.38821986 33.64450715 40.75611366
  47.72279922 54.56187431 61.30615982 68.        ]
 [ 4.         11.4262124  18.81143674 26.12096046 33.3307446  40.43048063
  47.42472351 54.3316838  61.1794899  68.        ]
 [ 4.         10.58273996 17.12839567 23.68323716 30.30586196 37.07328704
  44.08569828 51.46967982 59.37996381 68.        ]
 [ 4.          9.          9.          9.          9.          9.
   9.          9.          9.         68.        ]]
The result for solution: 744.46
Assistant: {'name': 'answer_question', 'parameters': '{"thought":"I have calculated the maximum gradient of concentration after 676 seconds using the given parameters and initial conditions. The result is 744.46 mol/L per cm, which falls into the range [607.1165071186451, 1042.2697596200178). Hence, the correct answer is option C.","answer":"C","sloving_process":"1. Setup the problem as a 2D diffusion equation with the given parameters: diffusion constant D=0.0004, time T=676 seconds, and domain lengths Lx=1 cm and Ly=8 cm.\\n\\n2. Define the initial condition as a checkerboard pattern with alternating concentrations of 100 mol/L and 0 mol/L. The cell sizes for the checkerboard pattern are set to 0.5 cm x 0.5 cm.\\n\\n3. Set the boundary conditions as Dirichlet with values 4 mol/L at the left boundary, 68 mol/L at the right boundary, 89 mol/L at the bottom boundary, and 9 mol/L at the top boundary.\\n\\n4. Solve the diffusion equation to find the maximum gradient of concentration after 676 seconds.\\n\\n5. The calculated maximum gradient is 744.46 mol/L per cm.\\n\\n6. Compare the result with the provided options and select the range that includes the calculated value. The correct answer is option C, as 744.46 falls within the range [607.1165071186451, 1042.2697596200178)."}'}

Question: In a 1D chemical diffusion process with step function with 76 mol/L for x < L/2 and 13 mol/L for x ≥ L/2 and Dirichlet boundary conditions with 74 mol/L at x=0 and 92 mol/L at x=L, what is the maximum gradient of concentration (mol/L) per cm after 745 seconds? (D=0.0006, L=9 cm)
Options:
A. 48.896
B. 55.007999999999996
C. 61.12
D. 67.232
Assistant: {'name': 'setup_and_solve', 'parameters': '{"dimension":"1D","D":0.0006,"T":745,"L":9,"ic_type":"step","ic_params":{"left_value":76,"right_value":13},"bc_type":"dirichlet","bc_params":{"left_value":74,"right_value":92},"solution_type":"get_max_gradient"}'}
Feedback from 'setup_and_solve': Simulation Result:
[74.         75.34568137 75.43401404 72.78700428 60.09755649 28.97226214
 16.82174812 18.01795559 36.98815764 92.        ]
The result for solution: 61.12
Assistant: {'name': 'answer_question', 'parameters': '{"thought":"The problem involves calculating the maximum gradient of concentration in a 1D diffusion process with given initial and boundary conditions. By setting up the diffusion equation with the specified parameters and solving it, we obtained the maximum gradient after 745 seconds.","answer":"C","sloving_process":"1. Identify the parameters: The diffusion coefficient D is 0.0006, the total length L is 9 cm, and the time T is 745 seconds.\\n2. Set up initial conditions: A step function with 76 mol/L for x < L/2 and 13 mol/L for x ≥ L/2.\\n3. Set boundary conditions: Dirichlet boundary conditions with 74 mol/L at x=0 and 92 mol/L at x=L.\\n4. Solve the diffusion equation and calculate the maximum gradient of concentration per cm.\\n5. The maximum gradient obtained from the simulation is 61.12 mol/L per cm, which corresponds to option C."}'}

Begin!
"""

def parse_function_call(text):
    """Parse the function call from model output text"""
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

def func_chain(messages, pde_solver, llm, tokenizer, sampling_params):
    tried = 0
    while True:
        tried += 1
        if tried > 50:
            return messages
            
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, tools=functions)
        outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
        response_text = outputs[0].outputs[0].text
        
        func_call = parse_function_call(response_text)
        
        if not func_call:
            print(response_text)
            continue

        messages.append(
            {"role": "assistant", "content": response_text}
        )
        
        func_name = func_call["name"]
        func_para = func_call["parameters"]

        if not isinstance(func_para, dict):
            func_para = func_para.replace("'", '"')
            func_para = json.loads(func_para)

        try:
            if func_name == "answer_question":
                return messages
            else:
                func_para.pop("thought", None)
                back_content = str(call_method(pde_solver, func_name, **func_para))
        except Exception as e:
            error_message = f"An error occurred: {str(e)}\n\nTraceback:\n{traceback.format_exc()}"
            back_content = error_message

        print(back_content)
        back_content = f"Feedback from '{func_name}': {back_content}"
        messages.append(
            {"role": "user", "content": back_content}
        )
        if len(messages) > 10:
            return None

def process_chunk(questions_subset, gpu_id, chunk_id, output_file, model_path):
    # Initialize vLLM and tokenizer for this process
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id + 7)
    llm = LLM(model=model_path)
    tokenizer = AutoTokenizer.from_pretrained("")
    sampling_params = SamplingParams(temperature=0.7, max_tokens=4096)

    for question in tqdm(questions_subset, desc=f"Chunk {chunk_id}"):
        if f"{model_path}_int" in question and question[f"{model_path}_int"]:
            with open(output_file, 'a') as f:
                json.dump(question, f)
                f.write('\n')
            continue
        pde_solver = DiffusionSolver()
        problem_text = f"Question: {question['question']}\n\nOptions: A. {question['options'][0]}\nB. {question['options'][1]}\nC. {question['options'][2]}\nD. {question['options'][3]}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": problem_text},
        ]
        
        return_ = func_chain(messages, pde_solver, llm, tokenizer, sampling_params)
        
        print("===")
        print(return_)
        print("===")
        
        question[f"{model_path}_int"] = return_
        
        with open(output_file, 'a') as f:
            json.dump(question, f)
            f.write('\n')

def split_data(data, num_chunks):
    random.shuffle(data)
    chunk_size = len(data) // num_chunks + 1
    return [data[i*chunk_size:(i+1)*chunk_size] for i in range(num_chunks)]

def main():
    print(json.dumps(setup_functions(), indent=4))
    
    multiprocessing.set_start_method('spawn')
    model_paths = [
    ]
    
    with open("pde_test.json", "r") as f:
        questions = json.load(f)
    
    for model_path in model_paths:
        num_gpus = 1
        num_chunks = num_gpus
        chunks = split_data(questions, num_chunks)

        output_file = "pde_test.jsonl"
        open(output_file, 'w').close()

        with ProcessPoolExecutor(max_workers=num_chunks) as executor:
            futures = [
                executor.submit(process_chunk, chunk, i % num_gpus, i, output_file, model_path) 
                for i, chunk in enumerate(chunks)
            ]
            
            for future in as_completed(futures):
                future.result()

        processed_data = []
        with open(output_file, 'r') as f:
            for line in f:
                processed_data.append(json.loads(line))
        
        questions = processed_data

        with open("pde_test.json", "w") as f:
            json.dump(processed_data, f, indent=4)

if __name__ == "__main__":
    main()