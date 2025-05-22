from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import subprocess
import hashlib
import math
import random
import json
import sys
import os
import signal
from vllm import LLM, SamplingParams
from tqdm import tqdm
import jsonlines
import torch
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROGRESS_FILE = "progress.jsonl"

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException("Code execution timed out")

def hash_string(input_string):
    sha256 = hashlib.sha256()
    sha256.update(input_string.encode("utf-8"))
    return sha256.hexdigest()

def write_and_run_code(code, timeout=10):  # timeout in seconds
    code_path = f"workspace/{hash_string(code)[:8]}.py"
    with open(code_path, "w") as f:
        f.write(code)
    
    try:
        # Set up the alarm signal
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(timeout)
        
        result = subprocess.run(
            ["python", code_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout  # subprocess timeout
        )
        
        # Cancel the alarm
        signal.alarm(0)
        
        output_message = (
            f"Output:\n{result.stdout.decode('utf-8')}" if result.stdout else ""
        )
        error_message = (
            f"\n\nError:\n{result.stderr.decode('utf-8')}" if result.stderr else ""
        )
        return_message = f"{output_message}{error_message}"
        return return_message if return_message else "No output"
        
    except TimeoutException:
        return "Error: Code execution timed out"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out"
    except Exception as e:
        return f"Error: {str(e)}"
    finally:
        # Cleanup: Cancel the alarm and remove the temporary file
        signal.alarm(0)
        try:
            os.remove(code_path)
        except:
            pass

def equiv(model_output, answer):
    model_output = float(model_output.replace("$", "").strip())
    ans = float(str(answer).strip())
    print(model_output, ans)
    return math.isclose(model_output, ans, rel_tol=0.05)

def func_chain(messages, llm, sampling_params, tokenizer):
    while True:
        # Convert messages to prompt format
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        outputs = llm.generate([prompt], sampling_params)
        func_call = outputs[0].outputs[0].text

        if "Feedback" in func_call:
            func_call = func_call.split("Feedback")[0].strip()
        messages.append({"role": "assistant", "content": func_call})
        
        try:
            if "write_and_run_code" not in func_call.lower():
                return messages
            else:
                code = func_call.split("```python")[1].split("```")[0].strip()
                back_content = write_and_run_code(code, timeout=10)  # 10 seconds timeout
            print(back_content)
        except Exception as e:
            print(e)
            back_content = f"Error: {e}"
        messages.append({"role": "user", "content": f"Feedback: {back_content}"})
        if len(messages) > 20:
            return None

system_prompt = """
Write a python code to answer the following question. Your code MUST include detailed print statements to show the step-by-step process and intermediate results. Print statements should clearly explain what is being calculated at each step. Don't include visualization in the code. The final answer shoud be presented in latex like '\\boxed{{\\frac{{1}}{{2}}}}'.

Each time, you should whether write a code or answer the question.

If you want to write a code, you answer should be in the following format:

Thought: <Your thought>
Action: write_and_run_code
Code:
```python
<Your code>
```

If you want to answer the question, you should answer in the following format:

Thought: <Your thought>
Action: answer_question
Solution: <Your solution>
Answer: <Your answer, a number>

There are some examples:
{examples}
Now begin!
"""

with open("math_demo.json", "r") as f:
    examples = json.load(f)

demos = {}

for t in [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]:
    demos[t] = ""
    for index, e in enumerate(examples[t]):
        e["agent"] = e["agent"][:2] + e["agent"][-3:]
        demos[t] += f"Problem {index}: {e['question']}\n"
        for i in e["agent"]:
            if i["role"] == "assistant":
                if i["function_call"]["name"] == "write_and_run_code":
                    func_para = json.loads(i["function_call"]["arguments"])
                    demos[
                        t
                    ] += f"Thought: {func_para['thought']}\nAction: write_and_run_code\nCode:\n```python\n{func_para['code']}\n```\n"
                elif i["function_call"]["name"] == "answer_question":
                    func_para = json.loads(i["function_call"]["arguments"])
                    demos[
                        t
                    ] += f"Thought: {func_para['thought']}\nAction: answer_question\n{e['agent_solution']}\n"
            elif i["role"] == "function":
                demos[t] += f"Feedback from '{i['name']}': {i['content']}"
        demos[t] += "\n"

def save_progress(processed_question):
    with jsonlines.open(PROGRESS_FILE, mode="a") as writer:
        writer.write(processed_question)

def process_chunk(questions_subset, gpu_id, chunk_id, model_id):
    print(f"GPU {gpu_id} - Chunk {chunk_id} - Processing {len(questions_subset)} questions")
    # Initialize vLLM
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    llm = LLM(
        model=model_id
    )
    
    sampling_params = SamplingParams(
        temperature=0.7,
        max_tokens=4096, 
        stop = ["Feedback", "<|eot_id|>", "<|end_of_text|>", "User:"]
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    for question in tqdm(questions_subset, desc=f"GPU {gpu_id} - Chunk {chunk_id}"):
        print(f"GPU {gpu_id} - Processing: {question['problem']}")
        messages = [
            {
                "role": "system",
                "content": system_prompt.format(examples=demos[question["type"]]),
            },
            {"role": "user", "content": f"Problem: {question['problem']}"},
        ]
        question[model_id + "_int"] = func_chain(messages, llm, sampling_params, tokenizer)
        save_progress(question)

    return "Chunk processing complete"

def split_data(data):
    done = []
    new_data = []
    with jsonlines.open(PROGRESS_FILE, mode="r") as reader:
        for obj in reader:
            done.append(obj["unique_id"])
    for d in data:
        if d["unique_id"] not in done:
            new_data.append(d)
    return new_data, done

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    
    MODEL = ''
    
    raw_data = []

    with open("test_math.json", "r") as f:
        raw_data = json.load(f)
    
    raw_data, done = split_data(raw_data)

    random.shuffle(raw_data)
    
    for d in raw_data:
        d["type"] = d["unique_id"].split("/")[1]

    num_gpus = torch.cuda.device_count()
    print(f"Available GPUs: {num_gpus}")
    
    num_chunks = min(8, num_gpus)
    
    chunk_size = len(raw_data) // num_chunks + 1
    chunks = [
        raw_data[i * chunk_size : (i + 1) * chunk_size] for i in range(num_chunks - 1)
    ]
    chunks.append(raw_data[(num_chunks - 1) * chunk_size :])

    with ProcessPoolExecutor(max_workers=num_chunks) as executor:
        futures = [
            executor.submit(process_chunk, chunk, i, i, MODEL)
            for i, chunk in enumerate(chunks)
        ]

        for future in as_completed(futures):
            print(future.result())

    final_processed_data = []
    with jsonlines.open(PROGRESS_FILE, mode="r") as reader:
        for obj in reader:
            final_processed_data.append(obj)

    with open("test_math.json", "w") as f:
        json.dump(final_processed_data, f, indent=4)
