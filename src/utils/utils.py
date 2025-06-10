import json
import logging
import os
import sys
from datetime import datetime

from openai import OpenAI
from colorama import Fore, init

messages_length = []

init(autoreset=True)

API_KEY = ""

client = OpenAI(
    api_key=API_KEY,
    base_url="https://api.openai.com/v1",
)


class LoggerAndPrinter:
    COLOR_MAPPING = {
        "red": Fore.RED,
        "green": Fore.GREEN,
        "yellow": Fore.YELLOW,
        "blue": Fore.BLUE,
        "magenta": Fore.MAGENTA,
        "cyan": Fore.CYAN,
        "white": Fore.WHITE,
        "black": Fore.BLACK,
    }

    def __init__(self, query=None, identifier=""):
        if query and len(query) > 50:
            query = query[:25] + "..." + query[-25:]
        self.terminal = sys.stdout
        self.log = logging.getLogger(__name__)
        self.log.setLevel(logging.INFO)
        log_dir = "last_exp"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = (
            os.path.join(log_dir, f"{timestamp}_{identifier}_{query}.log")
            if query
            else os.path.join(log_dir, f"my_log_{timestamp}_{identifier}.log")
        )

        fh = logging.FileHandler(log_filename)
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        fh.setFormatter(formatter)
        self.log.addHandler(fh)

    def write(self, message, color=None):
        if color and color in self.COLOR_MAPPING:
            message = self.COLOR_MAPPING[color] + message
        self.terminal.write(message)
        self.log.info(message.strip("\n"))

    def flush(self):
        self.terminal.flush()

    def print_colored(self, message, color):
        self.write(message + "\n", color)
        self.flush()


sys.stdout = LoggerAndPrinter()

chat_api = 0


def print_usage():
    print(messages_length)


class GPT:
    def __init__(self, version="gpt-4o-mini", temperature=0.7):
        self.version = version
        self.temperature = temperature
        self.accum_len = 0

    def gpt4(self, messages):
        response = client.chat.completions.create(
            model=self.version,
            messages=messages,
            # max_tokens=2000,
            temperature=self.temperature,
            stop=None,
        )

        messages_length.append(dict(response.usage))
        sys.stdout.print_colored(
            str(response.choices[0].message.content.strip()), "green"
        )

        return response.choices[0].message.content.strip()

    def gpt4_functions(self, messages, functions):
        print(" | ".join([i["function"]["name"] for i in functions]))

        MAX_TRIES = 5
        tried = 0
        while True:
            try:
                tried += 1
                if tried > MAX_TRIES:
                    return None

                response = None

                response = client.chat.completions.create(
                    model=self.version,
                    messages=messages,
                    tools=functions,
                    temperature=self.temperature,
                    # max_tokens=2000,
                    stop=None,
                )

                messages_length.append(dict(response.usage))
                func_call = response.choices[0]
                func_call = func_call.message.tool_calls
                func_call = func_call[0].function
                func_name = func_call.name
                func_para = func_call.arguments
                func_call = {"name": func_name, "arguments": func_para}
                json.loads(func_para)

                sys.stdout.print_colored(str(func_call), "yellow")

                self.accum_len = response.usage.total_tokens
                return func_call
            except Exception as e:
                import traceback

                traceback.print_exc()
                print(traceback.format_exc())
                print(e)

                print(messages)
                print(response)

                if response:
                    response = dict(response)
                    for k in response.keys():
                        v = response[k]
                        if type(v) != str:
                            response[k] = str(v)
                messages[-1]["content"] += "\nYou should call a function."


gpt = GPT()


def gpt4(messages):
    return gpt.gpt4(messages)


def gpt4_functions(messages, functions):
    return gpt.gpt4_functions(messages, functions)


class llama:
    def __init__(
        self,
        model_path="",
        device="cuda:7",
    ):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
        )

    def generate(self, messages, tools=None, max_new_tokens=3072):
        cal = 0
        if tools:
            print(" | ".join([i["function"]["name"] for i in tools]))
            inputs = self.tokenizer.apply_chat_template(
                messages,
                tools=tools,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            inputs = self.tokenizer.apply_chat_template(
                messages,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

        inputs = {key: val.to(self.model.device) for key, val in inputs.items()}

        while cal < 5:
            cal += 1
            try:
                out = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
                out = (
                    self.tokenizer.decode(out[0][len(inputs["input_ids"][0]) :])
                    .replace("<|python_tag|>", "")
                    .replace("<|eom_id|>", "")
                    .replace("<|eot_id|>", "")
                )
                out = out.replace(
                    "None<|end_header_id|><|start_header_id|>assistant<|end_header_id|>",
                    "",
                ).strip()
                out = out.replace(
                    "None<|end_header_id|><|start_header_id|>function<|end_header_id|>",
                    "",
                ).strip()
                sys.stdout.print_colored(out, "yellow")
                if tools:
                    if "\n" in out:
                        out = out.split("\n")[0]
                    if ";" not in out:
                        return json.loads(
                            out.replace("<|python_tag|>", "")
                            .replace("<|eom_id|>", "")
                            .replace("'", '"')
                        )
                    else:
                        out = (
                            out.replace("<|python_tag|>", "")
                            .replace("<|eom_id|>", "")
                            .replace("'", '"')
                        )
                        outs = out.split(";")
                        outs = [json.loads(i.strip()) for i in outs]
                        return outs[0]
                else:
                    return out.replace("<|eom_id|>", "")
            except Exception as e:
                print(e)
                print("Retrying...")
                continue
            cal += 1

        return None


if __name__ == "__main__":
    llama3 = llama()
    print(
        llama3.generate([{"role": "user", "content": "What is the capital of France?"}])
    )
    print(
        llama3.generate(
            [{"role": "user", "content": "How old is adele"}],
            [
                {
                    "type": "function",
                    "function": {
                        "name": "google_search",
                        "description": "Search the web for the answer to the user's question.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "The user's question.",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ],
        )
    )
