from dotenv import load_dotenv
load_dotenv()
import os
import json
from openai import OpenAI
from loguru import logger
import numpy as np
import random
import torch

''' save augmentation logs '''
os.makedirs('logs', exist_ok=True)
import datetime
now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
logger.remove(0)
logger.add(f"logs/{now}.log", format="<level>{level}</level> | <level>{message}</level>", rotation="10 MB")

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def query(user_input: str, 
    system_prompt: str = '', 
    model="gpt-4o-mini", 
    temperature: float = 1.5, 
    max_tokens: int = 200, 
    seed: int = 42
) -> str:
    '''
    Usage:
        Input the user input and system prompt, and get the response from the given model.

    Parameters:
        :user_input: the user input
        :system_prompt: the system prompt, default is empty string
        :model: the model to use, default is gpt-4o-mini
        :temperature: the temperature of the model, the higher the temperature, the more diverse the output, default is 1.5
        :max_tokens: the maximum number of tokens to generate, default is 200
        :seed: the random seed, default is 42

    Returns:
        The response generated by the model.
    '''
    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        max_tokens=max_tokens,  
        temperature=temperature,    # 温度在0-2之间，值越大，越有创造力
        seed=seed,
    )

    return completion.choices[0].message.content

class Log:
    def __init__(self, output_path: str):
        '''
        Usage:
            Create a log file for augmentation.
            It is used to keep track of the last index of the augmented data.
            So that we don not need to augment the data from scratch every time.

        Parameters:
            :output_path: the path of the output file.

        Returns:
            A file named 'augment.log' in the same directory as the output file.
            Example:
                {"filename": "train.jsonl", "idx": 110}
                {"filename": "dev.jsonl", "idx": 0}
        '''
        dirname = os.path.dirname(output_path)
        basename = os.path.basename(output_path)
        log_path = os.path.join(dirname, 'augment.log')
        self.log_path = log_path
        self.last_idx = 0
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.logs = []
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f_r:
                for line in f_r:
                    js = json.loads(line)
                    if js['filename'] == basename:
                        self.last_idx = int(js['idx'])
                    else:
                        self.logs.append(js)
        self.logs.append({'filename': basename, 'idx': self.last_idx})

    def update(self, idx: int):
        '''
        Usage:
            Update the last index of the augmented data.

        Parameters:
            :idx: the new index.

        Returns:
            Rewrite the 'augment.log' file with the new index.
        '''
        with open(self.log_path, 'w', encoding='utf-8') as f_a:
            self.logs[-1]['idx'] = idx
            for js in self.logs:
                f_a.write(json.dumps(js, ensure_ascii=False) + '\n')
        self.last_idx = idx

    def set_zero(self):
        '''
        Usage:
            Set the last index of the augmented data to 0.

        Returns:
            Rewrite the 'augment.log' file to set the last index to 0.
        '''
        self.last_idx = 0
        self.logs[-1]['idx'] = 0
        with open(self.log_path, 'w', encoding='utf-8') as f_a:
            for js in self.logs:
                f_a.write(json.dumps(js, ensure_ascii=False) + '\n')

def load_jsonl(file_path: str) -> list[dict]:
    '''
    Usage:
        Load a jsonl file into a list of dictionaries.
        If the json data is not in a valid jsonl format, it will be skipped.
        There are two possible formats:
            1. Each line is a valid json string.
            Example:
                {"text": "This is a sample text."}
            2. There is a indention of 4 spaces.
            Example:
                {
                    "text": "This is a sample text."
                    "labels": [
                        "label1", 
                        "label2"
                    ]
                }
    
    Parameters:
        :file_path: the path of the jsonl file.

    Returns:
        A list of dictionaries.
    '''
    lines = open(file_path, 'r', encoding='utf-8', errors='ignore').readlines()
    if not lines:
        return []
    data_list = []

    if lines[0] == '{\n':
        json_lines = []
        for line in lines:
            line = line.strip()
            if line:
                json_lines.append(line)
            
            if line == '}':
                json_str = ''.join(json_lines)
                json_obj = json.loads(json_str)
                data_list.append(json_obj)
                json_lines = []

    else:
        for line in lines:
            line = line.strip()
            if line:
                data_list.append(json.loads(line))

    return data_list

alpaca_prompt = (
    "Below is an instruction that describes a task, paired with an input that provides further context. "
    "Write a response that appropriately completes the request."
    "### Instruction:"
    "{}"
    "### Input:"
    "{}"
    "### Response:"
    "{}"
)
