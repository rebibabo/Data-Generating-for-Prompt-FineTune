from dotenv import load_dotenv
load_dotenv()
import os
import json
from openai import OpenAI
from loguru import logger
from abc import ABC, abstractmethod
import numpy as np
import random
import torch

logger.remove(0)
logger.add("data_aug.log", format="<level>{level}</level> | <level>{message}</level>", rotation="10 MB")

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
        with open(self.log_path, 'w', encoding='utf-8') as f_a:
            self.logs[-1]['idx'] = idx
            for js in self.logs:
                f_a.write(json.dumps(js, ensure_ascii=False) + '\n')
        self.last_idx = idx

    def set_zero(self):
        self.last_idx = 0
        self.logs[-1]['idx'] = 0
        with open(self.log_path, 'w', encoding='utf-8') as f_a:
            for js in self.logs:
                f_a.write(json.dumps(js, ensure_ascii=False) + '\n')

def load_jsonl(file_path: str) -> list[dict]:
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
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    "### Instruction:"
    "{}"
    "### Input:"
    "{}"
    "### Response:"
    "{}"
)
