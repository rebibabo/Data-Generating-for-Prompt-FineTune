import os
import sys
import json
from openai import OpenAI
from loguru import logger

logger.remove(0)
logger.add(sys.stderr, format="<level>{level}</level> | <level>{message}</level>")
logger.add("data_aug.log", rotation="10 MB")

client = OpenAI()
def query(user_input: str, 
    system_prompt: str = '', 
    model="gpt-4o-mini-2024-07-18", 
    temperature: float = 1.5, 
    max_tokens: int = 2000, 
    seed: int = 42
) -> str:
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
        self.f_w = open(log_path, 'w', encoding='utf-8')

    def update(self, idx: int):
        with open(self.log_path, 'w', encoding='utf-8') as f_a:
            self.logs[-1]['idx'] = idx
            for js in self.logs:
                f_a.write(json.dumps(js, ensure_ascii=False) + '\n')
        self.last_idx = idx
