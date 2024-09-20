import json
from rouge_chinese import Rouge
import jieba
from typing import Literal, Any, Callable
from utils import Log, query, logger
from tqdm import tqdm
import os

class DataAugmentation:
    rouge = Rouge()
    
    def __init__(self, key_name: str = 'input'):
        self.dataset = []
        self.references = []
        self.key_name = key_name

    @ staticmethod
    def from_file(file_path: str, key_name: str='input', ref: bool = True):
        dataAug = DataAugmentation()
        if not os.path.exists(file_path):
            logger.error(f"🐞 File not found: {file_path}")
            return None
        
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                dataAug.dataset = json.load(f)
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc='Loading dataset'):
                    js = json.loads(line)
                    dataAug.dataset.append(js)
        else:
            logger.error(f"🐞only support json or jsonl")
            return None

        if ref:
            for js in dataAug.dataset:
                if key_name not in js:
                    logger.error(f"🐞 key_name {key_name} not found in {file_path}")
                    return None
                reference = ' '.join(jieba.cut(js[key_name]))
                dataAug.references.append(reference)

        return dataAug

    @ staticmethod
    def from_dataset(dataset: list[dict], key_name: str='input', ref: bool = True):
        dataAug = DataAugmentation()
        dataAug.dataset = dataset
        if ref:
            for js in dataset:
                if key_name not in js:
                    logger.error(f"🐞 key_name {key_name} not found in dataset")
                    return None
                reference = ' '.join(jieba.cut(js[key_name]))
                dataAug.references.append(reference)
        return dataAug

    def _insert(self, 
        js: dict,
        pool: Any,
        last: bool = False,
        rouge_type: Literal['rouge-1', 'rouge-2', 'rouge-l'] = 'rouge-l',
        rouge_metric: Literal['f', 'p', 'r'] ='r',
        min_rouge_score: float = 0.7,
        max_length: int = 100,
    ) -> list[dict]:
        user_input = js[self.key_name]

        if len(user_input) > max_length:
            logger.warning(f"🤮 The length of user input is too long")
            if not last:
                return []

        hypothesis = ' '.join(jieba.cut(user_input))
        if min_rouge_score > 0:
            for reference in self.references:
                scores = self.rouge.get_scores(hypothesis, reference)
                score = scores[0][rouge_type][rouge_metric]
                if score > min_rouge_score:
                    logger.warning(f"🤢 repetitve user input: {user_input} => {score:.4f}")
                    if not last:
                        return []

        self.references.append(hypothesis)
        output_js = pool.add_query(js, last=last)
        self.dataset.extend(output_js)
        for js in output_js:
            logger.success(f"🎉 Successfully add the user input: {js[self.key_name]}")
        return output_js

    def cleanse(self, 
        pool: Any,
        save_path: str = '', 
        **kwargs
    ) -> list[dict]:
        dataset = self.dataset.copy()
        length = len(dataset)
        self.dataset = []
        for i, js in tqdm(enumerate(dataset), total=length):
            last = (i == length - 1)
            self._insert(js, pool, last=last, **kwargs)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, ensure_ascii=False, indent=4)
        return self.dataset

    def augment(self, 
        pool: Any,
        prompt_func: Callable,
        output_path: str,
        repeat_num: int = 3,
        from_log: bool = True,
        indent: int = None, 
        **kwargs
    ) -> list[dict]:
        log = Log(output_path)
        last_idx = log.last_idx if from_log else 0
        f = open(output_path, 'a', encoding='utf-8')
        augment_dataset = []
        for i, js in tqdm(enumerate(self.dataset[last_idx:]), total=len(self.dataset)-last_idx):
            history = []
            for j in range(repeat_num):
                prompt = prompt_func(js, history)
                aug_input = query(prompt)
                js[self.key_name] = aug_input
                last = (i == len(self.dataset) - 1 and j == repeat_num - 1)
                output_js = self._insert(js.copy(), pool, last=last, **kwargs)
                if output_js:
                    for js in output_js:
                        augment_dataset.append(js)
                        f.write(json.dumps(js, ensure_ascii=False, indent=indent) + '\n')
                history.append(aug_input)
            log.update(i+last_idx)
        f.close()
        return augment_dataset