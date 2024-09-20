from prompt import correct_prompt, natural_prompt, lazy_prompt, implicit_prompt
import json
from rouge_chinese import Rouge
import jieba
from typing import Literal
from utils import Log, query, logger
import numpy as np
from tqdm import tqdm
import os

class QueryPool:
    def __init__(self, 
        pool_size: int = 10,
        min_natural_score: int = 7, 
        min_correct_score: int = 7,
        repeat_time: int = 2
    ):
        self.pool_size = pool_size
        self.min_natural_score = min_natural_score
        self.min_correct_score = min_correct_score
        self.repeat_time = repeat_time
        self.output_js = []
        self.questions = []
        self.queries = []

    def get_scores(self, response: str):
        try:
            if response and response[0] == '[' and response[-1] == ']': 
                scores = eval(response)
                if isinstance(scores, list) and len(scores) == len(self.questions):
                    if not all(isinstance(score, (int, float)) for score in scores):
                        logger.error("🐞 All items in 'scores' must be numeric values.")
                        return -1
                    else:
                        return scores
                else:
                    logger.error(f"🐞 Invalid score response: {response}")
                    return -1
            else:
                logger.error(f"🐞 Invalid score response: {response}")
                return -1
        except Exception as e:
            logger.exception(f"🐞 Error in get_score function: {e}")
            return -1

    def get_naturalness_scores(self):
        prompt = natural_prompt.format(str(self.questions))
        tot_score = np.zeros(len(self.questions))
        for _ in range(self.repeat_time):
            while True:
                response = query(prompt)
                scores = self.get_scores(response)
                if scores != -1:
                    break
            tot_score += np.array(scores)
        scores = tot_score / self.repeat_time
        return scores

    def get_correctness_scores(self):
        questions_queries = ''
        for i, (question, query_) in enumerate(zip(self.questions, self.queries)):
            questions_queries += f"#问题{i+1}#\n{question}\n#意图{i+1}#\n{query_}\n\n"
        prompt = correct_prompt.format(questions_queries)
        tot_score = np.zeros(len(self.questions))
        for _ in range(self.repeat_time):
            while True:
                response = query(prompt)
                scores = self.get_scores(response)
                if scores != -1:
                    break
            tot_score += np.array(scores)
        scores = tot_score / self.repeat_time
        return scores
            
    def add_query(self, js: dict, last=False):
        if not last and len(self.queries) < self.pool_size:
            self.questions.append(js['input'])
            self.queries.append(js['query'])
            self.output_js.append(js)
            return []
        else:
            output_js = []
            if self.min_natural_score < 0:
                naturalness_scores = [0] * self.pool_size
            else:
                naturalness_scores = self.get_naturalness_scores()
            if self.min_correct_score < 0:
                correctness_scores = [0] * self.pool_size
            else:
                correctness_scores = self.get_correctness_scores()
            for n_score, c_score, js in zip(naturalness_scores, correctness_scores, self.output_js):
                if n_score < self.min_natural_score:
                    logger.warning(f"🥵 The naturalness of user input is too low: {js['input']} => {n_score}")
                elif c_score < self.min_correct_score:
                    logger.warning(f"🥶 The correctness of intentions is too low: {js['input']} {js['query']} => {c_score}")
                else:
                    output_js.append(js)
            self.questions = []
            self.queries = []
            self.output_js = []
            return output_js

class DataAugmentation:
    rouge = Rouge()
    
    def __init__(self):
        self.dataset = []
        self.references = []

    @ staticmethod
    def from_file(file_path: str):
        dataAug = DataAugmentation()
        if not os.path.exists(file_path):
            logger.error(f"🐞 File not found: {file_path}")
            return None
        
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                dataAug.dataset = json.load(f)
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    dataAug.dataset.append(json.loads(line))
        else:
            logger.error(f"🐞only support json or jsonl")

        return dataAug

    @ staticmethod
    def from_dataset(dataset: list[dict]):
        dataAug = DataAugmentation()
        dataAug.dataset = dataset
        return dataAug

    def get_score(self, response: str):
        try:
            if response and response[0].isdigit() and len(response) <= 2:
                score = int(response)
                return score
            else:
                logger.error(f"🐞 Invalid score response: {response}")
                return -1
        except Exception as e:
            logger.exception(f"🐞 Error in get_score function: {e}")
            return -1

    def _insert(self, 
        js: dict,
        pool: QueryPool,
        last: bool = False,
        rouge_type: Literal['rouge-1', 'rouge-2', 'rouge-l'] = 'rouge-l',
        rouge_metric: Literal['f', 'p', 'r'] ='r',
        min_rouge_score: float = 0.7,
    ) -> list[dict]:
        user_input = js["input"]

        if len(user_input) > 100:
            logger.warning(f"🤮 The length of user input is too long")
            return []

        hypothesis = ' '.join(jieba.cut(user_input))
        if min_rouge_score > 0:
            for reference in self.references:
                scores = self.rouge.get_scores(hypothesis, reference)
                score = scores[0][rouge_type][rouge_metric]
                if score > min_rouge_score:
                    logger.warning(f"🤢 repetitve user input: {user_input} with score {score:.4f}")
                    return []

        self.references.append(hypothesis)
        output_js = pool.add_query(js, last=last)
        self.dataset.extend(output_js)
        for js in output_js:
            logger.success(f"🎉 Successfully add the user input: {js['input']}")
        return output_js

    def cleanse(self, 
        save_path='', 
        pool_size=10, 
        min_natural_score=7, 
        min_correct_score=7, 
        **kwargs
    ) -> list[dict]:
        pool = QueryPool(pool_size=pool_size, min_natural_score=min_natural_score, min_correct_score=min_correct_score)
        dataset = self.dataset.copy()
        length = len(dataset)
        self.dataset = []
        for i, js in tqdm(enumerate(dataset)):
            last = (i == length - 1)
            with tqdm.external_write_mode():
                self._insert(js, pool, last=last, **kwargs)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, ensure_ascii=False, indent=4)
        return self.dataset

    def augment(self, 
        aug_prompt: str,
        output_path: str,
        repeat_num: int = 3,
        from_log: bool = True,
        pool_size: int = 10,
        indent: int = None, 
        min_natural_score: int = 7,
        min_correct_score: int = 7,
        **kwargs
    ) -> list[dict]:
        pool = QueryPool(pool_size=pool_size, min_natural_score=min_natural_score, min_correct_score=min_correct_score)
        log = Log(output_path)
        last_idx = log.last_idx if from_log else 0
        f = open(output_path, 'a', encoding='utf-8')
        augment_dataset = []
        for i, js in tqdm(enumerate(self.dataset[last_idx:]), total=len(self.dataset) - last_idx):
            with tqdm.external_write_mode():
                input_ = js['input']
                history = []
                for j in range(repeat_num):
                    prompt = aug_prompt.format(
                        input=input_, 
                        intentions=js['query'], 
                        history=history, 
                    )
                    aug_input = query(prompt)
                    js['input'] = aug_input
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

if __name__ == '__main__':
    # dataAug = DataAugmentation.from_file('../dataset/seed.json')
    # dataset = dataAug.cleanse(save_path='../dataset/clean_seed.json')

    dataAug = DataAugmentation.from_file('../dataset/clean_seed.json')

    # lazy_augment = dataAug.augment(
    #     lazy_prompt, 
    #     output_path='dataset/lazy_augment.jsonl'
    #     from_log=False
    # )

    implicit_augment = dataAug.augment(
        implicit_prompt, 
        output_path='../dataset/implicit_augment.jsonl',
        from_log=False
    )