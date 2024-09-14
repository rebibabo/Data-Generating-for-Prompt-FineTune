from openai import OpenAI
from prompt import correct_prompt, natural_prompt, lazy_prompt, implicit_prompt
import json
from rouge_chinese import Rouge
import jieba
from typing import Literal
from utils import Log, query, logger
import os

class DataAugmentation:
    client = OpenAI()
    references = []
    rouge = Rouge()
    dataset = []

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
                logger.error(f"🐞 Invalid score response")
                return -1
        except Exception as e:
            logger.exception(f"🐞 Error in get_score function: {e}")
            return -1

    def _insert(self, 
        user_input: str, 
        output: list[str] = [],
        rouge_type: Literal['rouge-1', 'rouge-2', 'rouge-l'] = 'rouge-l',
        rouge_metric: Literal['f', 'p', 'r'] ='r',
        min_rouge_score: float = 0.7,
        min_correct_score: int = 6,
        min_natural_score: int = 6,
    ) -> bool:
        if len(user_input) > 300:
            logger.warning(f"🤮 The length of user input is too long")
            return False

        if min_rouge_score > 0:
            hypothesis = ' '.join(jieba.cut(user_input))
            for reference in self.references:
                scores = self.rouge.get_scores(hypothesis, reference)
                score = scores[0][rouge_type][rouge_metric]
                if score > min_rouge_score:
                    logger.warning(f"🤢 repetitve user input: {user_input} with score {score:.4f}")
                    return False

        if min_natural_score > 0:
            prompt = natural_prompt.format(user_input)
            response = query(prompt)
            score = self.get_score(response)
            if score < min_natural_score:
                logger.warning(f"🥵 The naturalness of user input is too low: {user_input} => {score}")
                return False

        if min_correct_score > 0:
            for intention in output:
                prompt = correct_prompt.format(user_input, intention)
                response = query(prompt)
                score = self.get_score(response)
                if score < min_correct_score:
                    logger.warning(f"🥶 The correctness of output is too low: {user_input} {output} => {score}")
                    return False

        self.references.append(hypothesis)
        return True

    def cleanse(self, save_path='', **kwargs):
        new_dataset = []
        tot = len(self.dataset)
        for i, js in enumerate(self.dataset):
            user_input = js['input']
            output = js['output']
            if self._insert(user_input, output, **kwargs):
                logger.success(f"🎉 {i+1} / {tot} Successfully add the user input: {user_input}")
                new_dataset.append(js)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(new_dataset, f, ensure_ascii=False, indent=4)
        self.dataset = new_dataset
        return new_dataset

    def augment(self, 
        aug_prompt: str,
        output_path: str,
        repeat_num: int = 3,
        **kwargs
    ):
        log = Log(output_path)
        last_idx = log.last_idx
        f = open(output_path, 'a', encoding='utf-8')
        augment_dataset = []
        for i, js in enumerate(self.dataset[last_idx:]):
            input_ = js['input']
            history = []
            for j in range(repeat_num):
                prompt = aug_prompt.format(
                    input=input_, 
                    intentions=js['query'], 
                    history='\n'.join([
                        f'{i+1}. {h}' for i, h in enumerate(history)
                    ])
                )
                aug_input = query(prompt)
                if self._insert(aug_input, js['output'], **kwargs):
                    js_copy = js.copy()
                    js_copy['input'] = aug_input
                    js_copy['original_input'] = input_
                    logger.success(f"🎉 {i+1+last_idx} / {len(self.dataset)} - {j+1} / {repeat_num} - Successfully add the augmented input: {aug_input}")
                    history.append(aug_input)
                    augment_dataset.append(js_copy)
                    j += 1
                    f.write(json.dumps(js_copy, ensure_ascii=False, indent=4) + '\n')
            log.update(i+last_idx)
        f.close()
        return augment_dataset

if __name__ == '__main__':
    initial_size = 300
    rouge = 'rouge-l'
    rouge_metric = 'r'
    min_rouge_score = 0.7

    # dataAug = DataAugmentation.from_file('dataset/seed.jsonl')
    # dataset = dataAug.cleanse(save_path='dataset/clean_seed.json')

    dataAug = DataAugmentation.from_file('dataset/clean_seed.json')
    # lazy_augment = dataAug.augment(
    #     lazy_prompt, 
    #     output_path='dataset/lazy_augment.jsonl'
    # )

    implicit_augment = dataAug.augment(
        implicit_prompt, 
        output_path='dataset/implicit_augment.jsonl',
    )