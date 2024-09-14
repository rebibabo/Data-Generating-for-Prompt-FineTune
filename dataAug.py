from openai import OpenAI
from prompt import (example_prompt, correct_prompt, relevant_prompt, 
                    natural_prompt, lazy_prompt, implicit_prompt)
import random
import json
from rouge_chinese import Rouge
import jieba
import pandas as pd
from typing import Literal, Union
from loguru import logger

logger.add("data_aug.log", rotation="10 MB")
seed = 42
random.seed(seed)

client = OpenAI()
def query(user_input: str, 
    system_prompt: str = '', 
    model="gpt-4o-mini-2024-07-18", 
    temperature: float = 1.2, 
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

class DataAugmentation:
    client = OpenAI()
    references = []
    rouge = Rouge()

    def __init__(self, 
        rouge_type: Literal['rouge-1', 'rouge-2', 'rouge-l'] = 'rouge-l',
        rouge_metric: Literal['f', 'p', 'r'] ='r',
        min_rouge_score: float = 0.7,
    ):
        self.rouge_type = rouge_type
        self.rouge_metric = rouge_metric
        self.min_rouge_score = min_rouge_score

    def _insert(self, user_input: str, output: str='') -> bool:
        prompt = natural_prompt.format(user_input)
        response = query(prompt)
        if response and response[0].isdigit():
            score = int(response)
            if score < 7:
                logger.warning(f"😭 The naturalness of user input is too low: {user_input} => {score}")
                return False
        else:
            logger.warning(f"😭 Invalid score response: {response}")
            return False

        if output:
            prompt = correct_prompt.format(user_input, output)
            response = query(prompt)
            if response and response[0].isdigit():
                score = int(response)
                if score < 7:
                    logger.warning(f"😭 The correctness of output is too low: {user_input} {output} => {score}")
                    return False
            else:
                logger.warning(f"😭 Invalid score response: {response}")

        hypothesis = ' '.join(jieba.cut(user_input))
        do_insert = True
        for reference in self.references:
            scores = self.rouge.get_scores(hypothesis, reference)
            score = scores[0][self.rouge_type][self.rouge_metric]
            if score > self.min_rouge_score:
                do_insert = False
                logger.warning(f"😭 repetitve user input: {user_input} with score {score:.4f}")
                break
        if do_insert:
            self.references.append(hypothesis)
        return do_insert

    def load_dataset(self, dataset, input_key_name, output_key_name='', raw=False):
        new_dataset = []
        tot = len(dataset)
        for i, js in enumerate(dataset):
            user_input = js[input_key_name]
            output = str(js[output_key_name]) if output_key_name else ''
            if not raw or self._insert(user_input, output):
                logger.success(f"🎉 {i+1} / {tot} Successfully add the user input: {user_input}")
                new_dataset.append(js)
        return new_dataset

    def augment(self, 
        dataset: list[dict], 
        aug_prompt: str,
        input_key_name: str = 'input', 
        prompt_key_name: list[str] = ['input'],
        output_key_name: str = 'output',
        max_size: int = 10,
        min_score: int = 6
    ):
        augmented_dataset = []        
        while len(augmented_dataset) < max_size:
            random_js = random.choice(dataset)
            random_input = random_js[input_key_name]
            prompt_format_value = [random_js[k] for k in prompt_key_name]
            prompt = aug_prompt.format(*prompt_format_value)
            aug_input = query(prompt)
            if self._insert(aug_input, str(random_js[output_key_name])):
                js = random_js.copy()
                js[input_key_name] = aug_input
                js['original_input'] = random_input
                augmented_dataset.append(js)
                logger.success(f"🎉 Successfully add the augmented input: {aug_input}")
            # for line in aug_input.split('\n'):
            #     if not line or not line[0].isdigit():
            #         continue
            #     question = '. '.join(line.split('. ')[1:])
            #     if self._insert(question):
            #         js = random_js.copy()
            #         js[input_key_name] = question
            #         js['original_input'] = random_input
            #         augmented_dataset.append(js)
            #         logger.info(f"🤩 {len(augmented_dataset)} -> {question}")
            #     else:
            #         logger.warning(f"😭 repetitve question: {question}")

        return augmented_dataset

if __name__ == '__main__':
    instruction = (
        "你是一个强大的意图识别专家，你能准确地识别输入中的意图类别，如果输入中的意图存在于#意图列表#中，则将其加入到返回结果中。\n"
        "#意图列表#:\n{intentions}"
    )
    initial_size = 300
    augmentation_size = 100
    min_score = 6
    rouge = 'rouge-l'
    rouge_metric = 'r'
    min_rouge_score = 0.7

    dataAug = DataAugmentation(
        rouge_type=rouge,
        rouge_metric=rouge_metric,
        min_rouge_score=min_rouge_score,
    )

    dataset = []
    query_set = set()
    df = pd.read_excel('活动映射表.xlsx')
    mapping = {k: v for k, v in zip(df['query'], df['name'])}
    queries = list(mapping.keys())
    instruction = instruction.format(intentions=str(list(set(mapping.values()))))
    while len(dataset) < initial_size:
        random_size = random.randint(1, 2)
        random_queries = list(set(random.sample(queries, random_size)))
        if tuple(random_queries) in query_set:
            logger.warning(f"😭 repetitve query set: {random_queries}")
            continue
        query_set.add(tuple(random_queries))
        if len(random_queries) > 1:
            prompt = relevant_prompt.format(str(random_queries))
            response = query(prompt)
            if response and response[0].isdigit():
                score = int(response)
                if score < 7:
                    logger.warning(f"😭 The relevantness of query set is too low: {random_queries} => {score}")
                    continue
        random_names = list(set([mapping[q] for q in random_queries]))
        prompt = example_prompt.format(query=str(random_queries))
        question = query(prompt)
        if not question:
            continue
        js = {'instruction': instruction, 'input': question, 'query': random_queries, 'output': random_names}
        dataset.append(js)
        logger.success(f"🎉 {len(dataset)} / {initial_size} {question} => {random_queries}")

    with open('seed.json', 'w', encoding='utf-8') as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

    dataset = json.load(open('seed.json', 'r', encoding='utf-8'))
    clean_dataset = dataAug.load_dataset(
        dataset, 
        input_key_name='input', 
        output_key_name='output',
        raw=True
    )
    with open('clean_seed.json', 'w', encoding='utf-8') as f:
        json.dump(clean_dataset, f, ensure_ascii=False, indent=4)

    dataset = json.load(open('clean_seed.json', 'r', encoding='utf-8'))

    lazy_augment = dataAug.augment(
        dataset, 
        lazy_prompt, 
        input_key_name='input', 
        prompt_key_name=['input', 'query'],
        output_key_name='output',
        max_size=augmentation_size,
    )
    with open('lazy_augment.json', 'w', encoding='utf-8') as f:
        json.dump(lazy_augment, f, ensure_ascii=False, indent=4)
    
    implicit_augment = dataAug.augment(
        dataset, 
        implicit_prompt, 
        input_key_name='input', 
        prompt_key_name=['input', 'query'],
        output_key_name='output',
        max_size=augmentation_size,
    )
    with open('implicit_augment.json', 'w', encoding='utf-8') as f:
        json.dump(implicit_augment, f, ensure_ascii=False, indent=4)