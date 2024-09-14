from openai import OpenAI
from prompt import correct_prompt, natural_prompt, lazy_prompt, implicit_prompt
import json
from rouge_chinese import Rouge
import jieba
from typing import Literal
from utils import Log, query, logger

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

    def _insert(self, user_input: str, output: str='') -> bool:
        if len(user_input) > 300:
            logger.warning(f"🤮 The length of user input is too long: {user_input}")
            return False

        hypothesis = ' '.join(jieba.cut(user_input))
        for reference in self.references:
            scores = self.rouge.get_scores(hypothesis, reference)
            score = scores[0][self.rouge_type][self.rouge_metric]
            if score > self.min_rouge_score:
                logger.warning(f"🤢 repetitve user input: {user_input} with score {score:.4f}")
                return False

        if output:
            prompt = correct_prompt.format(user_input, output)
            response = query(prompt)
            score = self.get_score(response)
            if score < 7:
                logger.warning(f"🥶 The correctness of output is too low: {user_input} {output} => {score}")
                return False

        prompt = natural_prompt.format(user_input)
        response = query(prompt)
        score = self.get_score(response)
        if score < 6:
            logger.warning(f"🥵 The naturalness of user input is too low: {user_input} => {score}")
            return False

        self.references.append(hypothesis)
        return True

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
        output_path: str,
        input_key_name: str = 'input', 
        output_key_name: str = 'output',
        repeat_num: int = 3,
    ):
        log = Log(output_path)
        last_idx = log.last_idx
        f = open(output_path, 'a', encoding='utf-8')
        augment_dataset = []
        for i, js in enumerate(dataset[last_idx:]):
            input_ = js[input_key_name]
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
                if self._insert(aug_input, str(js[output_key_name])):
                    js_copy = js.copy()
                    js_copy[input_key_name] = aug_input
                    js_copy['original_input'] = input_
                    logger.success(f"🎉 {i+1} / {len(dataset)} - {j+1} / {repeat_num} - Successfully add the augmented input: {aug_input}")
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

    dataAug = DataAugmentation(
        rouge_type=rouge,
        rouge_metric=rouge_metric,
        min_rouge_score=min_rouge_score,
    )

    dataset = json.load(open('dataset/seed.json', 'r', encoding='utf-8'))
    clean_dataset = dataAug.load_dataset(
        dataset, 
        input_key_name='input', 
        output_key_name='output',
        raw=True
    )
    with open('dataset/clean_seed.json', 'w', encoding='utf-8') as f:
        json.dump(clean_dataset, f, ensure_ascii=False, indent=4)

    dataset = json.load(open('dataset/clean_seed.json', 'r', encoding='utf-8'))

    lazy_augment = dataAug.augment(
        dataset, 
        lazy_prompt, 
        input_key_name='input', 
        output_key_name='output',
        output_path='dataset/lazy_augment.jsonl'
    )

    # implicit_augment = dataAug.augment(
    #     dataset, 
    #     implicit_prompt, 
    #     input_key_name='input', 
    #     output_key_name='output',
    # )
    # with open('dataset/implicit_augment.json', 'w', encoding='utf-8') as f:
    #     json.dump(implicit_augment, f, ensure_ascii=False, indent=4)