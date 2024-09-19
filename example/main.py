import sys
import time
sys.path.append('..')

from abstract.dataAug import DataAugmentation
from abstract.queryPool import QueryPool
from abstract.finetune import ABCFineTune
from abstract.evaluate import ABCEvaluator
from prompt import natural_prompt, correct_prompt, lazy_prompt, implicit_prompt, alpaca_prompt

class Pool(QueryPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_score_prompts(self) -> dict[str, str]:
        questions = [x['input'] for x in self.output_js]
        queries = [x['query'] for x in self.output_js]
        _natural_prompt = natural_prompt.format(str(questions))
        questions_queries = ''
        for i, (question, query_) in enumerate(zip(questions, queries)):
            questions_queries += f"#问题{i+1}#\n{question}\n#意图{i+1}#\n{query_}\n\n"
        _correct_prompt = correct_prompt.format(questions_queries)
        return {'correct': _correct_prompt, 'natural': _natural_prompt}
        
    def get_score_thresholds(self) -> dict[str, float]:
        return {'correct': 7, 'natural': 7}

    def get_prompt_key_name(self) -> dict[str, list[str]]:
        return {'correct': ['input', 'query'], 'natural': ['input']}

class Evaluator(ABCEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, data: dict) -> tuple[str, str]:
        gold = data['output']
        instruction = data['instruction']
        input_ = data['input']
        while True:
            try:
                prompt = alpaca_prompt.format(
                    instruction,    # instruction
                    input_,         # input
                    "",             # output - leave this blank for generation!
                )
                output = eval(self.inference(prompt).split('### Response:\n')[-1])
                break
            except:
                time.sleep(1)
        return output, gold

    def metric(self, pred: str, gold: str) -> dict[str, float]:
        TP = len(set(pred) & set(gold))
        P = len(pred)
        R = len(gold)
        precision = TP / P if P > 0 else 0
        recall = TP / R if R > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {'precision': precision,'recall': recall, 'f1_score': f1_score}
    
    def is_wrong(self, pred, gold):
        return pred != gold

class FineTune(ABCFineTune):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def formatting_prompts_func(self, examples):
        instructions  = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            # Must add EOS_TOKEN, otherwise your generation will go on forever!
            text = alpaca_prompt.format(instruction, input, output) + self.EOS_TOKEN
            texts.append(text)
        return { "text" : texts, }

def lazy_func(js: dict, history: list[str]) -> str:
    prompt = lazy_prompt.format(
        input=js['input'], 
        intentions=js['input'], 
        history=history, 
    )
    return prompt

def implicit_func(js: dict, history: list[str]) -> str:
    prompt = implicit_prompt.format(
        input=js['input'], 
        intentions=js['input'], 
        history=history, 
    )
    return prompt

def main():
    pool = Pool(pool_size=10, repeat_time=2)
    # dataAug = DataAugmentation.from_file('../dataset/seed.json', ref=False)
    # dataAug.cleanse(pool, save_path='../dataset/clean_seed.json')

    # dataAug = DataAugmentation.from_file('../dataset/clean_seed.json')

    # lazy_augment = dataAug.augment(
    #     pool=pool,
    #     prompt_func=lazy_func,
    #     output_path='../dataset/lazy_augment.jsonl',
    #     from_log=False,
    #     indent=4
    # )

    # implicit_augment = dataAug.augment(
    #     pool=pool,
    #     prompt_func=implicit_func,
    #     output_path='../dataset/implicit_augment.jsonl',
    #     from_log=False,
    #     indent=4
    # )

    fineTune = FineTune(
        model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        Evaluator=Evaluator,
        pool=pool,
    )

    fineTune.finetune(
        max_step_each = 60,
        learning_rate = 2e-4,
        train_dataset_path = "dataset/train.jsonl",
        test_dataset_path = "dataset/test.jsonl",
        wrong_dataset_path = "dataset/wrong_data.jsonl",
        max_iter = 10,
        r = 16,
        lora_alpha = 16,
        repeat_num = 2,
        metric = "f1_score",
        aug_threshold = 0.02
    )

if __name__ == '__main__':
    main()