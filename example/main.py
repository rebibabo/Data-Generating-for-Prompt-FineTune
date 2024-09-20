import sys
import time
sys.path.append('..')

from abstract.queryPool import QueryPool
from abstract.finetune import FineTune
from abstract.evaluate import ABCEvaluator
from prompt import natural_prompt, correct_prompt, lazy_prompt, implicit_prompt, alpaca_prompt

class Pool(QueryPool):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_score_prompts(self) -> dict[str, str]:
        questions = [x['input'] for x in self.input_js]
        queries = [x['query'] for x in self.input_js]
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
        prompt = alpaca_prompt.format(
            instruction,    # instruction
            input_,         # input
            "",             # output - leave this blank for generation!
        )
        try_num = 0
        while True:
            output = self.inference(prompt).split('### Response:')[-1].replace('\n', '').strip()
            try:
                output = eval(output)
                break
            except:
                try_num += 1
                print(f"Invalid output: {output}. Retrying {try_num}...")
                time.sleep(1)
            if try_num >= 3:
                print("Failed to generate output. Please check the input and query.")
                return [], gold
        return output, gold

    def metric(self, pred, gold) -> dict[str, float]:
        TP = len(set(pred) & set(gold))
        P = len(pred)
        R = len(gold)
        precision = TP / P if P > 0 else 0
        recall = TP / R if R > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return {'precision': precision,'recall': recall, 'f1_score': f1_score}
    
    def is_wrong(self, pred, gold):
        return pred != gold

def formatting_prompts_func(examples, EOS):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS
        texts.append(text)
    return { "text" : texts, }

def lazy_func(js: dict, history: list[str]) -> str:
    prompt = lazy_prompt.format(
        input=js['input'], 
        intentions=js['query'], 
        history=history, 
    )
    return prompt

def implicit_func(js: dict, history: list[str]) -> str:
    prompt = implicit_prompt.format(
        input=js['input'], 
        intentions=js['query'], 
        history=history, 
    )
    return prompt

def main():
    '''
    unsloth/Qwen2.5-7B-bnb-4bit
    unsloth/Qwen2.5-7B-Instruct-bnb-4bit    
    unsloth/mistral-7b-v0.3-bnb-4bit    
    unsloth/mistral-7b-instruct-v0.3    #
    unsloth/mistral-7b-instruct-v0.3-bnb-4bit
    unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit
    '''
    pool = Pool(pool_size=10, repeat_time=2)

    models = [
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    ]

    fineTune = FineTune(
        model_name = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        Evaluator=Evaluator,
        pool=pool,
    )
    
    r = 64

    fineTune.finetune(
        formatting_prompts_func = formatting_prompts_func,
        max_step_each = 160,
        learning_rate = 2e-4,
        train_dataset_path = "../dataset/train.jsonl",
        test_dataset_path = "../dataset/test.jsonl",
        wrong_dataset_path = "../dataset/wrong_data.jsonl",
        model_save_path="../lora_model",
        max_iter = 10,
        r = r,
        lora_alpha = r,
        repeat_num = 3,
        # aug_funcs=[lazy_func, implicit_func],
        metric = "f1_score",
        aug_threshold = 0.001,
        output_info=f'model: {model}'
    )

if __name__ == '__main__':
    main()