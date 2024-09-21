import sys
import time
sys.path.append('..')

from abstract.queryPool import QueryPool
from abstract.finetune import FineTune
from abstract.evaluate import ABCEvaluator
from prompt import natural_prompt, correct_prompt, lazy_prompt, implicit_prompt, alpaca_prompt

instruction = '''
你是一个强大的意图识别专家，你能准确地识别输入中的意图类别，如果输入中的意图存在于#意图列表#中，则将其加入到返回结果中。
你的回答应该是一个由[]括起来的列表，只需要返回用户输入中的所有意图列表，不允许解释理由。
一个可能的回答样例为：["云朵大作战","小云果园","AI新头像"]
#意图列表#:
['云朵大作战', '猜谜开红包', '小云果园', 'AI新头像', '邀好友，攒云朵', '送3个月会员（焕新礼）', '连续备份有礼', '天天开盲盒', '召唤相册达人活动', '云盘欢乐透，月月赢好礼', '领1T超大云空间', '组团领红包', '抽抽乐，享好礼', '移动云盘会员日', '云朵中心', '用户回馈活动', '玩转公众号', '开启App通知领好礼', '云端看电影']
'''

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

import concurrent.futures
import time

class Evaluator(ABCEvaluator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def inference_with_timeout(self, prompt, timeout=5):
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(self.inference, prompt)
            try:
                return future.result(timeout=timeout)
            except concurrent.futures.TimeoutError:
                print(f"Inference took longer than {timeout} seconds.")
                return None

    def forward(self, data):
        gold = data['output']
        # instruction = data['instruction']
        input_ = data['input']
        prompt = alpaca_prompt.format(
            instruction,    # instruction
            input_,         # input
            "",             # output - leave this blank for generation!
        )
        # output = self.inference_with_timeout(prompt)
        output = self.inference(prompt)
        if output is None:
            return [], gold
        output = output.split('### Response:')[-1].replace('\n', '').strip()
        try:
            output = eval(output)
        except:
            print("Output is not a valid list.")
            print(output)
            output = []
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
    # for instruction, input, output in zip(instructions, inputs, outputs):
    for input, output in zip(inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        # instruction = instruction.split('。')[0] + '，你的回答应该是一个由[]括起来的列表，只需要返回意图列表，不允许解释理由。' + instruction.split('。')[-1]
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
    "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct",          
    "unsloth/Phi-3-medium-4k-instruct",
    '''
    pool = Pool(pool_size=10, repeat_time=2)

    fourbit_models = [
        # "unsloth/Phi-3.5-mini-instruct",
        # "unsloth/Meta-Llama-3.1-8B-bnb-4bit",
        # "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
        # "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit",
        # "unsloth/mistral-7b-v0.3-bnb-4bit",       
        # "unsloth/Qwen2.5-7B-bnb-4bit", 
        "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",    
        # "unsloth/gemma-2-9b-bnb-4bit",
    ]

    for model in fourbit_models:
        print(model)
        fineTune = FineTune(
            model_name = model,
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
            aug_funcs=[lazy_func, implicit_func],
            metric = "f1_score",
            aug_threshold = 0.001,
            output_info=f'model = {model}'
        )

if __name__ == '__main__':
    main()