import sys
sys.path.append('..')

from abstract.dataAug import DataAugmentation
from abstract.queryPool import QueryPool
from prompt import natural_prompt, correct_prompt, lazy_prompt, implicit_prompt

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

    dataAug = DataAugmentation.from_file('../dataset/clean_seed.json')

    lazy_augment = dataAug.augment(
        pool=pool,
        prompt_func=lazy_func,
        output_path='../dataset/lazy_augment.jsonl',
        from_log=False,
        indent=4
    )

    implicit_augment = dataAug.augment(
        pool=pool,
        prompt_func=implicit_func,
        output_path='../dataset/implicit_augment.jsonl',
        from_log=False,
        indent=4
    )

if __name__ == '__main__':
    main()