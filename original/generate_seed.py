from dotenv import load_dotenv
load_dotenv()
import pandas as pd
from prompt import example_prompt, relevant_prompt
import random
from utils import query, logger
import json

def generate_seed(initial_size=300, max_query_size=2, min_rel_score=7):
    instruction = (
        "你是一个强大的意图识别专家，你能准确地识别输入中的意图类别，如果输入中的意图存在于#意图列表#中，则将其加入到返回结果中。\n"
        "#意图列表#:\n{intentions}"
    )
    dataset = []
    query_set = set()
    df = pd.read_excel('../dataset/活动映射表.xlsx')
    mapping = {k: v for k, v in zip(df['query'], df['name'])}
    queries = list(mapping.keys())
    instruction = instruction.format(intentions=str(list(set(mapping.values()))))
    while len(dataset) < initial_size:
        random_size = random.randint(1, max_query_size)
        random_queries = list(set(random.sample(queries, random_size)))
        if tuple(random_queries) in query_set:
            logger.warning(f"🤢 repetitve query set: {random_queries}")
            continue
        query_set.add(tuple(random_queries))
        if len(random_queries) > 1:
            prompt = relevant_prompt.format(str(random_queries))
            response = query(prompt)
            if not response or len(response) > 10:
                logger.error(f"🤔 The response is not valid")
                continue
            if response[0].isdigit():
                score = int(response)
                if score < min_rel_score:
                    logger.warning(f"☠️ The relevantness of query set is too low: {random_queries} => {score}")
                    continue
        random_names = list(set([mapping[q] for q in random_queries]))
        prompt = example_prompt.format(query=str(random_queries))
        question = query(prompt)
        if not question:
            continue
        js = {'instruction': instruction, 'input': question, 'query': random_queries, 'output': random_names}
        dataset.append(js)
        logger.success(f"🎉 {len(dataset)} / {initial_size} {question} => {random_queries}")

        with open('../dataset/seed.json', 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=4)

if __name__ == '__main__':
    generate_seed(initial_size=300, max_query_size=3, min_rel_score=7)