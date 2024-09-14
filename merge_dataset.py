import json
from utils import load_jsonl

if __name__ == '__main__':
    with open('dataset/clean_seed.json', 'r', encoding='utf-8', errors='ignore') as f:
        clean_seed = json.load(f)

    with open('dataset/test.jsonl', 'w', encoding='utf-8', errors='ignore') as f:
        for js in clean_seed:
            json.dump(js, f, ensure_ascii=False)
            f.write('\n')

    lazy_augment = load_jsonl('dataset/lazy_augment.jsonl')

    implicit_augment = load_jsonl('dataset/implicit_augment.jsonl')

    merged_dataset = lazy_augment + implicit_augment

    with open('dataset/train.jsonl', 'w', encoding='utf-8', errors='ignore') as f:
        for js in merged_dataset:
            json.dump(js, f, ensure_ascii=False)
            f.write('\n')