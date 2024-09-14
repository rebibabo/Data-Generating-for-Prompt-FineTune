import json

with open('dataset/clean_seed.json', 'r', encoding='utf-8') as f:
    clean_seed = json.load(f)

with open('dataset/lazy_augment.json', 'r', encoding='utf-8') as f:
    lazy_augment = json.load(f)

with open('dataset/implicit_augment.json', 'r', encoding='utf-8') as f:
    implicit_augment = json.load(f)

merged_dataset = clean_seed + lazy_augment + implicit_augment

with open('dataset/merged_dataset.jsonl', 'w', encoding='utf-8') as f:
    for data in merged_dataset:
        new_js = {'input': data['input'], 'output': data['output'], 'instruction': data['instruction']}
        json.dump(new_js, f, ensure_ascii=False)
        f.write('\n')