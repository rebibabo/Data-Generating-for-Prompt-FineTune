from dataAug import DataAugmentation
from prompt import lazy_prompt, implicit_prompt

dataAug = DataAugmentation.from_file('dataset/wrong_data.jsonl')
augDataset = dataAug.augment(lazy_prompt, output_path='dataset/train.jsonl', from_log=False, repeat_num=3)

dataAug = DataAugmentation.from_file('dataset/wrong_data.jsonl')
augDataset = dataAug.augment(implicit_prompt, output_path='dataset/train.jsonl', from_log=False, repeat_num=3)