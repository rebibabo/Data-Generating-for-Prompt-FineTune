from unsloth import FastLanguageModel
from main import Evaluator

max_seq_length: int = 200,                 
dtype = None,                               
load_in_4bit: bool = True,  

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "../lora_model", # YOUR MODEL YOU USED FOR TRAINING
    load_in_4bit = load_in_4bit,
)
FastLanguageModel.for_inference(model)

evaluator = Evaluator(model, tokenizer, 100)
result = evaluator.evaluate(test_file_path="../dataset/test.jsonl")