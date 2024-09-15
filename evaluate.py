from dotenv import load_dotenv
load_dotenv()
from unsloth import FastLanguageModel
import json
from tqdm import tqdm

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "lora_model", # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

from transformers import TextStreamer
def query(instruction, user_input):
    FastLanguageModel.for_inference(model) # Enable native 2x faster inference
    inputs = tokenizer(
    [
        alpaca_prompt.format(
            instruction, # instruction
            user_input, # input
            "", # output - leave this blank for generation!
        )
    ], return_tensors = "pt").to("cuda")

    # text_streamer = TextStreamer(tokenizer)
    # _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 2048)

    output = model.generate(**inputs, max_length = 2048, do_sample = True, top_p = 0.9, top_k = 50, temperature = 1.0, num_return_sequences = 1)
    output = tokenizer.batch_decode(output, skip_special_tokens = True)[0]
    return output

def metric(pred, gold):
    TP = len(set(pred) & set(gold))
    P = len(pred)
    R = len(gold)
    precision = TP / P if P > 0 else 0
    recall = TP / R if R > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1_score

def evaluate(test_file: str, wrong_output_file: str):
    wrong_data = []
    tot_precision, tot_recall, tot_f1_score = 0, 0, 0
    with open(test_file, 'r') as f:
        lines = f.readlines()
        bar = tqdm(total=len(lines))
        for i, line in enumerate(lines):
            data = json.loads(line)
            instruction = data['instruction']
            input_ = data['input']
            output = eval(query(instruction, input_).split('### Response:\n')[-1])
            gold = data['output']
            precision, recall, f1_score = metric(output, gold)
            bar.update(1)
            tot_precision += precision
            tot_recall += recall
            tot_f1_score += f1_score
            bar.set_description(f"P: {tot_precision/(i+1):.4f}, R: {tot_recall/(i+1):.4f}, F1: {tot_f1_score/(i+1):.4f}")
            if output!= gold:
                wrong_data.append(data)
        bar.close()

    with open(wrong_output_file, 'w', encoding='utf-8') as f:
        for data in wrong_data:
            f.write(json.dumps(data) + '\n')