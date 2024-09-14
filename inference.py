from dotenv import load_dotenv
load_dotenv()
from unsloth import FastLanguageModel
import json

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

with open('test.jsonl', 'r') as f:
    lines = f.readlines()
    for line in lines:
        data = json.loads(line)
        instruction = data['instruction']
        input_ = data['input']
        output = query(instruction, input_).split('### Response:\n')[-1]
        print('input:', input_)
        print('generated output:', output)
        print('golden output:', data['output'])
        print()