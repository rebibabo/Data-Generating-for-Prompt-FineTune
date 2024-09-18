from dotenv import load_dotenv
load_dotenv()
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from evaluate import Evaluator
from dataAug import DataAugmentation
from prompt import lazy_prompt, implicit_prompt

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.

# 4bit pre quantized models we support for 4x faster downloading + no OOMs.
# See models at https://huggingface.co/unsloth

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
    # token = "hf_...", # use one if using gated models like meta-llama/Llama-2-7b-hf
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN
def formatting_prompts_func(examples):
    instructions  = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        # Must add EOS_TOKEN, otherwise your generation will go on forever!
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

arguments = TrainingArguments(
    per_device_train_batch_size = 2,
    gradient_accumulation_steps = 4,
    warmup_steps = 5,
    # num_train_epochs = 1, # Set this for 1 full training run.
    max_steps = 30,
    learning_rate = 2e-4,
    fp16 = not is_bfloat16_supported(),
    bf16 = is_bfloat16_supported(),
    logging_steps = 1,
    optim = "adamw_8bit",
    weight_decay = 0.01,
    lr_scheduler_type = "linear",
    seed = 3407,
    output_dir = "outputs",
)

dataset = load_dataset('json', data_files='dataset/train.jsonl', split='train')
train_dataset = dataset.map(formatting_prompts_func, batched = True,)

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc = 1,
    packing = False, # Can make training 5x faster for short sequences.
    args = arguments
)

for i in range(10):
    trainer_stats = trainer.train()
    model.save_pretrained("lora_model")
    tokenizer.save_pretrained("lora_model")

    evaluator = Evaluator(model, tokenizer)
    result = evaluator.evaluate(test_file='dataset/test.jsonl', wrong_output_path='dataset/wrong_data.jsonl')
    
    with open('results.txt', 'a') as f:
        f.write(f'Run {i+1}: {result}\n')

    dataAug = DataAugmentation.from_file('dataset/wrong_data.jsonl')
    augDataset = dataAug.augment(lazy_prompt, output_path='dataset/train.jsonl', from_log=False, repeat_num=3)
    augDataset = dataAug.augment(implicit_prompt, output_path='dataset/train.jsonl', from_log=False, repeat_num=3)

    dataset = load_dataset('json', data_files='dataset/train.jsonl', split='train')
    train_dataset = dataset.map(formatting_prompts_func, batched = True,)

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 1,
        packing = False, # Can make training 5x faster for short sequences.
        args = arguments
    )
