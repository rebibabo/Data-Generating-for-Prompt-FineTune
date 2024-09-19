from dotenv import load_dotenv
load_dotenv()
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from dataAug import DataAugmentation
from prompt import lazy_prompt, implicit_prompt
from abc import ABC, abstractmethod
from typing import Any
import torch

class ABCFineTune(ABC):
    def __init__(self, model_name, max_seq_length, dtype, load_in_4bit, Evaluator: Any, pool: Any):
        self.model_name = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit" # See models at https://huggingface.co/unsloth
        self.max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
        self.dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
        self.load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
        self.Evaluator = Evaluator
        self.pool = pool
        self.model, self.tokenizer = FastLanguageModel.from_pretrained(
            model_name = model_name,
            max_seq_length = max_seq_length,
            dtype = dtype,
            load_in_4bit = load_in_4bit,
        )
        self.EOS_TOKEN = self.tokenizer.eos_token # Must add EOS_TOKEN

    def get_peft_model(self, r: int = 16, lora_alpha: int = 16):
        model = FastLanguageModel.get_peft_model(
            self.model,
            r = r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj",],
            lora_alpha = lora_alpha,
            lora_dropout = 0, # Supports any, but = 0 is optimized
            bias = "none",    # Supports any, but = "none" is optimized
            # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
            use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
            random_state = 3407,
            use_rslora = False,  # We support rank stabilized LoRA
            loftq_config = None, # And LoftQ
        )
        return model

    @abstractmethod
    def formatting_prompts_func(self, examples):
        pass

    def finetune(self, 
        max_step_each: int = 60,
        learning_rate: float = 2e-4,
        train_dataset_path: str = "dataset/train.jsonl",
        test_dataset_path: str = "dataset/test.jsonl",
        wrong_dataset_path: str = "dataset/wrong_data.jsonl",
        model_save_path: str = "lora_model",
        max_iter: int = 10,
        r: int = 16,
        lora_alpha: int = 16,
        repeat_num: int = 3,
        metric: str = "",
        aug_threshold: float = 0.02
    ):

        arguments = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = max_step_each,
            learning_rate = learning_rate,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
        )

        dataset = load_dataset('json', data_files=train_dataset_path, split='train')
        train_dataset = dataset.map(self.formatting_prompts_func, batched = True,)

        last_score = 0

        for i in range(max_iter):

            model = self.get_peft_model(r=r, lora_alpha=lora_alpha)

            trainer = SFTTrainer(
                model = model,
                tokenizer = self.tokenizer,
                train_dataset = train_dataset,
                dataset_text_field = "text",
                max_seq_length = self.max_seq_length,
                dataset_num_proc = 1,
                packing = False, # Can make training 5x faster for short sequences.
                args = arguments
            )

            trainer_stats = trainer.train()

            evaluator = self.Evaluator(model, self.tokenizer)
            result = evaluator.evaluate(test_file=test_dataset_path, wrong_output_path=wrong_dataset_path)
            
            if metric not in result:
                print(f"Metric {metric} not found in result. Available metrics: {result.keys()}")
                return
            score = result[metric]

            with open(wrong_dataset_path, 'r', encoding='utf-8') as f:
                length = len(f.readlines())

            with open('results.txt', 'a') as f:
                f.write(f'Run {i+1}\n')
                for key, value in result.items():
                    f.write(f'\t{key}: {value: 0.4f}\n')
                f.write(f'\ttrain dataset size: {len(train_dataset)}\n')
                f.write(f'\twrong dataset size: {length}\n\n')
                f.flush()

            if score > last_score:
                last_score = score
                model.save_pretrained(model_save_path)
                print(f"Model saved at {model_save_path}")
                self.tokenizer.save_pretrained(model_save_path)
                if score - last_score <= aug_threshold:
                    break
            else:
                break

            dataAug = DataAugmentation.from_file(wrong_dataset_path)
            dataAug.augment(self.pool, lazy_prompt, output_path=train_dataset_path, from_log=False, repeat_num=repeat_num)

            dataAug = DataAugmentation.from_file(wrong_dataset_path)
            dataAug.augment(self.pool, implicit_prompt, output_path=train_dataset_path, from_log=False, repeat_num=repeat_num)

            dataset = load_dataset('json', data_files=train_dataset_path, split='train')
            train_dataset = dataset.map(self.formatting_prompts_func, batched = True,)

            del model
            torch.cuda.empty_cache()
