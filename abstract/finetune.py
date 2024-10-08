from dotenv import load_dotenv
load_dotenv()
from unsloth import FastLanguageModel, is_bfloat16_supported
from datasets import load_dataset
from trl import SFTTrainer
from transformers import TrainingArguments
from dataAug import DataAugmentation
from typing import Any, Callable
import torch

class FineTune:
    def __init__(self, 
        model_name: str = "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",      
        max_seq_length: int = 200,                 
        dtype = None,                               
        load_in_4bit: bool = True,                 
        Evaluator: Any = None, 
        pool: Any = None
    ):
        '''
        Parameters:
            :model_name: See models at https://huggingface.co/unsloth
            :max_seq_length: Choose any! We auto support RoPE Scaling internally!
            :dtype: None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
            :load_in_4bit: Use 4bit quantization to reduce memory usage. Can be False.
            :Evaluator: A class that has a method `evaluate` that takes a file path and returns a dictionary of metrics.
            :pool: A pool to use for data augmentation.
        '''
        self.model_name = model_name
        self.max_seq_length = max_seq_length 
        self.dtype = dtype 
        self.load_in_4bit = load_in_4bit 
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
        '''
        Usage:
            Input the r and lora_alpha values to get a PEFT model.
            The PEFT model is a modified version of the original model that uses 
                LoRA (Rank-Stabilized LoRA) to reduce the memory usage.

        Parameters:
            :r: The number of rounds of LoRA to use.
            :lora_alpha: The alpha value for LoRA.

        Returns:
            A PEFT model with the specified r and lora_alpha values.
        '''

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

    def finetune(self, 
        formatting_prompts_func: Callable,
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
        aug_funcs: list[Callable] = [],
        metric: str = "f1_score",
        aug_threshold: float = 0.02,
        output_info: str = ''
    ):
        '''
        Usage:
            Finetunes a model on a dataset using a formatting function to create prompts.
            Augment the wrong predictions using a list of augmentation functions.
            Saves the best model based on a metric.

        Parameters:
            :formatting_prompts_func: A function that takes a sample and returns a formatted prompt.
                Parameters:
                    :examples: A DatasetDict object containing the examples.
                    :EOS: The end-of-sentence token.
                Returns:
                    A list of formatted prompts wrapped in a Dict
                    Example:
                        {"text": ["The quick brown fox jumps over the lazy dog.", "The quick brown fox jumps over the lazy dog."]}
                Example:
                    def formatting_prompts_func(examples, EOS):
                        instructions = examples["instruction"]
                        inputs       = examples["input"]
                        outputs      = examples["output"]
                        texts = []
                        for instruction, input, output in zip(instructions, inputs, outputs):
                            # Must add EOS_TOKEN, otherwise your generation will go on forever!
                            text = alpaca_prompt.format(instruction, input, output) + EOS
                            texts.append(text)
                        return { "text" : texts, }
            :max_step_each: The maximum number of steps to train for each iteration.
            :learning_rate: The learning rate to use for training.
            :train_dataset_path: The path to the training dataset.
            :test_dataset_path: The path to the test dataset.
            :wrong_dataset_path: The path to the write the wrong predictions dataset.
            :model_save_path: The path to save the best model according to the metric.
            :max_iter: The maximum number of iterations to run. The end of each iteration is to augment the wrong predictions.
            :r: The number of rounds of LoRA to use.
            :lora_alpha: The alpha value for LoRA.
            :repeat_num: The number of times to repeat the augmentation.
            :aug_funcs: A list of augmentation functions to use.
                parameters:
                    :js: the input data(dict)
                    :history: the history of the generated input, a list of str
                    :return: the augmented input

                Example:
                    def prompt_func(js: dict, history: list[str]) -> str:
                        prompt = aug_prompt.format(
                            input=js['input'], 
                            intentions=js['query'], 
                            history=history, 
                        )
                        return prompt
            :metric: The metric to use for evaluation.
            :aug_threshold: The threshold for augmentation. If the score does not improve by this amount, stop augmenting.
            :output_info: A string to output at the beginning of the results.txt file.

        Returns:
            Save the best model according to the metric to the model_save_path.
            Write the results to a file called "results.txt".
            Write the wrong predictions to the wrong_dataset_path.
        '''
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
        train_dataset = dataset.map(formatting_prompts_func, batched = True, fn_kwargs={"EOS": self.EOS_TOKEN})

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

            if not aug_funcs:
                model.save_pretrained(model_save_path)
                self.tokenizer.save_pretrained(model_save_path)

            evaluator = self.Evaluator(model, self.tokenizer, max_new_tokens=100)
            result = evaluator.evaluate(test_file_path=test_dataset_path, wrong_output_path=wrong_dataset_path)
            
            for key, value in result.items():
                print(f"{key}: {value: 0.4f}")

            if metric not in result:
                print(f"Metric {metric} not found in result. Available metrics: {result.keys()}")
                return
            score = result[metric]

            with open(wrong_dataset_path, 'r', encoding='utf-8') as f:
                length = len(f.readlines())

            with open('results.txt', 'a') as f:
                if output_info:
                    f.write(f'{output_info}\n')
                else:
                    f.write(f'Run {i+1}\n')
                for key, value in result.items():
                    f.write(f'\t{key}: {value: 0.4f}\n')
                f.write(f'\ttrain dataset size: {len(train_dataset)}\n')
                f.write(f'\twrong dataset size: {length}\n\n')
                f.flush()

            if score > last_score:
                model.save_pretrained(model_save_path)
                print(f"Model saved at {model_save_path}")
                self.tokenizer.save_pretrained(model_save_path)
                if score - last_score <= aug_threshold:     # Stop augmenting if score does not improve by aug_threshold.
                    break
                last_score = score
            else:       # If score does not improve, do not save the model and break
                break

            if not aug_funcs:
                break

            for aug_func in aug_funcs:      # Augment the wrong predictions using the list of augmentation functions.
                dataAug = DataAugmentation.from_file(wrong_dataset_path)
                dataAug.augment(pool=self.pool, prompt_func=aug_func, output_path=train_dataset_path, from_log=False, repeat_num=repeat_num)

            dataset = load_dataset('json', data_files=train_dataset_path, split='train')        # reload the augmented dataset
            train_dataset = dataset.map(formatting_prompts_func, batched = True, fn_kwargs={"EOS": self.EOS_TOKEN})

            del model       # Free up memory
            torch.cuda.empty_cache()
