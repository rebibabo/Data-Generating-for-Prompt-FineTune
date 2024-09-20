from dotenv import load_dotenv
load_dotenv()
from abc import ABC, abstractmethod
from unsloth import FastLanguageModel
from transformers import TextStreamer
import json
from tqdm import tqdm
from typing import Any

class ABCEvaluator(ABC):
    def __init__(self, model, tokenizer):
        '''
        Parameters:
            :model: Hugging Face Transformers model that has been fastened by unsloth
            :tokenizer: Hugging Face Transformers tokenizer
        '''
        self.model = model
        self.tokenizer = tokenizer
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference

    def inference(self, 
        prompt: str,
        stream: bool = False
    ) -> str:
        '''
        Usage:
            Input the prompt and get the generated output.
            If stream is True, the model will generate the output in a streaming way.
            Otherwise, the model will generate the output in a non-streaming way.

        Parameters:
            :prompt: The input prompt
            :stream: Whether to use streaming inference or not. Default is False.

        Returns:
            The generated output.
        '''
        inputs = self.tokenizer([prompt], return_tensors = "pt").to("cuda")

        if stream:
            text_streamer = TextStreamer(self.tokenizer)
            output = self.model.generate(**inputs, streamer = text_streamer, max_new_tokens = 2048)

        else:
            output = self.model.generate(**inputs, max_length = 2048, do_sample = True, top_p = 0.9, top_k = 50, temperature = 1.0, num_return_sequences = 1)
            output = self.tokenizer.batch_decode(output, skip_special_tokens = True)[0]
        return output

    @abstractmethod
    def forward(self, data: dict) -> tuple[str, str]:
        '''
        Usage:
            This is an abstract method that needs to be implemented by the child class.
            It takes in a data dictionary and returns a tuple of predicted output and gold output.

        Parameters:
            :data: A data dictionary containing the input prompt and the gold output.

        Returns:
            A tuple of predicted output and gold output.

        Example:
            def forward(self, data: dict) -> tuple[str, str]:
                gold = data['output']
                instruction = data['instruction']
                input_ = data['input']
                prompt = alpaca_prompt.format(
                    instruction,    # instruction
                    input_,         # input
                    "",             # output - leave this blank for generation!
                )
                output = self.inference(prompt)
                return output, gold
        '''
        pass

    @abstractmethod
    def metric(self, pred: Any, gold: Any) -> dict[str, float]:
        '''
        Usage:
            This is an abstract method that needs to be implemented by the child class.
            It takes in a predicted output and gold output and returns a dictionary of metrics.

        Parameters:
            :pred: The predicted output. It can be str, list of str, or any other type.
            :gold: The gold output. It can be str, list of str, or any other type.

        Returns:
            A dictionary of metrics.
            Example:
                {'bleu": 0.9, 'ppl': 1.2}
                {'p': 0.9, 'r': 0.8, 'f1': 0.85}

        Example:
            def metric(self, pred, gold) -> dict[str, float]:
                TP = len(set(pred) & set(gold))
                P = len(pred)
                R = len(gold)
                precision = TP / P if P > 0 else 0
                recall = TP / R if R > 0 else 0
                f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                return {'precision': precision,'recall': recall, 'f1_score': f1_score}
        '''
        pass

    @abstractmethod
    def is_wrong(self, pred: Any, gold: Any) -> bool:
        '''
        Usage:
            This is an abstract method that needs to be implemented by the child class.
            It takes in a predicted output and gold output and returns a boolean value 
            indicating whether the prediction is wrong or not.

        Parameters:
            :pred: The predicted output. It can be str, list of str, or any other type.
            :gold: The gold output. It can be str, list of str, or any other type.

        Returns:
            A boolean value indicating whether the prediction is wrong or not.
            Notice:
                If the wrong condition is too easy to be satisfied, it will generate too many wrong predictions.
                Thus it will take a long time to augment the wrong data.
                So it is recommended to use a more complex wrong condition like setting a threshold.

        Example:
            def is_wrong(self, pred: str, gold: str) -> bool:
                return pred.lower() != gold.lower()

            A better wrong condition can be:
            def is_wrong(self, pred: str, gold: str) -> bool:
                return similarity(pred, gold) < 0.7
        '''
        pass

    def evaluate(self, 
        test_file_path: str,
        wrong_output_path: str = '',
    ) -> dict[str, float]:
        '''
        Usage:
            This method evaluates the model on a test file and returns a dictionary of metrics.
            If the wrong_output_path is provided, it will save the wrong predictions to the file.

        Parameters:
            :test_file_path: The path to the test file.
            :wrong_output_path: The path to the file to save the wrong predictions. Default is ''.

        Returns:
            A dictionary of metrics.
        '''
        wrong_data = []
        total_metric = {}
        with open(test_file_path, 'r') as f:
            if test_file_path.endswith('.jsonl'):
                lines = f.readlines()
                dataset = [json.loads(line) for line in lines]
            elif test_file_path.endswith('.json'):
                dataset = json.load(f)
            else:
                raise ValueError('Unsupported file format')
        
        for data in tqdm(dataset, desc='Evaluating'):
            pred, gold = self.forward(data)
            metric = self.metric(pred, gold)
            for k, v in metric.items():
                if k not in total_metric:
                    total_metric[k] = 0.0
                total_metric[k] += v
            if self.is_wrong(pred, gold):
                wrong_data.append(data)

        if wrong_output_path:
            with open(wrong_output_path, 'w', encoding='utf-8') as f:
                for data in wrong_data:
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')

        return {k: v / len(dataset) for k, v in total_metric.items()}