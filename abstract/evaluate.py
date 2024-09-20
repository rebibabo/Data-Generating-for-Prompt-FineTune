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
        self.model = model
        self.tokenizer = tokenizer
        FastLanguageModel.for_inference(self.model) # Enable native 2x faster inference

    def inference(self, 
        prompt: str,
        stream: bool = False
    ) -> str:
        
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
        pass

    @abstractmethod
    def metric(self, pred: Any, gold: Any) -> dict[str, float]:
        pass

    @abstractmethod
    def is_wrong(self, pred: Any, gold: Any) -> bool:
        pass

    def evaluate(self, 
        test_file_path: str,
        wrong_output_path: str = '',
    ) -> dict[str, float]:
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