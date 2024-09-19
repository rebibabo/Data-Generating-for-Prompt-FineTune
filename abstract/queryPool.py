from abc import ABC, abstractmethod
from utils import query, logger
from copy import deepcopy
import numpy as np

class QueryPool(ABC):
    def __init__(self, 
        pool_size: int = 10,
        repeat_time: int = 2,
    ):
        self.pool_size = pool_size
        self.repeat_time = repeat_time
        self.output_js = []

    @abstractmethod
    def get_score_prompts(self) -> dict[str, str]:
        pass

    @abstractmethod
    def get_score_thresholds(self) -> dict[str, float]:
        pass

    @abstractmethod
    def get_prompt_key_name(self) -> dict[str, list[str]]:
        pass

    def get_scores(self, response: str):
        try:
            if response and response[0] == '[' and response[-1] == ']': 
                if not response[1:-1].strip().replace(' ','').replace('\n','').replace(',', '').isdigit():
                    logger.error("🐞 All items in 'scores' must be numeric values.")
                    return -1
                scores = eval(response)
                if isinstance(scores, list):
                    if len(scores) == len(self.output_js):
                        return scores
                    else:
                        logger.error(f"🐞 The number of scores does not match the number of inputs: {len(scores)} vs {len(self.output_js)}")
                        return -1
            logger.error(f"🐞 Invalid score response: {response}")
            return -1
        except Exception as e:
            logger.exception(f"🐞 Error in get_score function: {e}")
            return -1

    def get_all_scores(self):
        prompts = self.get_score_prompts()
        tot_scores = {}
        for name, prompt in prompts.items():
            tot_score = np.zeros(len(self.output_js))
            for _ in range(self.repeat_time):
                while True:
                    response = query(prompt)
                    scores = self.get_scores(response)
                    if scores != -1:
                        break
                tot_score += np.array(scores)
            scores = tot_score / self.repeat_time
            tot_scores[name] = scores.tolist()
        return tot_scores
            
    def add_query(self, js: dict, last=False) -> list[dict]:
        if not last and len(self.output_js) < self.pool_size:
            self.output_js.append(js)
            return []
        else:
            output_js = deepcopy(self.output_js)
            all_scores = self.get_all_scores()
            for name, threshold in self.get_score_thresholds().items():
                prompt_key_names = self.get_prompt_key_name()[name]
                if threshold < 0:
                    scores = [0] * len(self.output_js)
                else:
                    scores = all_scores[name]
                for score, js in zip(scores, self.output_js):
                    if score < threshold:
                        prompt_key_values = ' '.join([str(js[key_name]) for key_name in prompt_key_names])
                        logger.warning(f"🥶 The {name} score of user input is too low: {prompt_key_values} => {score}")
                        if js in output_js:
                            output_js.remove(js)
                        break
            self.output_js = []
            return output_js
