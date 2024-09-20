from abc import ABC, abstractmethod
from utils import query, logger
from copy import deepcopy
import numpy as np

class QueryPool(ABC):
    def __init__(self, 
        pool_size: int = 10,
        repeat_time: int = 2,
    ):
        '''
        Usage:
            This is an abstract class for query pool.
            It is used to score a batch(pool_size) of user inputs in an AI-powered way.
            When the pool is full, it will calculate the scores of all the inputs,
            and return the output_js that satisfy all the score thresholds.

        Parameters:
            :pool_size: the maximum number of inputs in the pool.
            :repeat_time: the number of times to repeat the scoring process. 
                The larger it is, the more accurate the scores will be, it will also take longer time.
        '''
        self.pool_size = pool_size
        self.repeat_time = repeat_time
        self.input_js = []     # the pool of user inputs

    @abstractmethod
    def get_score_prompts(self) -> dict[str, str]:
        '''
        Usage:
            This is an abstract method that needs to be implemented by the child class.

        Parameters:
            :self.input_js: a list of input js that need to be scored and filtered.

        Returns:
            A dictionary of prompts for each score type. The prompt can contain the useful info of the self.input_js.
            And make sure that the response to the prompt is a valid score list
            
            Note: Valid means that the response contains the same number of items as the self.input_js.
                And each item in the response is a numeric value.

            Prompt example:
                I want you evaluate the naturality and grammar correctness of the following #question# list:
                You should give #scores# from 1 to 10, with 10 indicating high naturality and correct grammar, 
                and 1 indicating low naturality and incorrect grammar.
                Please return a list of #scores#, one for each question, enclosed in [], without any other reason, 
                and the 'scores' should not appear in the #scores#.

                #question#: 
                [
                    "What is the capital of France?", 
                    "What is the currency of the USA?", 
                    "What is the national anthem of the UK?"
                ]
                #scores#:

            Response example:
                [10, 9, 10]

        Example:
            def get_score_prompts(self) -> dict[str, str]:
                questions = [x['input'] for x in self.input_js]
                queries = [x['query'] for x in self.input_js]
                _natural_prompt = natural_prompt.format(str(questions))
                _correct_prompt = correct_prompt.format(str(queries))
                return {'correct': _correct_prompt, 'natural': _natural_prompt}
        '''
        pass

    @abstractmethod
    def get_score_thresholds(self) -> dict[str, float]:
        '''
        Usage:
            This is an abstract method that needs to be implemented by the child class.

        Returns:
            A dictionary of score thresholds for each score type.
            The score threshold is a float number according to the score range of the prompt.
            If the score is lower than the threshold, the input will be filtered out.
            If the threshold is negative, it means that all inputs will be kept.
            The key name of the dictionary should match the key name of the get_score_prompts()'s return dict.

        Example:
            def get_score_thresholds(self) -> dict[str, float]:
                return {'correct': 7, 'natural': 7}
        '''
        pass

    @abstractmethod
    def get_prompt_key_name(self) -> dict[str, list[str]]:
        '''
        Usage:
            This is an abstract method that needs to be implemented by the child class.

        Returns:
            A dictionary of prompt key names for each score type.
            The prompt key name is a list of key names in the input_js that will be used to log the augment info.
            The key name of the dictionary should match the key name of the get_score_prompts()'s return dict.

        Example:
            def get_prompt_key_name(self) -> dict[str, list[str]]:
                return {'correct': ['query'], 'natural': ['input']}
        '''
        pass

    def get_scores(self, response: str):
        '''
        Usage:
            This is a helper function to get the scores from a response of one of the scoring prompt.
            It will check if the response is a valid score list, and return the scores as a list.
            If the response is invalid, it will return -1.

        Parameters:
            :response: the response of one of the scoring prompt.

        Note:
            Valid means that the response contains the same number of items as the self.input_js.
            And each item in the response is a numeric value.

        Returns:
            A list of scores, or -1 if the response is invalid.
        '''
        try:
            if response and response[0] == '[' and response[-1] == ']': 
                if not response[1:-1].strip().replace(' ','').replace('\n','').replace(',', '').isdigit():
                    logger.error("🐞 All items in 'scores' must be numeric values.")
                    return -1
                scores = eval(response)
                if isinstance(scores, list):
                    if len(scores) == len(self.input_js):
                        return scores
                    else:
                        logger.error(f"🐞 The number of scores does not match the number of inputs: {len(scores)} vs {len(self.input_js)}")
                        return -1
            logger.error(f"🐞 Invalid score response: {response}")
            return -1
        except Exception as e:
            logger.exception(f"🐞 Error in get_score function: {e}")
            return -1

    def get_all_scores(self):
        '''
        Usage:
            1. Get the prompts for each score type.
            2. Repeat the scoring process for each score type for self.repeat_time times.
            3. Calculate the average score for each input.

        Returns:
            A dictionary of the average scores for each score type.
        '''
        prompts = self.get_score_prompts()
        tot_scores = {}
        for name, prompt in prompts.items():
            tot_score = np.zeros(len(self.input_js))
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
        '''
        Usage:
            This function is used to add a new query to the query pool.
            If the pool is not full, it will add the query to the pool and return an empty list.
            If the pool is full, it will calculate the scores of all the inputs,
            and return the output_js that satisfy all the score thresholds.

        Parameters:
            :js: the new query to be added to the pool.
            :last: a boolean flag to indicate if this is the last query to be added. 
                If it is, the pool will score all the rest of the inputs and return the output_js.

        Returns:
            A list of output_js that satisfy all the score thresholds.
        '''
        if not last and len(self.input_js) < self.pool_size:
            self.input_js.append(js)
            return []
        else:
            output_js = deepcopy(self.input_js)
            all_scores = self.get_all_scores()
            for name, threshold in self.get_score_thresholds().items():
                prompt_key_names = self.get_prompt_key_name()[name]
                if threshold < 0:
                    scores = [0] * len(self.input_js)
                else:
                    scores = all_scores[name]
                for score, js in zip(scores, self.input_js):
                    if score < threshold:        # filter out the input if its score is lower than threshold
                        prompt_key_values = ' '.join([str(js[key_name]) for key_name in prompt_key_names])
                        logger.warning(f"🥶 The {name} score of user input is too low: {prompt_key_values} => {score}")
                        if js in output_js:    
                            output_js.remove(js)
                        break
            self.input_js = []
            return output_js
