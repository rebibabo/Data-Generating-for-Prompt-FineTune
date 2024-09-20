import json
from rouge_chinese import Rouge
import jieba
from typing import Literal, Any, Callable
from utils import Log, query, logger
from tqdm import tqdm
import os

class DataAugmentation:
    ''' This is a class for data augmentation. '''
    rouge = Rouge()
    
    def __init__(self, key_name: str = 'input'):
        ''' 
        Parameters:
            :key_name: the key name of the input in the dataset. Each data is a dict with key_name as the input. 
        '''
        self.dataset = []           # list of dict
        self.references = []        # list of str that has been tokenized
        self.key_name = key_name    

    @ staticmethod
    def from_file(file_path: str, key_name: str='input', ref: bool = True) -> 'DataAugmentation':
        '''
        Usage:
            This is a factory method to create a DataAugmentation object from a file. 
        
        Parameters:
            :file_path: the path of the file. 
            :key_name: the key name of the input in the dataset. 
            :ref: whether to load the reference of each input from the file. 

        Return:
            :dataAug: the DataAugmentation object
        '''
        dataAug = DataAugmentation()
        if not os.path.exists(file_path):
            logger.error(f"🐞 File not found: {file_path}")
            return None
        
        if file_path.endswith('.json'):
            with open(file_path, 'r', encoding='utf-8') as f:
                dataAug.dataset = json.load(f)
        elif file_path.endswith('.jsonl'):
            with open(file_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in tqdm(lines, desc='Loading dataset'):
                    js = json.loads(line)
                    dataAug.dataset.append(js)
        else:
            logger.error(f"🐞only support json or jsonl")
            return None

        if ref:
            for js in dataAug.dataset:
                if key_name not in js:
                    logger.error(f"🐞 key_name {key_name} not found in {file_path}")
                    return None
                reference = ' '.join(jieba.cut(js[key_name]))
                dataAug.references.append(reference)

        return dataAug

    @ staticmethod
    def from_dataset(dataset: list[dict], key_name: str='input', ref: bool = True) -> 'DataAugmentation':
        '''
        Usage:
            This is a factory method to create a DataAugmentation object from a dataset. 
        
        Parameters:
            :dataset: the dataset, a list of dict. 
            :key_name: the key name of the input in the dataset. 
            :ref: whether to load the reference of each input from the dataset. 

        Return:
            :dataAug: the DataAugmentation object
        '''
        dataAug = DataAugmentation()
        dataAug.dataset = dataset
        if ref:
            for js in dataset:
                if key_name not in js:
                    logger.error(f"🐞 key_name {key_name} not found in dataset")
                    return None
                reference = ' '.join(jieba.cut(js[key_name]))
                dataAug.references.append(reference)
        return dataAug

    def _insert(self, 
        js: dict,             
        pool: Any,            
        last: bool = False,     
        rouge_type: Literal['rouge-1', 'rouge-2', 'rouge-l'] = 'rouge-l',      
        rouge_metric: Literal['f', 'p', 'r'] ='r',    
        min_rouge_score: float = 0.7,           
        max_length: int = 100,          
    ) -> list[dict]:
        '''
        Usage:
            Given a input data, a pool of query, and some parameters, 
            insert the input data to the pool and get the batch output data that satisfy the pool's condition.

        Parameters:
            :js: the input data(dict)
            :pool: the pool of query which is a subclass of QueryPool
            :last: whether it is the last query in the pool, if so, pool will submit the rest data to score
            :rouge_type: the type of rouge metric to use
            :rouge_metric: the metric to use in rouge score, f for f1, p for precision, r for recall
            :min_rouge_score: the minimum rouge score , representing the threshold of the similarity between the input and the reference
            :max_length: the maximum length of the input

        Return:
            :output_js: the batch output data that satisfy the pool's condition
        '''
        user_input = js[self.key_name]

        if len(user_input) > max_length:
            logger.warning(f"🤮 The length of user input is too long")
            if not last:
                return []

        hypothesis = ' '.join(jieba.cut(user_input))
        if min_rouge_score > 0:
            for reference in self.references:
                scores = self.rouge.get_scores(hypothesis, reference)
                score = scores[0][rouge_type][rouge_metric]
                if score > min_rouge_score:     # detect the repetitive input
                    logger.warning(f"🤢 repetitve user input: {user_input} => {score:.4f}")
                    if not last:
                        return []

        self.references.append(hypothesis)
        output_js = pool.add_query(js, last=last)       # add the query to the pool and get the batch output data that satisfy the pool's condition
        self.dataset.extend(output_js)
        for js in output_js:
            logger.success(f"🎉 Successfully add the user input: {js[self.key_name]}")
        return output_js

    def cleanse(self, 
        pool: Any,           
        save_path: str = '',  
        **kwargs              
    ) -> list[dict]:
        '''
        Usage:
            Given a pool that contains the condition to filter the data, and scoring data(pool_size) in a time
            Save the cleaned dataset, if not provided, the cleaned dataset will not be saved.

        Parameters:
            :pool: the pool of query which is a subclass of QueryPool
            :save_path: the path to save the cleaned dataset
            :kwargs: other arguments for the _insert method

        Return:
            :cleaned_dataset: the cleaned dataset
        '''
        dataset = self.dataset.copy()
        length = len(dataset)
        self.dataset = []
        for i, js in tqdm(enumerate(dataset), total=length):
            last = (i == length - 1)
            self._insert(js, pool, last=last, **kwargs)
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(self.dataset, f, ensure_ascii=False, indent=4)
        return self.dataset

    def augment(self, 
        pool: Any,                  
        prompt_func: Callable,      
        output_path: str,    
        repeat_num: int = 3,
        from_log: bool = True, 
        indent: int = None,   
        **kwargs
    ) -> list[dict]:
        '''
        Usage:
            Given a pool that contains the condition to filter the data, and a prompt_func to generate the prompt for each input,
            Output the augmented dataset to output_path.

        Parameters:
            :pool: the pool of query which is a subclass of QueryPool
            :prompt_func: the function to generate the prompt for each input
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

                Note:
                    The response of the prompt_func should be the augmented input.

                Prompt Example:
                    I want you act as a Prompt Rewriter.
                    Your objective is to rewrite a #Given Prompt# into a more implicit version.
                    Implicit means that the prompt does not necessarily contain the key words in #intentions# but still contains the same intentions.
                    But the #Rewritten Prompt# MUST be natural and not too verbose to imitate a real user input.
                    Also, the #Rewritten Prompt# MUST be different from the #Given Prompt# and #Previous Generated Prompts#.

                    Example:
                    #Given Prompt#: How to improve my English?
                    #intentions#: ["improving English"]
                    #Previous Generated Prompts#: 
                    1. I'd like to improve my spoken English as I am not very good at it.
                    2. I wish to find an English teacher who can help me improve my English.
                    #Rewritten Prompt#: What's the key to enhance my English?

                    #Given Prompt#: {input}
                    #intentions#: {intentions}
                    #Previous Generated Prompts#: 
                    {history}
                    #Rewritten Prompt#: {prompt}
            :output_path: the path to save the augmented dataset
            :repeat_num: the number of times to repeat the augmentation for each input
            :from_log: whether to start from the last index in the log file or from the beginning
            :indent: the indent of the json file

        Return:
            :augment_dataset: the augmented dataset

            Record the index of the augmented dataset in the augment.log file under the same directory as output_path.
                Example:
                    {"filename": "train.jsonl", "idx": 110}
                Usage:
                    If there is an interruption in the augmentation process, we can start from the last index in the log file.

            Record the augment infomation in the log/{time}.log file according to the utils.py.
                Example:
                    SUCCESS | 🎉 Successfully add the user input: How to improve my English?
                    WARNING | 🥶 The correct score of user input is too low: How to improve my English? ['Franch'] => 3
                    WARNING | 🥶 The natural score of user input is too low: How improve English? => 5
                    WARNING | 🤢 repetitve user input: How to improve my English? => 0.8534
                Usage:
                    We can check the log file to see the details of the augmentation process.
        '''
        log = Log(output_path)
        last_idx = log.last_idx if from_log else 0
        f = open(output_path, 'a', encoding='utf-8')
        augment_dataset = []
        for i, js in tqdm(enumerate(self.dataset[last_idx:]), 
                            total=len(self.dataset)-last_idx,
                            desc=f'Augmenting by {prompt_func.__name__}'):
            history = []
            for j in range(repeat_num):
                prompt = prompt_func(js, history)
                aug_input = query(prompt)
                js[self.key_name] = aug_input
                last = (i == len(self.dataset) - 1 and j == repeat_num - 1)
                output_js = self._insert(js.copy(), pool, last=last, **kwargs)
                if output_js:
                    for js in output_js:
                        augment_dataset.append(js)
                        f.write(json.dumps(js, ensure_ascii=False, indent=indent) + '\n')
                history.append(aug_input)
            log.update(i+last_idx)
        f.close()
        return augment_dataset