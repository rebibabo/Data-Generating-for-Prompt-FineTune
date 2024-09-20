## Data Augmentation Pipeline Introduction

![image](https://github.com/user-attachments/assets/98830d3f-6440-476a-94cf-8d7e77963208)

Scenario: Generate question + intention pairs based on the activity mapping table.xlsx

See the code in the "original" directory

- Step 1: Generate Seed Dataset

  Randomly sample information from the query column, use a hash set to remove duplicate query combinations, and use the **example** prompt to generate user input. If the number of query combinations is greater than 1, the **relevant** prompt is also needed to judge the relevance between queries and remove unrelated queries (relevance less than 7).

  ```
  ☠️ The relevantness of query set is too low: ['3-month membership (renewal gift)', 'Claim 1T extra cloud space'] => 6
  ☠️ The relevantness of query set is too low: ['Synthesis', 'Free membership'] => 3
  ☠️ The relevantness of query set is too low: ['Orchard', 'Cloud phone lottery'] => 4
  ☠️ The relevantness of query set is too low: ['Win gifts every month', 'AI new avatar'] => 4
  ```

- Step 2: Data Cleaning

  Load the previously generated seed dataset, and cleaning mainly includes 3 steps:

  - Use the **natural** prompt to judge the naturalness of the input command, and remove unnatural inputs.

    ```
    🥵 The naturalness of user input is too low: How can I guess riddles to open red packets? How can I invite friends to accumulate clouds together? => 5
    🥵 The naturalness of user input is too low: How to participate in the cash activity? What information does "Draw gifts every month" have? => 6
    ```

  - Use Rouge to remove duplicate input prompts, adjustable parameters include

    - rouge_type: rouge-1, rouge-2, **rouge-l**
    - rouge_metric: f, p, **r**
    - min_rouge_score: [0, 1], default=**0.7**

    ```
    🤢 Repetitive user input: How can I get cloud benefits? What services does the Cloud Center provide? with score 0.7692
    🤢 Repetitive user input: How can I summon a photo album master to help me process photos? with score 0.8000
    ```

  - Use the **correct** prompt to judge whether the generated user input can correspond to the intention recognition answer, and remove incorrect inputs.

    ```
    🥶 The correctness of output is too low: How can I check the latest lottery results? Also, what is the "Draw gifts every month" activity, and what are the ways to participate? ['Cloud disk fun transparency, win gifts every month'] => 5
    🥶 The correctness of output is too low: How can I learn to plant trees, and what kind of trees are better to plant? ['Small cloud orchard'] => 6
    🥶 The correctness of output is too low: How can the system help me work more effectively? ['Group to get red packets'] => 5
    🥶 The correctness of output is too low: What are the benefits of today's pet fan day? ['Mobile cloud disk membership day'] => 4
    🥶 The correctness of output is too low: How to participate in the cloud phone lottery activity? ['Watch movies in the cloud'] => 3
    ```

- Step 3: Data Augmentation

  Increase the diversity of user input, and each input generated before being inserted into the dataset must be cleaned

  - Use the **lazy** prompt to simulate lazy user input, and remove irrelevant information

    ```json
    {
        "input": "How to participate in the feedback activity?",
        "output": [
            "User feedback activity"
        ],
        "original_input": "How to participate in the feedback activity, is there a specific process or discount information?"
    },
    {
        "input": "How to participate in the cash activity? Can you tell me about the 'Draw gifts every month' situation?",
        "output": [
            "Group to get red packets",
            "Cloud disk fun transparency, win gifts every month"
        ],
        "original_input": "How to participate in the cash activity? What good gifts might there be in this activity for a lottery? I want to know about the 'Draw gifts every month' information."
    },
    ```

  - Use the **implicit** prompt to simulate scenarios where user input is vague, trying to identify potential intentions

    ```json
    {
        "input": "How can I participate in monthly activities? By the way, what are the recommended prizes?",
        "output": [
            "Cloud disk fun transparency, win gifts every month"
        ],
        "original_input": "How to participate in the 'Draw gifts every month' activity? Can you tell me what gifts are available?"
    },
    {
        "input": "What ways can I use to more efficiently use the various functions of the public account?",
        "output": [
            "Play with the public account"
        ],
        "original_input": "How to use the functions of the public account? Are there any tips to make me better at playing with the public account?"
    },
    ```
- step 4: Prompt Fine-Tuning

    Merge the augmented dataset into the train dataset, and use the cleaned seed dataset as the validation set.

    For each iterantion, we fine-tune the prompt model using the train dataset from scratch, and use the validation set to evaluate the performance of the prompt model.

    Then, we pick up the wrong preditions from the validation set and augment them, and append them to the train dataset for the next iteration.

    If the increase of the performance between the previous and current iteration is not significant, we stop the augmentation process.

    Lastly, we merge the train dataset and the validation set into the final dataset, and use it for training the ultimate model.

## Introduction to the project
We propose this pipeline in the "abstract" directory, which contains 5 files:
- dataAug.py

    We define the DataAugmentation class, which can clean the seed dataset according to the prompt rules and the similarity betwwen each query, and generate the augmented dataset.

- queryPool.py

    We define the QueryPool class, which is an abstract class for query pool. It is used to score a batch(pool_size) of user inputs in an AI-powered way.
    When the pool is full, it will calculate the scores of all the inputs, and return the output_js that satisfy all the score thresholds.
    
    There are 3 abstract methods that need to be implemented:
    - get_score_prompts
    - get_score_thresholds
    - get_prompt_key_name

    See more details in the file.

- finetune.py

    We define the FineTune class, which is used to fine-tune the prompt model using the train dataset from scratch, and use the validation set to evaluate the performance of the prompt model.
    During each iteration , it will augment the wrong predictions using a list of augmentation functions.
    Finally, Saves the best model based on a metric.

- evaluate.py

    We define the Evaluate class, which is an abstract class for evaluation. It is used to evaluate the performance of the prompt model.
    There are 3 abstract methods that need to be implemented:
    - forward
    - metric
    - is_wrong
    See more details in the file.

We implement the pipeline in the "example" directory, see more details in the files.

## Experiments

### The effect of the number of iterations

We fine-tune the prompt model using the 1014 train dataset from scratch, and use the 175 clean seed dataset to evaluate the performance of the prompt model.

The model should output the intention of the query using a string list, and we evaluate the performance of the model using the precision, recall, and F1-score metrics.

We fine-tune the prompt model using the default parameters and evaluate the performance of the model for different number of iterations.

The results show that the prompt model achieves a high level of performance after 160 iterations.

![image](https://github.com/user-attachments/assets/3ef7ed08-a34a-48d8-8f0a-c27e6a3fd2de)

![image](https://github.com/user-attachments/assets/b80e7f49-2d92-45bc-bb2e-ed649a72817c)

