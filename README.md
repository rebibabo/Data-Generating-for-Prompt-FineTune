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

    We define the DataAugmentation class, which can clean the seed dataset according to the prompt rules and the similarity between each query, and generate the augmented dataset.

- queryPool.py

    We define the QueryPool class, which is an abstract class for query pool. It is used to score a batch(pool_size) of user inputs in an AI-powered way.
    When the pool is full, it will calculate the scores of all the inputs, and return the output_js that satisfy all the score thresholds.
    
    There are 3 abstract methods that need to be implemented:
    - get_score_prompts
    - get_score_thresholds
    - get_prompt_key_name

    See more details in the file.

- finetune.pybe
be
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

We implement the pipeline in the "example" directory

See more details in the files.

## Experiments

### The effect of the number of iterations

We fine-tune the prompt model using the 1014 train dataset from scratch, and use the 175 clean seed dataset to evaluate the performance of the prompt model.

The model should output the intention of the query using a string list, and we evaluate the performance of the model using the precision, recall, and F1-score metrics.

We fine-tune the prompt model using the default parameters and evaluate the performance of the model for different number of iterations.

The results show that the prompt model achieves a high level of performance after 160 iterations.

![image](https://github.com/user-attachments/assets/3ef7ed08-a34a-48d8-8f0a-c27e6a3fd2de)

### The rationale behind the LoRA

The formula of the LoRA (Low-Rank Adaptation) is:

$$W = W_0 + \alpha \cdot A \times B$$

where $$W_0$$ is the original model weights, $$A$$ is the low-rank matrix, and $$B$$ is the prompt matrix.

The shape of $$A$$ and $$B$$ are determined by the rank parameter. A: $$m \times r$$, B: $$r \times n$$, W: $$m \times n$$.

The PEFT model will lock the weights of the original model weight $$W_0$$, and only update the matrix $$A$$ and $$B$$.

This can substantially reduce the computational cost of training the model, as only a small number of parameters need to be updated.

### The effect of the rank

A higher rank allows for more fine-grained adjustments but may approach the computational cost of training the entire model. A lower rank reduces the number of parameters but may not adapt as effectively to new tasks.

It can be observed that the model performance is optimal when the rank reaches around 64. If the rank is increased further, overfitting occurs, leading to a decline in model performance. Therefore, it is considered to use a rank of 64 subsequently.

It is noticable that the impact of increasing the rank on training time is subtle.

![image](https://github.com/user-attachments/assets/b80e7f49-2d92-45bc-bb2e-ed649a72817c)

### The effect of the alpha

Alpha often refers to a hyperparameter that controls the scale or magnitude of the low-rank adaptation being applied to the model.

A higher alpha allows for more fine-grained adjustments. A lower alpha may not adapt as effectively to new tasks.

We can see that the model performance is optimal when the alpha is around 1.

![image](https://github.com/user-attachments/assets/02a5c65a-aa9f-40ad-9a3e-c2ef41a9edbb)


### The performance of different models

We choose 5 models to evaluate their performance on the task of intention recognition. We find that the performance of different models is highly sensitive to the instruction prompt, thus we rewrite the instruction prompt for each model, the other parameters are kept the same. That is 160 small iterations each time, r = 64 and alpha = 1. The length of the training dataset is 1014.

We found that the best model is mistral-7b, with a f1-score of 0.972, signifantly higher than the other models. The worst model is llama, with a f1-score of 0.748, which results from the low understanding of the task, the response of the model is invalid frequently.
So we can't parse the response to the list of intentions, which is the key to the fail of the model.

![image](https://github.com/user-attachments/assets/f6fa09a4-f42b-471a-b662-c1d9ce8271f2)

We also evaluate the number of wrong predictions and invalid predictions of each model, it is obvious that there are the most invalid predictions of the Llama model, which is 31/175, mistral, phi and Qwen are the best models, which follow the prompt rules strictly.

Although all the predictions from phi are valid, but due to the small number of parameters, the model can't learn the correct mapping between the prompt and the intention effectively, so its wrong predictions are most frequent.

The best model is mistral-7b, which has only 22 wrong predictions and 0 invalid predictions among 175 validate dataset.

![image](https://github.com/user-attachments/assets/9b385123-1954-4c25-bf0d-834452cce451)

We also record the training and inference time of each model, most the models can finish the training around 10 minutes, the fastest model is qwen2.5, each inference takes less than 1 second.

![image](https://github.com/user-attachments/assets/b34e22eb-23c9-48e3-9717-43cd70043ec4)


### The performance of data augmentation

We fine-tune the Qwen2.5 model to varify the effectiveness of data augmentation.

The model is fine-tuned using the 1014 train dataset from scratch, and use the 175 clean seed dataset to evaluate the performance of the model.

Each augmentation iteration will generate a new set of augmented data from the wrong predictions, and append it to the train dataset, then fine-tune the model using the new train dataset from scratch.

After 1 augmentation iteration, the model achieves a higher level of performance, with a f1-score from 0.875 to 0.907.
But after 2 augmentation iterations, the model performance decreases, with a f1-score from 0.907 to 0.823. And we save the best model based on the f1-score, that is from iteration 2.

![image](https://github.com/user-attachments/assets/85ac08d1-ee92-4906-b0cb-2ad231997048)
