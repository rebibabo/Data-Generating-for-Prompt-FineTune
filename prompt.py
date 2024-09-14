non_annotated_prompt = '''
Task: Extractive Question Answering
Objective: Generate as many distinct #question# and #answer# pairs as possible from each #paragraph# provided, which are related to the coding style. Each #question# should differ from the others and cover various aspects of the #paragraph#. 
Important: If the #paragraph# contains code snippets, ensure to generate questions and answers related to ALL the correct and incorrect usage of the code.

An example is as followed:
#paragraph#: 
When trailing commas are redundant, they are often helpful when a version control system is used, when a list of values, arguments or imported items is expected to be extended over time. The pattern is to put each value (etc.) on a line by itself, always adding a trailing comma, and add the close parenthesis/bracket/brace on the next line. However it does not make sense to have a trailing comma on the same line as the closing delimiter (except in the above case of singleton tuples):

    # Correct:
    FILES = [
        'setup.cfg',
        'tox.ini',
        ]
    initialize(FILES,
               error=True,
               )
​    
    # Wrong:
    FILES = ['setup.cfg', 'tox.ini',]
    initialize(FILES, error=True,)

#question#: When can trailing commas be helpful in a codebase?
#answer#: Trailing commas can be helpful when using a version control system or when a list of values, arguments, or imported items is expected to be extended over time.

#question#: What is the general pattern for using trailing commas when dealing with lists or arguments?
#answer#: The pattern is to place each value on a separate line, always include a trailing comma, and place the closing parenthesis, bracket, or brace on the next line.

#question#: Provide an example of correct usage of trailing commas in a list assignment.
#answer#: 
```
FILES = [
    'setup.cfg',
    'tox.ini',
]
```

#question#: What is the correct way to format the following wrong initialization function call with trailing commas?
```
initialize(FILES, error=True,)
```
#answer#: 
```
initialize(FILES,
           error=True,
           )
```

...

Now, you are to generate as many QA as you can.
#paragraph#:
{paragraph}
'''

controdictary_prompt = '''
We would like you to evaluate whether the following two sentences express directly opposite requirements.
You should give an overall score on a scale of 1 to 100, where a higher score indicates that the sentences have opposite meanings or requirements, and a lower score indicates that the sentences are unrelated or not opposites.
You must just give a score without any other reasons.

#Question 1#:
{q1}

#Question 2#:
{q2}

#Score#:
'''
