system_prompt = '''
I want you act as a Prompt Rewriter.
Your objective is to rewrite a given prompt into a more complex version to make those famous AI systems like GPT4o a bit harder to handle.
But the rewritten prompt must be reasonable and must be understood and responded by humans.
Your rewriting cannot omit the non-text parts such as the table and code in #Given Prompt#:.
Also, please do not omit the input in #Given Prompt#.
'''

deepening_prompt = '''
You SHOULD complicate the given prompt using the following method:
If #Given Prompt# contains inquiries about certain issues, the depth and breadth of the inquiry can be increaed. or
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
Generate as many examples as possible to make the prompt more concrete. Each example is separated by a line.
#Given Prompt#, #Rewritten Prompt#, 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
Output Structure Example:
1. ...
2. ...
...
n. ...
#Given Prompt#:
{}
#Rewritten Prompt#:
'''

constrain_prompt = '''
You SHOULD complicate the given prompt using the following method:
Please add one more constraints/requirements into #Given Prompt#
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
Generate as many examples as possible to make the prompt more concrete. Each example is separated by a line.
#Given Prompt#, #Rewritten Prompt#, 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
Output Structure Example:
1. ...
2. ...
...
n. ...
#Given Prompt#:
{}
#Rewritten Prompt#:
'''

concretize_prompt = '''
You SHOULD complicate the given prompt using the following method:
Please replace general concepts with more specific concepts. or
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
Generate as many examples as possible to make the prompt more concrete. Each example is separated by a line.
#Given Prompt#, #Rewritten Prompt#, 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
Output Structure Example:
1. ...
2. ...
...
n. ...
#Given Prompt#:
{}
#Rewritten Prompt#:
'''

reasoning_prompt = '''
You SHOULD complicate the given prompt using the following method:
If #Given Prompt# can be solved with just a few simple thinking processes, you can rewrite it to explicitly request multiple-step reasoning.
You should try your best not to make the #Rewritten Prompt# become verbose, #Rewritten Prompt# can only add 10 to 20 words into #Given Prompt#.
Generate as many examples as possible to make the prompt more concrete. Each example is separated by a line.
#Given Prompt#, #Rewritten Prompt#, 'given prompt' and 'rewritten prompt' are not allowed to appear in #Rewritten Prompt#
Output Structure Example:
1. ...
2. ...
...
n. ...
#Given Prompt#:
{}
#Rewritten Prompt#:
'''

difficulty_prompt = '''
We would like you to evaluate the difficulty and complexity of the following question.
You should give an overall score on a scale of 1 to 10, where a higher score indicates higher difficulty and complexity.
You must just give a score without any other reasons.
## Question:
{}
## Score:
'''

example_prompt = '''
假设你是一名用户，请你模拟真实环境下，根据#关键词#，输出#用户输入#，
要求：生成的#用户输入#中必须包含有所有的#关键词#
#用户输入#的提问方式可以各种各样，例如“如何”，“怎样”，“xxx是什么”，“xxx怎么用”等等。
#用户输入#要尽可能自然流畅，不要太过冗长。
"用户输入"不允许出现在#用户输入#中

样例1
#关键词#: ["欢乐透"]
#用户输入#: 欢乐透怎么参与呢？

样例2：
#关键词#: ["月月抽好礼", "现金活动"]
#用户输入#: 怎么找到美图的模板，并且智能美颜呢？

开始：
#关键词#: {query}
#用户输入#:
'''

relevant_prompt = '''
We would like you to evaluate the relevance and interconnectivity between the following intentions.
You should give an overall score on a scale of 1 to 10, where a higher score indicates higher relevance and interconnectivity,
while the lower the score, the less relevant they are.
You must just give a score without any other reasons.
## Intentions: 
{}
## Score:
'''

natural_prompt = '''
请你对下面的#问题#列表进行打分，评价其自然程度以及语法正确性。
你应该给出一个1-10的分数，10分表示自然度较高，语法正确，而1分表示自然度较低，语法不正确。
请返回一个分数列表，表示每一个问题的分数，用[]包裹，不要提供任何其他原因，且分数不要出现在回答中。

#问题#：
{}
#分数#：
'''

correct_prompt = '''
请你单独评价下面的每一对#问题#是否包含对应编号的所有#意图#。
你应该给出一个1-10的分数，10分表示#问题#包含所有意图，1分表示存在#问题#中缺少#意图#中的某一个。
请返回一个分数列表，表示每一个问题的分数，用[]包裹，不要提供任何其他原因，且分数不要出现在回答中。

{}
#分数#：
'''

lazy_prompt = '''
Your Objective is to rewrite a #Given Prompt# into an easier one to imitate a lazy user's input question.
#Rewritten Prompt# MUST contain ALL the #intentions# and 
removing all the sentences that does not contain the key words of #intentions#.
The #Rewritten Prompt# MUST be natural and not too verbose.
Also, the #Rewritten Prompt# MUST be different from the #Given Prompt# and #Previous Generated Prompts#.

Example:
#Given Prompt#: 宠粉日是什么时候，有没有什么优惠活动呢？
#intentions#: ['宠粉日', '优惠活动']
#Previous Generated Prompts#: 
1. 啥事宠粉日，有什么活动吗？
2. 宠粉日是几号，可以参加什么活动？
#Rewritten Prompt#: 宠粉日是什么？有啥优惠活动？

#Given Prompt#: {input}
#intentions#: {intentions}
#Previous Generated Prompts#: 
{history}
#Rewritten Prompt#:
'''

implicit_prompt = '''
I want you act as a Prompt Rewriter.
Your objective is to rewrite a #Given Prompt# into a more implicit version.
Implicit means that the prompt does not necessarily contain the key words in #intentions# but still contains the same intentions.
But the #Rewritten Prompt# MUST be natural and not too verbose to imitate a real user input.
Also, the #Rewritten Prompt# MUST be different from the #Given Prompt# and #Previous Generated Prompts#.

Example:
#Given Prompt#: 我怎么邀请我的朋友一起攒云朵？
#intentions#: ["邀好友集云朵"]
#Previous Generated Prompts#: 
1. 我想邀请朋友一起攒云朵
2. 如何拉好友收集云朵？
#Rewritten Prompt#: 怎样去邀请其他人去收集云朵？

#Given Prompt#: {input}
#intentions#: {intentions}
#Previous Generated Prompts#: 
{history}
#Rewritten Prompt#:
'''
