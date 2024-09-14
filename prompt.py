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
你是一名aibox的用户，假设你正在和aibox聊天，请你模拟真实环境下，根据你的#意图列表#，输出真实的#用户输入#，
要求：生成的#用户输入#中必须包含有#意图列表#中的所有意图。
#用户输入#的提问方式可以各种各样，例如“如何”，“怎样”，“xxx是什么”，“xxx怎么用”等等。
"用户输入"不允许出现在#用户输入#中

样例1
#意图列表#: 合影
#用户输入#: 怎么在图库中找到之前的一张合影？

样例2：
#意图列表#: 智能美颜，美图收藏馆
#用户输入#: 怎么找到美图的模板，并且智能美颜呢？

开始：
#意图列表#: {query}
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
We would like you to evaluate the naturalness of the following question.
You should give an overall score on a scale of 1 to 10, where a higher score indicates higher naturalness.
You must just give a score without any other reasons.
## Question:
{}
## Score:
'''

correct_prompt = '''
We would like you to evaluate the correctness of the answer to the question, the answer is about the intentions of the question.
You should give an overall score on a scale of 1 to 10, where a higher score indicates higher correctness.
If the question contains the keywords of the intentions, the score will be 10.
You must just give a score without any other reasons.

Example:
## Question: 
如何申请免费的云盘会员？
## Answer: 
['云盘会员']
## Score: 10

## Question: 
有没有什么办法可以吸引更多志愿者来参加社区活动？
## Answer: 
["组团领红包"]
## Score: 3

## Question:
{}
## Answer:
{}
## Score:
'''

lazy_prompt = '''
I want you act as a Prompt Rewriter.
Your Objective is to rewrite a #Given Prompt# into a more easier version to make it look more like a real conversation.
But the #Rewritten Prompt# MUST contain the #intentions# of the original prompt and 
removing all the sentences that does not contain the key words of #intentions#.
The #Rewritten Prompt# MUST be natural and not too verbose.

Example:
#Given Prompt#: 我能不能申请送3个月会员（焕新礼）？另外，图库里怎么找到合影，能否使用智能美颜功能进行修图，然后把图片保存到美图收藏馆？
#intentions#: ["送3个月会员（焕新礼）"]
#Rewritten Prompt#: 怎么申请送3个月会员（焕新礼）？

#Given Prompt#: 宠粉日是什么时候，有没有什么优惠活动呢？我能否使用移动云盘的会员日特惠？
#intentions#: ['宠粉日']
#Rewritten Prompt#: 宠粉日是什么时候？

#Given Prompt#: {}
#intentions#: {}
#Rewritten Prompt#:
'''

implicit_prompt = '''
I want you act as a Prompt Rewriter.
Your objective is to rewrite a #Given Prompt# into a more implicit version.
Implicit means that the prompt does not necessarily contain the key words in #intentions# but still contains the same #intentions#.
But the #Rewritten Prompt# MUST be natural and not too verbose to imitate a real user input.

Example:
#Given Prompt#: 我怎么邀请我的朋友一起攒云朵？
#intentions#: ["邀好友集云朵"]
#Rewritten Prompt#: 怎样去邀请其他人去收集云朵？

#Given Prompt#: 怎么参与相册达人活动？
#intentions#: ["相册达人"]
#Rewritten Prompt#: 有没有一个比比看谁的相册有更多美景的活动？

#Given Prompt#: {}
#intentions#: {}
#Rewritten Prompt#:
'''
