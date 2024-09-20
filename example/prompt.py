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
#Rewritten Prompt# should be written in Chinese.

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
#Rewritten Prompt# should be written in Chinese.

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

alpaca_prompt = (
    "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."
    "### Instruction:"
    "{}"
    "### Input:"
    "{}"
    "### Response:"
    "{}"
)
