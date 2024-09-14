from my_tool import query

print(query(
'''
Evaluate if the #intention# is a clear intent within the #question#.
You should give an overall score on a scale of 1 to 10.
10 indicates a clear and direct intent, and 1 indicates the intent is not present or unclear.
Only provide the score without any additional explanation.

#question#:
我怎么用我的积分参加天天抽好礼？
#intention#:
首充礼
#score#:
'''
))