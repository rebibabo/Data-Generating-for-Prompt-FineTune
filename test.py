from my_tool import query

print(query(
'''
We need you to evaluate the correctness of the intention of the question.
You should give an overall score on a scale of 1 to 10.
If the intention is clearly stated in the question, the score should be higher. 
If the intention is unclear or not stated, the score should be much lower.
Only provide the score without any additional explanation.

## Question:
我怎么用我的积分参加天天抽好礼？
## intentions:
菜鸟驿站
## Score:
'''
))