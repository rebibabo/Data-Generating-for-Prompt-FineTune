## 数据增强

场景：根据活动映射表.xlsx生成query + intention的问答对

- 第一步：生成种子数据集

  随机对query列的信息进行采样，采用hash set去掉重复的query组合，使用**example**提示词生成用户输入，如果query组合个数大于1，则还需要使用**relevant**提示词，判断各个query之间的关联度，剔除掉毫不相关的query（相关度小于7）

  ```
  😭 The relevantness of query set is too low: ['送3个月会员（焕新礼）', '领1T超大云空间'] => 6
  😭 The relevantness of query set is too low: ['合成', '免费会员'] => 3
  😭 The relevantness of query set is too low: ['果园', '云手机抽奖'] => 4
  😭 The relevantness of query set is too low: ['月月赢好礼', 'AI新头像'] => 4
  ```

- 第二步：数据清洗

  加载前面生成好的种子数据集，清洗主要包括3步

  - 使用**natural**提示词判断输入指令的自然程度，剔除掉不自然的输入

    ```
    😭 The naturalness of user input is too low: 怎么才能猜谜开红包呢？我想如何邀好友一起攒云朵？ => 5
    😭 The naturalness of user input is too low: 现金活动怎么参与？“月月抽好礼”有什么信息？ => 6
    ```

  - 使用Rouge剔除掉重复的输入提示词，可调整参数有

    - rouge_type: rouge-1, rouge-2, **rouge-l**
    - rouge_metric: f, p, **r**
    - min_rouge_score: [0, 1], default=**0.7**

    ```
    😭 repetitve user input: 怎样才能获取云朵福利？云朵中心提供什么服务？ with score 0.7692
    😭 repetitve user input: 怎么召唤一位相册达人帮我处理照片？ with score 0.8000
    ```

  - 使用**correct**提示词，判断生成的用户输入能否对应意图识别答案，剔除掉错误的输入

    ```
    😭 The correctness of output is too low: 怎样查看最新的彩票开奖信息？另外，月月抽好礼是什么活动，有什么参与方式？ ['云盘欢乐透，月月赢好礼'] => 5
    😭 The correctness of output is too low: 怎么才能学会种树，种什么树比较好呢？ ['小云果园'] => 6
    😭 The correctness of output is too low: 系统怎么才能更有效地帮我工作呢？ ['组团领红包'] => 5
    😭 The correctness of output is too low: 今天的宠粉日有什么福利吗？ ['移动云盘会员日'] => 4
    😭 The correctness of output is too low: 如何参与云手机的抽奖活动？ ['云端看电影'] => 3
    ```

- 第三步：数据增强

  增加用户输入的多样性，生成的每一条输入在插入到数据集前都要进行清洗

  - 使用**lazy**提示词，模拟懒惰的用户输入，剔除掉无关的信息

    ```json
    {
        "input": "怎么参与回馈活动？",
        "output": [
            "用户回馈活动"
        ],
        "original_input": "怎么参与回馈活动，有没有具体的流程或优惠信息？"
    },
    {
        "input": "如何参与现金活动？能不能告诉我关于“月月抽好礼”的情况？",
        "output": [
            "组团领红包",
            "云盘欢乐透，月月赢好礼"
        ],
        "original_input": "如何参与现金活动？这个活动中可能会有什么好礼可以抽奖吗？我想了解一下关于“月月抽好礼”的信息。"
    },
    {
        "input": "怎样才能更好地玩转公众号？",
       	"output": [
            "玩转公众号"
        ],
        "original_input": "怎样使用公众号的功能？有什么技巧可以让我更好地玩转公众号吗？"
    },
    {
        "input": "如何获取会员专享的内容和3个月会员的信息？",
        "output": [
            "移动云盘会员日",
            "送3个月会员（焕新礼）"
        ],
        "original_input": "如何才能获取会员专享的内容，另外3个月会员是什么呢？"
    }
    ```

  - 使用**implicit**提示词，模拟用户输入含糊的场景，试图识别潜在的意图

    ```json
    {
        "input": "如何能参与每月的活动？顺便想知道有什么奖品推荐。",
        "output": [
            "云盘欢乐透，月月赢好礼"
        ],
        "original_input": "怎么参加月月赢好礼活动？能告诉我有什么礼品吗？"
    },
    {
        "input": "我可以用什么方式更高效地利用公众号的各种功能？",
        "output": [
            "玩转公众号"
        ],
        "original_input": "怎样使用公众号的功能？有什么技巧可以让我更好地玩转公众号吗？"
    },
    {
        "input": "能不能告诉我获取3个月的会员方式和会员能享受到哪些特殊内容？",
        "output": [
            "送3个月会员（焕新礼）",
            "移动云盘会员日"
        ],
        "original_input": "我想知道如何获得3个月的会员资格，还想了解一下会员专享的内容有什么。"
    },
    {
        "input": "我该怎样找到一些能帮我整理照片的高手呢？",
        "output": [
            "召唤相册达人活动"
        ],
        "original_input": "怎么可以召唤相册达人来帮助我处理我的照片呢？"
    },
    ```

    

## 提示词

example提示词

```
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
```

relevant提示词

```
We would like you to evaluate the relevance and interconnectivity between the following intentions.
You should give an overall score on a scale of 1 to 10, where a higher score indicates higher relevance and interconnectivity,
while the lower the score, the less relevant they are.
You must just give a score without any other reasons.
## Intentions: 
{}
## Score:
```

natural提示词

```
We would like you to evaluate the naturalness of the following question.
You should give an overall score on a scale of 1 to 10, where a higher score indicates higher naturalness.
You must just give a score without any other reasons.
## Question:
{}
## Score:
```

correct提示词

```
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
```

lazy提示词

```
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
```

implicit提示词

```
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
```



