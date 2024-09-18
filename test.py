from tqdm import tqdm
import time
 
# 创建一个范围为10的进度条
for i in tqdm(range(10)):
    # 在每个迭代周期内使用tqdm.write()输出
    tqdm.write(f"Processing item {i+1}")
 
    # 模拟一些处理时间
    time.sleep(1)