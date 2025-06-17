# data/generate_data.py

import random
import json
import os

os.makedirs("data", exist_ok=True)

samples = []
num_samples = 1000
num_nodes = 3

def std_dev(state):
    avg = sum(state) / len(state)
    return (sum((s - avg) ** 2 for s in state) / len(state)) ** 0.5

for _ in range(num_samples):
    # 初始化节点负载在合理范围
    state = [round(random.uniform(0.1, 0.7), 2) for _ in range(num_nodes)]

    # 动作：分配任务到负载最低的节点或者随机节点
    min_load = min(state)
    candidate_nodes = [i for i, load in enumerate(state) if load == min_load]
    action = random.choice(candidate_nodes)  # 优先选负载最低的节点

    # 负载变化：该节点负载增加0.05~0.15之间的随机数
    load_increase = round(random.uniform(0.05, 0.15), 2)
    load_change = [0.0]*num_nodes
    load_change[action] = load_increase

    next_state = [min(1.0, round(s + delta, 2)) for s, delta in zip(state, load_change)]

    reward = -std_dev(next_state)  # 奖励是负的标准差，越均衡越好

    samples.append({
        "state": state,
        "action": action,
        "next_state": next_state,
        "reward": round(reward, 4)
    })

with open("data/samples.json", "w") as f:
    json.dump(samples, f, indent=2)

print("生成了1000条合理的模拟数据，保存在 data/samples.json")

