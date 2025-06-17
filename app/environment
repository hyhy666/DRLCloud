# app/environment.py

import random

class CloudEnvironment:
    def __init__(self, num_nodes=3):
        self.num_nodes = num_nodes
        self.reset()

    def reset(self):
        # 初始化每个节点的 CPU 占用率，值域 0 ~ 1
        self.state = [round(random.uniform(0.2, 0.7), 2) for _ in range(self.num_nodes)]
        return self.state

    def step(self, action):
        """
        执行动作：action 是要将任务分配到的节点索引
        返回：新状态、奖励、是否结束
        """
        done = False

        # 奖励为资源利用率越均衡越好
        load_change = [0] * self.num_nodes
        load_change[action] += 0.1  # 假设新任务增加 0.1 负载

        self.state = [min(1.0, s + delta) for s, delta in zip(self.state, load_change)]
        reward = -self._std_dev(self.state)

        return self.state, reward, done

    def _std_dev(self, state):
        avg = sum(state) / len(state)
        return (sum((s - avg) ** 2 for s in state) / len(state)) ** 0.5

