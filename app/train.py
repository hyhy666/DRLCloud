import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from app.environment import CloudEnvironment
from app.dqn import DQN, ReplayBuffer

# 超参数
num_episodes = 1000
batch_size = 64
gamma = 0.99
learning_rate = 0.001
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 0.999
target_update_freq = 10

# 初始化环境和模型
env = CloudEnvironment(num_nodes=3)
state_dim = env.num_nodes
action_dim = env.num_nodes

policy_net = DQN(state_dim, action_dim)
target_net = DQN(state_dim, action_dim)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
replay_buffer = ReplayBuffer(10000)

epsilon = epsilon_start

def select_action(state, epsilon):
    if random.random() < epsilon:
        return random.randint(0, action_dim - 1)
    else:
        state_tensor = torch.tensor([state], dtype=torch.float32)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
            return torch.argmax(q_values).item()

for episode in range(num_episodes):
    state = env.reset()
    total_reward = 0

    for t in range(100):  # 最大步数
        action = select_action(state, epsilon)
        next_state, reward, done = env.step(action)
        replay_buffer.push(state, action, reward, next_state)
        state = next_state
        total_reward += reward

        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states = replay_buffer.sample(batch_size)

            # 计算Q目标值
            q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
            with torch.no_grad():
                max_next_q_values = target_net(next_states).max(1)[0]
                expected_q_values = rewards + gamma * max_next_q_values

            loss = nn.MSELoss()(q_values, expected_q_values)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(epsilon_end, epsilon * epsilon_decay)

    if episode % target_update_freq == 0:
        target_net.load_state_dict(policy_net.state_dict())

    if episode % 50 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

# 保存训练好的模型
torch.save(policy_net.state_dict(), "dqn.pth")
print("Training complete and model saved as dqn.pth")
