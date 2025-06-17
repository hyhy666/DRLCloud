# app/api.py

from fastapi import APIRouter
from app.environment import CloudEnvironment
from app.dqn import DQN
import torch

router = APIRouter()

# 初始化环境与模型（简单示例：每次请求都重建）
env = CloudEnvironment(num_nodes=3)
model = DQN(state_dim=3, action_dim=3)
model.eval()  # 推理模式（不训练）

@router.get("/allocate")
def allocate_task():
    state = env.reset()  # 获取初始状态
    state_tensor = torch.tensor([state], dtype=torch.float32)
    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values, dim=1).item()

    next_state, reward, done = env.step(action)

    return {
        "current_state": state,
        "chosen_action": action,
        "next_state": next_state,
        "reward": reward
    }

