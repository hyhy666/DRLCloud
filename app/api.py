from fastapi import APIRouter
from app.environment import CloudEnvironment
from app.dqn import DQN
import torch

router = APIRouter()

env = CloudEnvironment(num_nodes=3)
model = DQN(state_dim=3, action_dim=3)

# 加载训练好的模型参数
model.load_state_dict(torch.load("app/dqn.pth", map_location=torch.device('cpu')))
model.eval()

@router.get("/allocate")
def allocate_task():
    state = env.reset()
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
