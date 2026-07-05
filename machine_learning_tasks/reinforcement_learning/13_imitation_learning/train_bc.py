"""
任务 13：模仿学习（Imitation Learning）
代表模型：行为克隆（Behavior Cloning，手写策略网络）
损失函数：交叉熵损失（离散动作）
使用合成状态-动作对训练策略网络。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 超参数
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
STATE_DIM = 10
ACTION_DIM = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyNetwork(nn.Module):
    """手写策略网络。"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_synthetic_dataset(num_samples=5000):
    """生成专家轨迹数据：state -> expert action。"""
    states = torch.randn(num_samples, STATE_DIM)
    # 简单专家策略：根据状态的加权和选择动作
    weights = torch.randn(STATE_DIM)
    scores = states @ weights
    expert_actions = torch.remainder((scores * 100).long().abs(), ACTION_DIM)
    return TensorDataset(states, expert_actions)


def main():
    dataset = get_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    policy = PolicyNetwork(STATE_DIM, ACTION_DIM).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

    policy.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        correct = 0
        total = 0
        for states, actions in loader:
            states = states.to(DEVICE)
            actions = actions.to(DEVICE)

            optimizer.zero_grad()
            logits = policy(states)
            loss = criterion(logits, actions)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            correct += (logits.argmax(dim=1) == actions).sum().item()
            total += actions.size(0)

        avg_loss = total_loss / len(loader)
        acc = correct / total
        print(
            f"Epoch [{epoch + 1}/{EPOCHS}]  Loss: {avg_loss:.4f}  Acc: {acc:.4f}"
        )


if __name__ == "__main__":
    main()
