"""
任务 2：回归（Regression）
代表模型：MLP（多层感知机，手写模型结构）
损失函数：均方误差（MSE）
使用合成数据演示从输入特征预测连续值。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# 超参数
BATCH_SIZE = 64
EPOCHS = 10
LR = 1e-3
INPUT_DIM = 10
OUTPUT_DIM = 1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """手写多层感知机"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        ''' 3 个全连接层 '''
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def get_synthetic_dataset(num_samples=2000):
    """构造 y = X @ w + b + 噪声 的回归数据"""
    x = torch.randn(num_samples, INPUT_DIM) # 输入
    true_w = torch.randn(INPUT_DIM, OUTPUT_DIM) # 权重
    true_b = torch.randn(OUTPUT_DIM) # 偏置
    y = x @ true_w + true_b + 0.1 * torch.randn(num_samples, OUTPUT_DIM) # 加上一点噪声
    return TensorDataset(x, y)


def main():
    train_loader = DataLoader(
        get_synthetic_dataset(), batch_size=BATCH_SIZE, shuffle=True
    )

    model = MLP(INPUT_DIM, 64, OUTPUT_DIM).to(DEVICE)
    criterion = nn.MSELoss() # 损失函数: 均方误差
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  MSE Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
