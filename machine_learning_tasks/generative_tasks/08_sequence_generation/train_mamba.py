"""
任务 8：序列生成 (Sequence Generation)
名称：状态空间模型语言建模 (State Space Model Language Modeling)
领域分类：自然语言处理 / 序列生成
代表架构/算法: Mamba (Linear-Time Sequence Modeling with Selective State Spaces), Gu et al., 2023
核心思想与机制: 结合了循环神经网络（RNN）的高效推理与Transformer的表达能力，通过引入选择性状态空间机制（Selective SSM），使状态空间模型的参数能够根据输入动态调整，从而实现长序列的高效建模。
数学公式/目标函数: 
    - 连续系统: h'(t) = A h(t) + B x(t), y(t) = C h(t) + D x(t)
    - 离散化: h_t = A_bar * h_{t-1} + B_bar * x_t
    - 损失函数: L = - sum(log P(w_t | w_{<t}))
数据输入规范:
    - 输入 (input_ids): 形状 [B, Seq_Len]，类型为 LongTensor，物理含义为词表索引。
    - 输出 (logits): 形状 [B, Seq_Len, Vocab_Size]，类型为 FloatTensor，物理含义为词表上的未归一化对数概率。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

class SimpleMambaBlock(nn.Module):
    """
    简易选择性 Mamba 核心模块，用于处理一维序列特征。

    数学原理 / 变换逻辑:
        1. 门控分流: xz = Linear(x), 分割为 x_proj_in 和 z。
        2. 局部感知: x_conv = SiLU(Conv1D(x_proj_in))。
        3. 选择性投影: 动态生成步长 delta 及状态矩阵参数 B, C。
        4. SSM 核心循环: 离散化状态转移并递归更新隐状态 h，得到 y。
        5. 门控与输出: y = y * SiLU(z)，随后通过线性层投影回 d_model。

    Args:
        d_model (int): 输入特征维度。
        d_state (int): 状态空间隐状态维度，默认 16。
        expand (float): 内部维度扩展系数，默认 2.0。

    Inputs:
        x (Tensor): 输入张量，shape: [B, L, d_model]

    Outputs:
        out (Tensor): 输出张量，shape: [B, L, d_model]
    """
    def __init__(self, d_model, d_state=16, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_inner = int(expand * d_model)
        self.d_state = d_state

        # 输入投影：将输入扩展为 2 倍内部维度（用于分支门控和 SSM）
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D 因果卷积层（模拟本地上下文捕获）
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            kernel_size=4,
            padding=3,
            groups=self.d_inner
        )

        # 选择性参数投影：将 x 映射生成 delta (步长), B, C
        self.x_proj = nn.Linear(self.d_inner, self.d_inner + d_state * 2, bias=False)
        
        # 持续性参数 A (初始化为负数，保证状态衰减稳定)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        
        # D 跃迁参数
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # 输出投影
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def forward(self, x):
        """
        前向传播方法。
        
        Inputs:
            x (Tensor): 形状为 [B, Seq_Len, d_model]
            
        Outputs:
            out (Tensor): 形状为 [B, Seq_Len, d_model]
        """
        # x 形状: (Batch, Seq_Len, d_model)
        batch_size, seq_len, _ = x.shape
        
        # 1. 投影与门控分流
        xz = self.in_proj(x) # [B, L, 2 * d_inner]
        x_proj_in, z = xz.chunk(2, dim=-1) # 各自为 [B, L, d_inner]

        # 2. 1D 因果卷积
        x_conv = x_proj_in.transpose(1, 2) # [B, d_inner, L]
        x_conv = self.conv1d(x_conv)[:, :, :seq_len] # 截断保持因果，[B, d_inner, L]
        x_conv = x_conv.transpose(1, 2) # [B, L, d_inner]
        x_conv = F.silu(x_conv) # [B, L, d_inner]

        # 3. 选择性参数生成 (delta, B, C)
        x_dbl = self.x_proj(x_conv) # [B, L, d_inner + 2 * d_state]
        delta, B, C = torch.split(
            x_dbl, 
            [self.d_inner, self.d_state, self.d_state], 
            dim=-1
        ) # delta: [B, L, d_inner], B: [B, L, d_state], C: [B, L, d_state]
        
        delta = F.softplus(delta) # 确保步长为正数，[B, L, d_inner]
        A = -torch.exp(self.A_log) # [d_inner, d_state] 负实数保证稳定

        # 4. 选择性状态空间核心扫描 (Pure PyTorch 循环实现)
        y = torch.zeros_like(x_conv) # [B, L, d_inner]
        h = torch.zeros(batch_size, self.d_inner, self.d_state, device=x.device, dtype=x.dtype) # [B, d_inner, d_state]
        
        for t in range(seq_len):
            dt_t = delta[:, t, :] # [B, d_inner]
            x_t = x_conv[:, t, :] # [B, d_inner]
            b_t = B[:, t, :]    # [B, d_state]
            c_t = C[:, t, :]    # [B, d_state]

            # 矩阵指数离散化近似: A_bar = exp(delta * A)
            dA = torch.exp(dt_t.unsqueeze(-1) * A) # [B, d_inner, d_state]
            dB = dt_t.unsqueeze(-1) * b_t.unsqueeze(1) # [B, d_inner, d_state]

            # 状态更新: h = dA * h + dB * x
            h = dA * h + dB * x_t.unsqueeze(-1) # [B, d_inner, d_state]
            
            # 输出计算: y = sum(h * C)
            y_t = torch.sum(h * c_t.unsqueeze(1), dim=-1) # [B, d_inner]
            y[:, t, :] = y_t

        # 加上残差项 D * x
        y = y + x_conv * self.D.unsqueeze(0).unsqueeze(0) # [B, L, d_inner]
        
        # 5. 门控相乘与最终投影
        y = y * F.silu(z) # [B, L, d_inner]
        out = self.out_proj(y) # [B, L, d_model]
        return out


class PureMambaLM(nn.Module):
    """
    基于纯 PyTorch 实现的 Mamba 语言模型 (PureMambaLM)。

    数学原理 / 变换逻辑:
        将输入的词索引通过 Embedding 映射为连续向量，堆叠多个 SimpleMambaBlock 进行序列特征提取，
        经 LayerNorm 后通过线性分类头输出词表上的概率分布。

    Args:
        vocab_size (int): 词表大小。
        d_model (int): 模型隐藏层维度，默认 128。
        n_layer (int): Mamba 块的堆叠层数，默认 2。

    Inputs:
        input_ids (Tensor): 输入 Token 索引，shape: [B, Seq_Len]
        labels (Tensor, optional): 目标 Token 索引，shape: [B, Seq_Len]

    Outputs:
        dict: 包含 logits [B, Seq_Len, vocab_size] 与可选的 loss。
    """
    def __init__(self, vocab_size, d_model=128, n_layer=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([SimpleMambaBlock(d_model) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # 权重共享
        self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids) # [B, L, d_model]
        for layer in self.layers:
            x = x + layer(x) # 残差连接，[B, L, d_model]
        x = self.norm(x) # [B, L, d_model]
        logits = self.lm_head(x) # [B, L, vocab_size]

        loss = None
        if labels is not None:
            # 语言模型经典的 Shift 机制：用位置 t 预测 t+1
            shift_logits = logits[..., :-1, :].contiguous() # [B, L-1, vocab_size]
            shift_labels = labels[..., 1:].contiguous() # [B, L-1]
            
            # 损失函数：交叉熵
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )
        return {"logits": logits, "loss": loss}


class ToyDataset(Dataset):
    """
    用于测试的虚拟数据集。
    """
    def __init__(self, data):
        self.data = data
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return {"input_ids": torch.tensor(self.data[idx], dtype=torch.long)}


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab_size = 100

    # 构造一些虚拟训练样本 (Token IDs)
    raw_data = [
        [12, 34, 56, 78, 90, 23, 45, 67, 89, 11, 22, 33, 44, 55, 66, 77],
        [55, 44, 33, 22, 11, 89, 67, 45, 23, 90, 78, 56, 34, 12, 99, 88]
    ] * 10

    dataset = ToyDataset(raw_data)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 实例化纯 PyTorch 模型
    model = PureMambaLM(vocab_size=vocab_size, d_model=64, n_layer=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    model.train()
    print("开始纯 PyTorch Mamba 训练...")
    
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            labels = input_ids.clone()
            labels[labels == 0] = -100 # 假设 0 为填充符，忽略不计算损失

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader):.4f}")

    print("训练完成！")

if __name__ == "__main__":
    main()