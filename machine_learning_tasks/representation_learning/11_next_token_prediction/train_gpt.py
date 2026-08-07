"""
任务定义:
    任务编号: Task 11 - 下一词元预测 (Next-Token Prediction / Causal Language Modeling)
    领域分类: 自然语言处理 (NLP) / 生成式语言模型 (Generative LM)

代表架构/算法:
    GPT (Generative Pre-trained Transformer - Decoder-Only Architecture)
    经典论文: "Improving Language Understanding by Generative Pre-Training" (Radford et al., 2018)

核心思想与机制:
    使用基于自回归 (Autoregressive) 的纯解码器 Transformer 架构。通过因果掩码 (Causal Mask / Upper Triangular Mask)
    屏蔽未来词元的信息，使得模型在预测第 t 个词元时，仅能获取位置 1 到 t-1 的上下文信息。

数学公式/目标函数:
    1. 因果多头注意力机制 (Causal Multi-Head Attention):
       Attention(Q, K, V) = Softmax((Q * K^T) / sqrt(d_k) + M) * V
       其中 M 为因果掩码矩阵, 上三角 (不含对角线) 为 -inf, 其余为 0。

    2. 自回归交叉熵损失 (Autoregressive Cross-Entropy Loss):
       L_CLM = - (1 / N) * sum_{t=1}^{N} log P(x_t | x_1, x_2, ..., x_{t-1}; theta)

数据输入/输出规范:
    输入 (x): 张量 shape = [B, L] (类型: torch.long, 标记 Token ID 序列)
    输出 (logits): 张量 shape = [B, L, Vocab_Size] (类型: torch.float32, 对应词表未归一化概率)
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ==============================================================================
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
VOCAB_SIZE = 1000
SEQ_LEN = 64
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
FF_DIM = 256
DROPOUT = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ==============================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ==============================================================================
def get_synthetic_dataset(num_samples: int = 2000, seq_len: int = SEQ_LEN, vocab_size: int = VOCAB_SIZE) -> TensorDataset:
    """
    生成用于自回归语言模型训练的合成 Token 序列数据集。

    Args:
        num_samples (int): 样本数量。
        seq_len (int): 序列总长度 (输入长度 + 1)。
        vocab_size (int): 词表大小。

    Outputs:
        TensorDataset包含:
            x (Tensor): [Num_Samples, Seq_Len], 类型为 Long Tensor
    """
    # 随机生成范围在 [2, vocab_size) 之间的 Token (保留 0/1 作为特殊 Padding/BOS/EOS Token)
    x = torch.randint(2, vocab_size, (num_samples, seq_len), dtype=torch.long)
    return TensorDataset(x)


# ==============================================================================
# 5. 核心子模块 / Encoder / Decoder (Sub-components)
# ==============================================================================
class CausalMultiHeadAttention(nn.Module):
    """
    手写因果多头自注意力模块 (Causal Multi-Head Self-Attention).

    数学原理 / 变换逻辑:
        Q = X * W_q, K = X * W_k, V = X * W_v
        Scores = (Q * K^T) / sqrt(d_k)
        Scores_masked = Scores + Mask  (Mask 中被屏蔽位置填充 -inf)
        Attn_Weights = Softmax(Scores_masked)
        Output = Attn_Weights * V * W_o

    Args:
        d_model (int): 输入与输出特征维度 C。
        nhead (int): 注意力头数 H。
        dropout (float): Dropout 概率。

    Inputs:
        x (Tensor): 输入特征矩阵，shape: [B, L, C]
        attn_mask (Tensor, optional): 因果掩码矩阵，shape: [1, 1, L, L] 或 [B, 1, L, L]，屏蔽未来位置

    Outputs:
        out (Tensor): 多头注意力计算后的投影特征，shape: [B, L, C]
    """
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) 必须能被 nhead ({nhead}) 整除"
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape  # [B, L, C]

        # 1. 线性投影并维度拆分 -> [B, H, L, D_head]
        q = self.w_q(x).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)

        # 2. 计算缩放点积注意力分数 Scores = Q @ K^T / sqrt(d_head)
        # [B, H, L, D_head] @ [B, H, D_head, L] -> [B, H, L, L]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)

        # 3. 应用因果掩码 (Mask 值为 True 的位置替换为 -inf)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))

        # 4. Softmax 归一化与 Dropout
        attn_weights = torch.softmax(scores, dim=-1)  # [B, H, L, L]
        attn_weights = self.attn_dropout(attn_weights)

        # 5. 加权求和注意力输出
        # [B, H, L, L] @ [B, H, L, D_head] -> [B, H, L, D_head]
        out = torch.matmul(attn_weights, v)

        # 6. 拼接多头并线性投影 -> [B, L, C]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.proj_dropout(self.w_o(out))
        return out


class FeedForward(nn.Module):
    """
    两层前馈全连接神经网络 (MLP/FeedForward Block).

    数学原理 / 变换逻辑:
        FFN(x) = GELU(x * W1 + b1) * W2 + b2

    Args:
        d_model (int): 输入/输出维度。
        dim_feedforward (int): 隐藏层升维维度。
        dropout (float): Dropout 概率。

    Inputs:
        x (Tensor): 输入特征，shape: [B, L, C]

    Outputs:
        out (Tensor): 输出特征，shape: [B, L, C]
    """
    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, L, C] -> [B, L, Dim_FF] -> [B, L, C]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DecoderLayer(nn.Module):
    """
    单层 Transformer 解码器块 (Transformer Decoder Layer - Post-LN/Standard Architecture).

    数学原理 / 变换逻辑:
        x = LayerNorm(x + Dropout(SelfAttention(x)))
        x = LayerNorm(x + Dropout(FeedForward(x)))

    Args:
        d_model (int): 特征维度。
        nhead (int): 注意力头数。
        dim_feedforward (int): FFN 中间隐含层维度。
        dropout (float): 正则化 Dropout 概率。

    Inputs:
        x (Tensor): 输入序列表示，shape: [B, L, C]
        attn_mask (Tensor, optional): 因果掩码，shape: [1, 1, L, L]

    Outputs:
        x (Tensor): 经过自注意力与 FFN 后的更新表示，shape: [B, L, C]
    """
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = CausalMultiHeadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        # 自注意力子层 + 残差连接 + LayerNorm
        attn_out = self.self_attn(x, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))  # [B, L, C]

        # 前馈网络子层 + 残差连接 + LayerNorm
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))    # [B, L, C]
        return x


# ==============================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ==============================================================================
class GPT(nn.Module):
    """
    GPT 解码器语言模型主体 (Decoder-Only Generative Pre-trained Transformer).

    架构逻辑:
        Token Embedding + Learned Positional Embedding -> N x Decoder Layers -> Head -> Unnormalized Logits

    Args:
        vocab_size (int): 词表大小。
        max_seq_len (int): 支持的最大上下文长度。
        d_model (int): 特征嵌入维度。
        nhead (int): 注意力头数。
        num_layers (int): Transformer 解码层堆叠层数。
        ff_dim (int): FFN 隐藏特征维度。
        dropout (float): Dropout 比例。

    Inputs:
        x (Tensor): 输入 Token ID 序列，shape: [B, L]

    Outputs:
        logits (Tensor): 预测下一个 Token 的类别对数概率，shape: [B, L, Vocab_Size]
    """
    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        max_seq_len: int = SEQ_LEN,
        d_model: int = D_MODEL,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len

        # 词嵌入与可学习位置编码
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        # 解码器层堆叠
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, nhead, ff_dim, dropout) for _ in range(num_layers)
        ])

        # 输出语言模型 Head
        self.head = nn.Linear(d_model, vocab_size)

        # 注册全局因果上三角掩码矩阵 (True 对应 -inf 屏蔽区域)
        mask = torch.triu(torch.ones(max_seq_len, max_seq_len), diagonal=1).bool()
        self.register_buffer("causal_mask", mask.view(1, 1, max_seq_len, max_seq_len))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape  # [B, L]
        assert seq_len <= self.max_seq_len, f"输入序列长度 {seq_len} 超过最大允许长度 {self.max_seq_len}"

        # 1. 生成位置索引 [0, 1, ..., L-1] -> [L]
        pos = torch.arange(0, seq_len, dtype=torch.long, device=x.device)

        # 2. Token Embedding 与 Position Embedding 融合
        # [B, L] -> [B, L, C]
        tok_out = self.token_emb(x)
        # [L] -> [L, C] -> 广播到 [B, L, C]
        pos_out = self.pos_emb(pos)
        h = self.emb_dropout(tok_out + pos_out)  # [B, L, C]

        # 3. 动态截取适应当前序列长度的因果掩码 [1, 1, L, L]
        attn_mask = self.causal_mask[:, :, :seq_len, :seq_len]

        # 4. 逐层通过 Decoder Block
        for layer in self.layers:
            h = layer(h, attn_mask)  # [B, L, C]

        # 5. 投影至词表空间得到 Logits [B, L, Vocab_Size]
        logits = self.head(h)
        return logits


# ==============================================================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ==============================================================================
class AutoregressiveCrossEntropyLoss(nn.Module):
    """
    自回归交叉熵损失。封装多维展平计算，自动处理 Logits [B, L, V] 与 Target [B, L] 的维度匹配。
    """
    def __init__(self):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            logits (Tensor): 模型未归一化预测输出, shape: [B, L, Vocab_Size]
            targets (Tensor): 标注的目标 Token 序列, shape: [B, L]
        Outputs:
            loss (Tensor): 标量 Loss 值
        """
        # [B, L, Vocab_Size] -> [B * L, Vocab_Size]
        logits_flat = logits.reshape(-1, logits.size(-1))
        # [B, L] -> [B * L]
        targets_flat = targets.reshape(-1)
        return self.criterion(logits_flat, targets_flat)


# ==============================================================================
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ==============================================================================
def main():
    print(f"正在使用运行设备: {DEVICE}")

    # 1. 构建数据集与数据加载器
    dataset = get_synthetic_dataset(num_samples=2000, seq_len=SEQ_LEN, vocab_size=VOCAB_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 初始化模型、损失函数与优化器
    model = GPT(
        vocab_size=VOCAB_SIZE,
        max_seq_len=SEQ_LEN,
        d_model=D_MODEL,
        nhead=NHEAD,
        num_layers=NUM_LAYERS,
        ff_dim=FF_DIM,
        dropout=DROPOUT
    ).to(DEVICE)

    criterion = AutoregressiveCrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    # 3. 训练循环
    model.train()
    print("开始训练...")
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, (batch_x,) in enumerate(loader):
            batch_x = batch_x.to(DEVICE)  # [B, Seq_Len]

            # 错位构建自回归 Target:
            # Inputs 为第 0 到 L-2 个 Token; Labels 为第 1 到 L-1 个 Token
            inputs = batch_x[:, :-1]   # [B, L-1]
            targets = batch_x[:, 1:]   # [B, L-1]

            optimizer.zero_grad()

            # 前向传播 logits: [B, L-1, Vocab_Size]
            logits = model(inputs)

            # 计算损失并反向传播
            loss = criterion(logits, targets)
            loss.backward()

            # 梯度裁剪 (稳定 Transformer 训练)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | Next-Token Loss: {avg_loss:.4f}")

    # 4. 推理采样演示 (简单贪婪搜索生成演示)
    model.eval()
    with torch.no_grad():
        prompt = torch.tensor([[10, 20, 30]], device=DEVICE)  # 初始 Prompt, shape [1, 3]
        print(f"\n[推理测试] 初始 Prompt Token IDs: {prompt.cpu().tolist()[0]}")

        generated = prompt
        for _ in range(5):  # 采样生成 5 个后续 Token
            logits = model(generated)                     # [1, L, Vocab_Size]
            next_token_logits = logits[:, -1, :]          # 取最后一个位置 [1, Vocab_Size]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True) # [1, 1]
            generated = torch.cat([generated, next_token], dim=1)            # 拼接到序列尾部

        print(f"[推理测试] 生成的序列 Token IDs: {generated.cpu().tolist()[0]}")


if __name__ == "__main__":
    main()