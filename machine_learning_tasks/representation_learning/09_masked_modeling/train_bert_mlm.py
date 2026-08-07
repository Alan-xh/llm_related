"""
任务定义：
    - 任务编号：Task 09
    - 任务名称：掩码语言建模 (Masked Language Modeling, MLM) / 自监督生成式预训练
    - 领域分类：自然语言处理 (NLP) / 语言模型

代表架构/算法：
    - BERT (Bidirectional Encoder Representations from Transformers)
    - 论文来源：Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for 
      Language Understanding", NAACL 2019.

核心思想与机制：
    - 核心思想：利用双向 Transformer 编码器提取上下文语义表示，采用自监督掩码预测作为预训练目标。
    - 机制流程：
        1. 掩码策略 (Masking Strategy)：从输入序列中随机选择 15% 的 Token，其中 80% 替换为 [MASK]，
           10% 替换为随机 Token，10% 保持不变。
        2. 位置与词表嵌入 (Embeddings)：将 Token ID 与 Position ID 分别映射为 d_model 维向量并相加。
        3. 双向编码 (Transformer Encoder)：通过多层多头自注意力机制 (Multi-Head Self-Attention)
           和前馈网络 (FFN) 构建全局上下文表征。
        4. MLM 预测 Head：利用线性层投影回词表空间，计算被掩码位置的 Cross-Entropy Loss。

数学公式 / 目标函数：
    - 多头注意力公式 (Multi-Head Attention):
        Attention(Q, K, V) = Softmax(Q K^T / sqrt(d_k)) * V
        MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
        where head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)
    - 掩码语言模型优化目标 (MLM Loss):
        L_{MLM}(\theta) = - \sum_{i \in M} \log P(x_i \mid \mathbf{\tilde{x}}; \theta)
        其中 M 表示被遮挡的索引集合，\mathbf{\tilde{x}} 表示被遮挡后的输入序列。

数据输入规范：
    - 输入张量 (Input Tensor):
        - `x`: [B, Seq_Len] (Dtype: torch.long, Token IDs)
    - 输出张量 (Output Tensor):
        - `logits`: [B, Seq_Len, Vocab_Size] (Dtype: torch.float32, Unnormalized Probabilities)
    - 损失掩码 (Loss Mask):
        - `labels`: [B, Seq_Len] (Dtype: torch.long, 非遮挡位置标为 -100 以忽略梯度计算)
"""

import math
from typing import Tuple, Optional
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ======================================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ======================================================================================
BATCH_SIZE: int = 32
EPOCHS: int = 5
LR: float = 1e-3
VOCAB_SIZE: int = 1000
SEQ_LEN: int = 64
MASK_ID: int = 1
PAD_ID: int = 0
D_MODEL: int = 128
NHEAD: int = 4
NUM_LAYERS: int = 4
FF_DIM: int = 256
DROPOUT: float = 0.1
DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ======================================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ======================================================================================
def mask_tokens(inputs: torch.Tensor, mask_prob: float = 0.15) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    对输入的 Token 序列执行 80-10-10 MLM 掩码处理策略。

    Args:
        inputs (Tensor): 原始 Token ID 张量，shape: [B, Seq_Len]
        mask_prob (float): 被选择掩码的概率，默认 0.15

    Returns:
        masked_inputs (Tensor): 掩码后的输入张量，shape: [B, Seq_Len]
        labels (Tensor): MLM 训练目标张量，未被选中位置标为 -100，shape: [B, Seq_Len]
    """
    labels = inputs.clone()  # [B, Seq_Len]
    
    # 构建 15% 掩码选择矩阵（排除 PAD Token）
    probability_matrix = torch.full(inputs.shape, mask_prob, device=inputs.device)
    non_pad_mask = inputs != PAD_ID
    probability_matrix.masked_fill_(~non_pad_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()  # [B, Seq_Len]
    labels[~masked_indices] = -100  # 忽略未遮挡 Token 的 Loss 计算

    # 80% 替换为 [MASK]
    indices_replaced = torch.bernoulli(torch.full(inputs.shape, 0.8, device=inputs.device)).bool() & masked_indices
    inputs[indices_replaced] = MASK_ID

    # 10% 替换为 随机 Token (10% / (100% - 80%) = 0.5)
    indices_random = torch.bernoulli(torch.full(inputs.shape, 0.5, device=inputs.device)).bool() & masked_indices & ~indices_replaced
    random_tokens = torch.randint(2, VOCAB_SIZE, inputs.shape, dtype=torch.long, device=inputs.device)
    inputs[indices_random] = random_tokens[indices_random]

    # 余下 10% 保持原样 (剩余 masked_indices 但未被上述命中的位置)
    return inputs, labels


def get_synthetic_dataset(num_samples: int = 2000) -> TensorDataset:
    """
    生成合成数据集用于测试和模型训练。

    Args:
        num_samples (int): 样本总量

    Returns:
        TensorDataset: 封装后的合成 PyTorch 数据集
    """
    # 随机生成 Token ID (2 ~ VOCAB_SIZE-1，保留 0 为 PAD, 1 为 MASK)
    x = torch.randint(2, VOCAB_SIZE, (num_samples, SEQ_LEN), dtype=torch.long)
    return TensorDataset(x)


# ======================================================================================
# 5. 核心子模块 / Encoder / Decoder (Sub-components)
# ======================================================================================
class MultiHeadAttention(nn.Module):
    """
    标准多头自注意力机制 (Multi-Head Self-Attention)。

    数学原理 / 变换逻辑:
        1. 线性投影: Q = X * W_Q, K = X * W_K, V = X * W_V
        2. 维度分割: [B, N, C] -> [B, N, h, d_k] -> [B, h, N, d_k]
        3. 注意力打分: Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) * V
        4. 多头拼接并输出: Output = Concat(head_1, ..., head_h) * W_O

    Args:
        d_model (int): 输入与输出特征维度 (C)。
        nhead (int): 注意力头数 (h)。
        dropout (float): Dropout 概率。

    Inputs:
        x (Tensor): 输入特征序列，shape: [B, Seq_Len, d_model]
        attn_mask (Tensor, optional): 注意力掩码张量，shape: [B, 1, 1, Seq_Len] 或 [B, 1, Seq_Len, Seq_Len]

    Outputs:
        out (Tensor): 多头注意力融合后的输出，shape: [B, Seq_Len, d_model]
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) 必须能被 nhead ({nhead}) 整除"
        
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.d_model = d_model

        # 投影矩阵权重定义
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seq_len, _ = x.shape  # [B, Seq_Len, d_model]

        # 1. 线性变换与维度重塑拆分多头
        # [B, Seq_Len, d_model] -> [B, Seq_Len, nhead, d_head] -> [B, nhead, Seq_Len, d_head]
        Q = self.w_q(x).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)
        K = self.w_k(x).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)
        V = self.w_v(x).view(bsz, seq_len, self.nhead, self.d_head).transpose(1, 2)

        # 2. 计算 Scaled Dot-Product Attention
        # Q: [B, nhead, Seq_Len, d_head], K^T: [B, nhead, d_head, Seq_Len]
        # scores: [B, nhead, Seq_Len, Seq_Len]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, -1e9)

        # Softmax 归一化权重计算
        attn_weights = torch.softmax(scores, dim=-1)  # [B, nhead, Seq_Len, Seq_Len]
        attn_weights = self.dropout(attn_weights)

        # 3. 加权求和融合特征: [B, nhead, Seq_Len, Seq_Len] x [B, nhead, Seq_Len, d_head] -> [B, nhead, Seq_Len, d_head]
        out = torch.matmul(attn_weights, V)

        # 4. 拼接多头特征并线性输出
        # [B, nhead, Seq_Len, d_head] -> [B, Seq_Len, nhead, d_head] -> [B, Seq_Len, d_model]
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        out = self.w_o(out)  # [B, Seq_Len, d_model]
        
        return out


class FeedForward(nn.Module):
    """
    逐位置前馈网络 (Position-wise Feed-Forward Network)。

    数学原理 / 变换逻辑:
        FFN(x) = GELU(x W_1 + b_1) W_2 + b_2

    Args:
        d_model (int): 输入输出特征维度。
        dim_feedforward (int): 隐藏层升维维度。
        dropout (float): Dropout 概率。

    Inputs:
        x (Tensor): 输入张量，shape: [B, Seq_Len, d_model]

    Outputs:
        out (Tensor): 输出张量，shape: [B, Seq_Len, d_model]
    """

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, Seq_Len, d_model] -> [B, Seq_Len, dim_feedforward]
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        # [B, Seq_Len, dim_feedforward] -> [B, Seq_Len, d_model]
        x = self.fc2(x)
        return x


class TransformerEncoderLayer(nn.Module):
    """
    单层 Transformer 编码器层 (Pre-LN 架构，提升训练稳定性)。

    数学原理 / 变换逻辑:
        x' = x + Dropout(SelfAttention(LayerNorm(x)))
        out = x' + Dropout(FFN(LayerNorm(x')))

    Args:
        d_model (int): 模型通道维度。
        nhead (int): 多头注意力头数。
        dim_feedforward (int): 前馈网络隐藏层维度。
        dropout (float): 正则化 Dropout 概率。

    Inputs:
        x (Tensor): 输入隐藏特征，shape: [B, Seq_Len, d_model]
        mask (Tensor, optional): 注意力掩码，shape: [B, 1, 1, Seq_Len]

    Outputs:
        out (Tensor): 编码特征，shape: [B, Seq_Len, d_model]
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout=dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # 子层 1: Self-Attention (Pre-LN)
        norm_x = self.norm1(x)  # [B, Seq_Len, d_model]
        attn_out = self.self_attn(norm_x, attn_mask=mask)  # [B, Seq_Len, d_model]
        x = x + self.dropout1(attn_out)  # 残差连接

        # 子层 2: Feed Forward (Pre-LN)
        norm_x = self.norm2(x)  # [B, Seq_Len, d_model]
        ff_out = self.feed_forward(norm_x)  # [B, Seq_Len, d_model]
        x = x + self.dropout2(ff_out)  # 残差连接
        return x


class TransformerEncoder(nn.Module):
    """
    多层 Transformer 编码器堆叠 (Transformer Encoder Stack)。

    Args:
        d_model (int): 模型通道维度。
        nhead (int): 多头注意力头数。
        num_layers (int): EncoderLayer 堆叠层数。
        dim_feedforward (int): 前馈网络通道数。
        dropout (float): Dropout 概率。

    Inputs:
        x (Tensor): 初始嵌入表示，shape: [B, Seq_Len, d_model]
        mask (Tensor, optional): 注意力掩码，shape: [B, 1, 1, Seq_Len]

    Outputs:
        x (Tensor): 深层上下文表示，shape: [B, Seq_Len, d_model]
    """

    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask=mask)  # [B, Seq_Len, d_model]
        return self.final_norm(x)


# ======================================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ======================================================================================
class BertMLM(nn.Module):
    """
    BERT 掩码语言建模顶层网络 (BERT for Masked Language Modeling)。

    架构设计:
        Token Embedding + Positional Embedding -> Transformer Encoders -> MLM Prediction Head

    Args:
        vocab_size (int): 词表大小。
        d_model (int): 模型隐藏层维度。
        seq_len (int): 序列最大长度。
        nhead (int): 注意力头数。
        num_layers (int): 编码器层数。
        ff_dim (int): FFN 维度。
        dropout (float): Dropout 概率。

    Inputs:
        x (Tensor): 输入 Token IDs，shape: [B, Seq_Len]

    Outputs:
        logits (Tensor): 针对每个 Token 的词表预测概率分布 (未归一化 logits)，
                        shape: [B, Seq_Len, Vocab_Size]
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = D_MODEL,
        seq_len: int = SEQ_LEN,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT
    ):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=ff_dim,
            dropout=dropout
        )

        # MLM 预测头 (Prediction Head)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = x.shape  # [B, Seq_Len]

        # 1. 位置序列构造: [Seq_Len] -> 广播匹配 [B, Seq_Len]
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(bsz, -1)

        # 2. 词嵌入与位置嵌入叠加
        # token_emb(x): [B, Seq_Len, d_model], pos_emb(pos): [B, Seq_Len, d_model]
        h = self.token_emb(x) + self.pos_emb(pos)  # [B, Seq_Len, d_model]
        h = self.emb_dropout(h)

        # 3. 构造 Padding Attention Mask (将 PAD 位置遮挡)
        # [B, Seq_Len] -> [B, 1, 1, Seq_Len]
        pad_mask = (x != PAD_ID).unsqueeze(1).unsqueeze(2)

        # 4. 经过双向 Transformer Encoder
        h = self.encoder(h, mask=pad_mask)  # [B, Seq_Len, d_model]

        # 5. 投影到 Vocabulary logits 空间
        logits = self.head(h)  # [B, Seq_Len, Vocab_Size]
        return logits


# ======================================================================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ======================================================================================
def compute_mlm_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    计算 MLM 任务的交叉熵损失 (自动忽略 -100 标签位置)。

    Args:
        logits (Tensor): 模型预测结果，shape: [B, Seq_Len, Vocab_Size]
        labels (Tensor): 掩码真实标签，shape: [B, Seq_Len]

    Returns:
        loss (Tensor): 标量 Loss 值
    """
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    # 展平特征计算 Cross Entropy
    # logits: [B * Seq_Len, Vocab_Size], labels: [B * Seq_Len]
    loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))
    return loss


# ======================================================================================
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ======================================================================================
def main():
    print(f"--> 使用运行设备: {DEVICE}")

    # 1. 数据管道构建
    dataset = get_synthetic_dataset(num_samples=2000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 2. 模型初始化
    model = BertMLM().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)

    print(f"--> 模型构建完成，总可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 3. 训练循环 (Training Loop)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for step, (x_batch,) in enumerate(loader):
            x_batch = x_batch.to(DEVICE)  # [B, Seq_Len]

            # 动态执行掩码处理
            inputs, labels = mask_tokens(x_batch)  # inputs: [B, Seq_Len], labels: [B, Seq_Len]

            # 前向传播
            optimizer.zero_grad()
            logits = model(inputs)  # [B, Seq_Len, Vocab_Size]

            # Loss 计算与反向传播
            loss = compute_mlm_loss(logits, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1:02d}/{EPOCHS:02d}] | Average MLM Loss: {avg_loss:.4f}")

    print("--> 训练完成！执行简单验证推断...")

    # 4. 推断体验 (Inference Demo)
    model.eval()
    with torch.no_grad():
        test_sample = torch.randint(2, VOCAB_SIZE, (1, SEQ_LEN), device=DEVICE)
        masked_input, ground_truth = mask_tokens(test_sample)
        
        test_logits = model(masked_input)  # [1, Seq_Len, Vocab_Size]
        preds = torch.argmax(test_logits, dim=-1)  # [1, Seq_Len]

        masked_positions = (ground_truth != -100).squeeze(0)
        print(f"示例序列掩码位置总数: {masked_positions.sum().item()}")
        print(f"真实 Token (Ground Truth): {ground_truth[0][masked_positions].cpu().numpy()[:5]}")
        print(f"预测 Token (Predictions) : {preds[0][masked_positions].cpu().numpy()[:5]}")


if __name__ == "__main__":
    main()