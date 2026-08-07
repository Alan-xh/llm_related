"""
任务 8：序列生成 (Sequence Generation / Machine Translation)
代表架构: Transformer Seq2Seq Model (Pure PyTorch Implementation)
核心论文: Attention Is All You Need (Vaswani et al., NIPS 2017)

核心思想与机制:
    本实现包含完整的编码器-解码器(Encoder-Decoder)架构，严格从零实现多头自注意力机制 (MHA)、
    正弦/余弦位置编码 (Positional Encoding) 以及带因果掩码 (Casual Mask) 的解码过程。
    通过 Teacher Forcing 模式进行高效的并行训练。

数学公式与目标函数:
    1. Scaled Dot-Product Attention:
       Attention(Q, K, V) = softmax( (Q * K^T) / sqrt(d_k) ) * V
    2. Positional Encoding:
       PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
       PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
    3. Multi-Head Attention:
       MultiHead(Q, K, V) = Concat(head_1, ..., head_h) * W^O
       where head_i = Attention(Q * W_i^Q, K * W_i^K, V * W_i^V)
    4. Cross-Entropy Loss:
       L = - sum_{t=1}^T log P(y_t | y_<t, X)

数据输入规范:
    Input (src)       : [B, S_src] (Token IDs)
    Input (tgt_input) : [B, S_tgt] (Token IDs, Teacher Forcing)
    Output (logits)   : [B, S_tgt, Vocab_Size]
"""

import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ===================================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ===================================================================================
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
VOCAB_SIZE = 50
SRC_LEN = 20
TGT_LEN = 25
EMB_DIM = 64
NHEAD = 4
NUM_LAYERS = 2
FF_DIM = 128
DROPOUT = 0.1
PAD_IDX = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ===================================================================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ===================================================================================
def get_synthetic_dataset(num_samples: int = 2000) -> TensorDataset:
    """
    生成合成序列数据对（模拟机器翻译任务）。

    Args:
        num_samples (int): 样本数量。

    Outputs:
        TensorDataset: 包含 src 与 tgt 的 PyTorch 数据集。
            - src shape: [num_samples, SRC_LEN]
            - tgt shape: [num_samples, TGT_LEN]
    """
    # 词表范围留出 0 作为 PAD, 1 作为 BOS/EOS
    src = torch.randint(2, VOCAB_SIZE, (num_samples, SRC_LEN))
    tgt = torch.randint(2, VOCAB_SIZE, (num_samples, TGT_LEN))
    return TensorDataset(src, tgt)


# ===================================================================================
# 5. 核心子模块 / Encoder / Decoder (Sub-components)
# ===================================================================================
class PositionalEncoding(nn.Module):
    """
    绝对正弦/余弦位置编码模块。

    数学原理:
        PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

    Args:
        d_model (int): 隐藏层特征维度。
        max_len (int): 序列最大长度。
    """

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )  # [d_model / 2]

        pe[:, 0::2] = torch.sin(position * div_term)  # [max_len, d_model/2]
        pe[:, 1::2] = torch.cos(position * div_term)  # [max_len, d_model/2]

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 输入特征，shape: [B, S, d_model]

        Outputs:
            Tensor: 叠加位置编码后的特征，shape: [B, S, d_model]
        """
        seq_len = x.size(1)
        # x: [B, S, d_model] + pe: [1, S, d_model] -> [B, S, d_model]
        return x + self.pe[:, :seq_len, :]


class MultiHeadAttention(nn.Module):
    """
    手写多头自注意力/交叉注意力模块 (Multi-Head Attention)。

    数学原理:
        Attention(Q, K, V) = softmax( (Q @ K^T) / sqrt(d_k) ) @ V

    Args:
        d_model (int): 模型的特征维度。
        nhead (int): 注意力头数。
        dropout (float): Dropout 概率。
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % nhead == 0, "d_model 必须能被 nhead 整除"
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Inputs:
            query (Tensor): [B, S_q, d_model]
            key (Tensor):   [B, S_k, d_model]
            value (Tensor): [B, S_k, d_model]
            attn_mask (Tensor, optional): [B, 1, S_q, S_k] 或 [1, 1, S_q, S_k] (True 表示需要被掩盖)

        Outputs:
            out (Tensor): 经过 MHA 处理后的特征，shape: [B, S_q, d_model]
        """
        bsz = query.size(0)

        # 线性变换与分头: [B, S, d_model] -> [B, S, nhead, d_head] -> [B, nhead, S, d_head]
        Q = self.w_q(query).view(bsz, -1, self.nhead, self.d_head).transpose(1, 2)
        K = self.w_k(key).view(bsz, -1, self.nhead, self.d_head).transpose(1, 2)
        V = self.w_v(value).view(bsz, -1, self.nhead, self.d_head).transpose(1, 2)

        # Q @ K^T: [B, nhead, S_q, d_head] @ [B, nhead, d_head, S_k] -> [B, nhead, S_q, S_k]
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)

        if attn_mask is not None:
            # 将 mask 为 True 的位置替换为 -inf
            scores = scores.masked_fill(attn_mask, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # attn @ V: [B, nhead, S_q, S_k] @ [B, nhead, S_k, d_head] -> [B, nhead, S_q, d_head]
        out = torch.matmul(attn, V)

        # 拼接头: [B, nhead, S_q, d_head] -> [B, S_q, nhead, d_head] -> [B, S_q, d_model]
        out = out.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)

        return self.w_o(out)


class FeedForward(nn.Module):
    """
    位置逐帧前馈神经网络 (Position-wise Feed-Forward Network)。

    数学原理:
        FFN(x) = max(0, x * W1 + b1) * W2 + b2

    Args:
        d_model (int): 输入输出特征维度。
        dim_feedforward (int): 隐藏层升维维度。
        dropout (float): Dropout 概率。
    """

    def __init__(self, d_model: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            x (Tensor): [B, S, d_model]

        Outputs:
            out (Tensor): [B, S, d_model]
        """
        # [B, S, d_model] -> [B, S, dim_feedforward]
        x = self.dropout(self.activation(self.fc1(x)))
        # [B, S, dim_feedforward] -> [B, S, d_model]
        return self.fc2(x)


class TransformerEncoderLayer(nn.Module):
    """Transformer 编码器单层模块。"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Inputs:
            src (Tensor): [B, S_src, d_model]
            src_mask (Tensor, optional): [B, 1, 1, S_src]

        Outputs:
            src (Tensor): [B, S_src, d_model]
        """
        # Residual + LayerNorm (Post-LN Architecture)
        attn_out = self.self_attn(src, src, src, attn_mask=src_mask)
        src = self.norm1(src + self.dropout(attn_out))

        ff_out = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_out))
        return src


class TransformerDecoderLayer(nn.Module):
    """Transformer 解码器单层模块（包含 Self-Attention 和 Cross-Attention）。"""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.cross_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.feed_forward = FeedForward(d_model, dim_feedforward, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Inputs:
            tgt (Tensor): [B, S_tgt, d_model]
            memory (Tensor): Encoder 输出，[B, S_src, d_model]
            tgt_mask (Tensor, optional): 因果掩码，[1, 1, S_tgt, S_tgt]
            memory_mask (Tensor, optional): [B, 1, 1, S_src]

        Outputs:
            tgt (Tensor): [B, S_tgt, d_model]
        """
        # Masked Self-Attention
        attn_out = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_out))

        # Cross-Attention (Query: tgt, Key/Value: memory)
        cross_out = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask)
        tgt = self.norm2(tgt + self.dropout(cross_out))

        # Feed Forward
        ff_out = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_out))
        return tgt


class TransformerEncoder(nn.Module):
    """Transformer 编码器整体堆叠。"""

    def __init__(
        self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Inputs:
            src (Tensor): [B, S_src, d_model]
            src_mask (Tensor, optional): [B, 1, 1, S_src]

        Outputs:
            src (Tensor): [B, S_src, d_model]
        """
        for layer in self.layers:
            src = layer(src, src_mask)
        return src


class TransformerDecoder(nn.Module):
    """Transformer 解码器整体堆叠。"""

    def __init__(
        self, d_model: int, nhead: int, num_layers: int, dim_feedforward: int, dropout: float = 0.1
    ):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Inputs:
            tgt (Tensor): [B, S_tgt, d_model]
            memory (Tensor): [B, S_src, d_model]
            tgt_mask (Tensor, optional): [1, 1, S_tgt, S_tgt]
            memory_mask (Tensor, optional): [B, 1, 1, S_src]

        Outputs:
            tgt (Tensor): [B, S_tgt, d_model]
        """
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask, memory_mask)
        return tgt


# ===================================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture)
# ===================================================================================
class Seq2SeqTransformer(nn.Module):
    """
    端到端 Transformer 序列到序列生成模型。

    Args:
        vocab_size (int): 词表大小。
        emb_dim (int): 词嵌入维度。
        nhead (int): 注意力头数。
        num_layers (int): Encoder 和 Decoder 层数。
        ff_dim (int): 前馈网络隐藏维度。
        dropout (float): Dropout 概率。
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        emb_dim: int = EMB_DIM,
        nhead: int = NHEAD,
        num_layers: int = NUM_LAYERS,
        ff_dim: int = FF_DIM,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.src_emb = nn.Embedding(vocab_size, emb_dim)
        self.tgt_emb = nn.Embedding(vocab_size, emb_dim)
        self.pos_encoder = PositionalEncoding(emb_dim)

        self.encoder = TransformerEncoder(emb_dim, nhead, num_layers, ff_dim, dropout)
        self.decoder = TransformerDecoder(emb_dim, nhead, num_layers, ff_dim, dropout)

        self.fc_out = nn.Linear(emb_dim, vocab_size)
        self.emb_scale = math.sqrt(emb_dim)

    def generate_square_subsequent_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """
        生成解码器的下三角因果掩码 (Causal Mask)。

        Outputs:
            mask (Tensor): [1, 1, sz, sz]，上三角部分（不含对角线）全为 True
        """
        mask = torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, sz, sz]

    def forward(self, src: torch.Tensor, tgt_input: torch.Tensor) -> torch.Tensor:
        """
        Inputs:
            src (Tensor): 源语言序列，shape: [B, S_src]
            tgt_input (Tensor): 目标语言输入（Shifted Right），shape: [B, S_tgt]

        Outputs:
            logits (Tensor): 预测词表分布概率，shape: [B, S_tgt, Vocab_Size]
        """
        tgt_len = tgt_input.size(1)

        # 1. 构造 Mask
        # tgt_mask: [1, 1, S_tgt, S_tgt]
        tgt_mask = self.generate_square_subsequent_mask(tgt_len, tgt_input.device)

        # 2. Embedding + Positional Encoding
        # src_emb: [B, S_src] -> [B, S_src, EMB_DIM]
        src_feat = self.pos_encoder(self.src_emb(src) * self.emb_scale)
        # tgt_emb: [B, S_tgt] -> [B, S_tgt, EMB_DIM]
        tgt_feat = self.pos_encoder(self.tgt_emb(tgt_input) * self.emb_scale)

        # 3. Encoder Forward
        # memory: [B, S_src, EMB_DIM]
        memory = self.encoder(src_feat)

        # 4. Decoder Forward
        # out: [B, S_tgt, EMB_DIM]
        out = self.decoder(tgt_feat, memory, tgt_mask=tgt_mask)

        # 5. Output Projection
        # logits: [B, S_tgt, Vocab_Size]
        logits = self.fc_out(out)
        return logits


# ===================================================================================
# 7. 损失函数与评估指标 & 8. 训练/推理逻辑 (Training Execution)
# ===================================================================================
def main():
    print(f"Executing on device: {DEVICE}")

    # 数据集构建
    dataset = get_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型与优化器初始化
    model = Seq2SeqTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 训练循环
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for src, tgt in loader:
            src = src.to(DEVICE)  # [B, S_src]
            tgt = tgt.to(DEVICE)  # [B, S_tgt]

            # Teacher Forcing: 输入切片为 [0, T-1], 标签切片为 [1, T]
            tgt_input = tgt[:, :-1]  # [B, S_tgt-1]
            tgt_label = tgt[:, 1:]  # [B, S_tgt-1]

            optimizer.zero_grad()

            # Forward Forward: logits -> [B, S_tgt-1, VOCAB_SIZE]
            logits = model(src, tgt_input)

            # Flatten 张量计算交叉熵损失
            loss = criterion(
                logits.reshape(-1, VOCAB_SIZE),  # [(B * (S_tgt-1)), VOCAB_SIZE]
                tgt_label.reshape(-1),  # [(B * (S_tgt-1))]
            )

            loss.backward()
            # 梯度裁剪防梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}] | CrossEntropy Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()