"""
任务 8：序列生成（Sequence Generation / 机器翻译）
代表模型：Transformer Seq2Seq（手写实现，不调用 nn.Transformer）
损失函数：交叉熵损失（教师强制）
使用合成序列对演示编码器-解码器训练。
"""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 超参数
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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    """手写多头注意力。"""

    def __init__(self, d_model, nhead):
        super().__init__()
        assert d_model % nhead == 0
        self.nhead = nhead
        self.d_head = d_model // nhead
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, query, key, value, attn_mask=None):
        bsz = query.size(0)

        Q = self.w_q(query).view(bsz, -1, self.nhead, self.d_head).transpose(1, 2)
        K = self.w_k(key).view(bsz, -1, self.nhead, self.d_head).transpose(1, 2)
        V = self.w_v(value).view(bsz, -1, self.nhead, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)

        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(bsz, -1, self.d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    """前馈网络。"""

    def __init__(self, d_model, dim_feedforward):
        super().__init__()
        self.fc1 = nn.Linear(d_model, dim_feedforward)
        self.fc2 = nn.Linear(dim_feedforward, d_model)
        self.activation = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        attn_out = self.self_attn(src, src, src)
        src = self.norm1(src + self.dropout(attn_out))
        ff_out = self.feed_forward(src)
        src = self.norm2(src + self.dropout(ff_out))
        return src


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.cross_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, tgt, memory, tgt_mask):
        attn_out = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)
        tgt = self.norm1(tgt + self.dropout(attn_out))
        cross_out = self.cross_attn(tgt, memory, memory)
        tgt = self.norm2(tgt + self.dropout(cross_out))
        ff_out = self.feed_forward(tgt)
        tgt = self.norm3(tgt + self.dropout(ff_out))
        return tgt


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, src):
        for layer in self.layers:
            src = layer(src)
        return src


class TransformerDecoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, tgt_mask):
        for layer in self.layers:
            tgt = layer(tgt, memory, tgt_mask)
        return tgt


class Seq2SeqTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)
        self.tgt_emb = nn.Embedding(VOCAB_SIZE, EMB_DIM)
        self.pos_src = nn.Embedding(SRC_LEN, EMB_DIM)
        self.pos_tgt = nn.Embedding(TGT_LEN, EMB_DIM)

        self.encoder = TransformerEncoder(EMB_DIM, NHEAD, NUM_LAYERS, FF_DIM)
        self.decoder = TransformerDecoder(EMB_DIM, NHEAD, NUM_LAYERS, FF_DIM)
        self.fc = nn.Linear(EMB_DIM, VOCAB_SIZE)

    def forward(self, src, tgt_input):
        src_pos = torch.arange(SRC_LEN, device=src.device)
        tgt_pos = torch.arange(tgt_input.size(1), device=tgt_input.device)

        src_emb = self.src_emb(src) + self.pos_src(src_pos)
        tgt_emb = self.tgt_emb(tgt_input) + self.pos_tgt(tgt_pos)

        # 下三角因果掩码
        tgt_len = tgt_input.size(1)
        tgt_mask = torch.triu(torch.ones(tgt_len, tgt_len), diagonal=1).bool().to(tgt_input.device)
        tgt_mask = tgt_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)

        memory = self.encoder(src_emb)
        out = self.decoder(tgt_emb, memory, tgt_mask)
        return self.fc(out)


def get_synthetic_dataset(num_samples=2000):
    # 源序列和目标序列使用相同词表，模拟翻译对
    src = torch.randint(2, VOCAB_SIZE, (num_samples, SRC_LEN))
    tgt = torch.randint(2, VOCAB_SIZE, (num_samples, TGT_LEN))
    return TensorDataset(src, tgt)


def main():
    dataset = get_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = Seq2SeqTransformer().to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for src, tgt in loader:
            src = src.to(DEVICE)
            tgt = tgt.to(DEVICE)

            tgt_input = tgt[:, :-1]
            tgt_label = tgt[:, 1:]

            optimizer.zero_grad()
            logits = model(src, tgt_input)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), tgt_label.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  CrossEntropy Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
