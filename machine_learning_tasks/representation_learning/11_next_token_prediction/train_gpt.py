"""
任务 11：下一词元预测（Next-Token Prediction）
代表模型：GPT（手写 Decoder-only Transformer，不调用 nn.TransformerEncoder）
损失函数：自回归交叉熵损失
使用合成 token 序列训练一个因果语言模型。
"""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import qwen3
# 超参数
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
VOCAB_SIZE = 1000
SEQ_LEN = 64
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
FF_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CausalMultiHeadAttention(nn.Module):
    """手写因果多头自注意力。"""

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

    def forward(self, x, attn_mask):
        bsz = x.size(0)
        Q = self.w_q(x).view(bsz, -1, self.nhead, self.d_head).transpose(1, 2)
        K = self.w_k(x).view(bsz, -1, self.nhead, self.d_head).transpose(1, 2)
        V = self.w_v(x).view(bsz, -1, self.nhead, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
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
        self.activation = nn.GELU()

    def forward(self, x):
        return self.fc2(self.activation(self.fc1(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = CausalMultiHeadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        attn_out = self.self_attn(x, attn_mask)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.layers = nn.ModuleList([
            DecoderLayer(D_MODEL, NHEAD, FF_DIM) for _ in range(NUM_LAYERS)
        ])
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

        # 注册因果掩码 (1, 1, SEQ_LEN, SEQ_LEN)
        mask = torch.triu(torch.ones(SEQ_LEN, SEQ_LEN), diagonal=1).bool()
        self.register_buffer("causal_mask", mask.view(1, 1, SEQ_LEN, SEQ_LEN))

    def forward(self, x):
        bsz, seq_len = x.shape
        pos = torch.arange(seq_len, device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)

        # 截取当前序列长度的掩码
        attn_mask = self.causal_mask[:, :, :seq_len, :seq_len]
        for layer in self.layers:
            h = layer(h, attn_mask)
        return self.head(h)


def get_synthetic_dataset(num_samples=2000):
    x = torch.randint(2, VOCAB_SIZE, (num_samples, SEQ_LEN))
    return TensorDataset(x)


def main():
    dataset = get_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = GPT().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for (x,) in loader:
            x = x.to(DEVICE)
            inputs = x[:, :-1]
            labels = x[:, 1:]

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), labels.reshape(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  Next-Token Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
