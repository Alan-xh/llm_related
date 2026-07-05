"""
任务 9：掩码建模（Masked Modeling / 生成式自监督）
代表模型：BERT（手写 Transformer 编码器，不调用 nn.TransformerEncoder）
损失函数：MLM 交叉熵损失
使用合成 token 序列训练一个 Transformer 编码器，预测被遮挡的 token。
"""
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# 超参数
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-3
VOCAB_SIZE = 1000
SEQ_LEN = 64
MASK_ID = 1
D_MODEL = 128
NHEAD = 4
NUM_LAYERS = 4
FF_DIM = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MultiHeadAttention(nn.Module):
    """手写多头自注意力。"""

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

    def forward(self, x):
        bsz = x.size(0)
        Q = self.w_q(x).view(bsz, -1, self.nhead, self.d_head).transpose(1, 2)
        K = self.w_k(x).view(bsz, -1, self.nhead, self.d_head).transpose(1, 2)
        V = self.w_v(x).view(bsz, -1, self.nhead, self.d_head).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_head)
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


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead)
        self.feed_forward = FeedForward(d_model, dim_feedforward)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_out = self.self_attn(x)
        x = self.norm1(x + self.dropout(attn_out))
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class BertMLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(VOCAB_SIZE, D_MODEL)
        self.pos_emb = nn.Embedding(SEQ_LEN, D_MODEL)
        self.encoder = TransformerEncoder(D_MODEL, NHEAD, NUM_LAYERS, FF_DIM)
        self.head = nn.Linear(D_MODEL, VOCAB_SIZE)

    def forward(self, x):
        pos = torch.arange(x.size(1), device=x.device)
        h = self.token_emb(x) + self.pos_emb(pos)
        h = self.encoder(h)
        return self.head(h)


def mask_tokens(inputs, mask_prob=0.15):
    """随机遮挡部分 token，返回遮挡后的输入和标签。"""
    labels = inputs.clone()
    rand = torch.rand(inputs.shape)
    mask = (rand < mask_prob) & (inputs != 0)

    inputs[mask] = MASK_ID
    labels[~mask] = -100  # 只计算被遮挡位置
    return inputs, labels


def get_synthetic_dataset(num_samples=2000):
    x = torch.randint(2, VOCAB_SIZE, (num_samples, SEQ_LEN))
    return TensorDataset(x)


def main():
    dataset = get_synthetic_dataset()
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = BertMLM().to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for (x,) in loader:
            x = x.to(DEVICE)
            inputs, labels = mask_tokens(x)

            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits.view(-1, VOCAB_SIZE), labels.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  MLM Loss: {avg_loss:.4f}")


if __name__ == "__main__":
    main()
