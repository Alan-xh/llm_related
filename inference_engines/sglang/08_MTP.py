"""
08_MTP.py
=========
SGLang DeepSeek MTP(Multi-Token Prediction):训练时每个层预测多个未来 token,
推理时作为 speculative decoding 的 proposer。

DeepSeek-V3 的 MTP:
    - 训练时:深度 MTP,每个 MTP module 预测下一个 token,可堆叠多层
    - 推理时:把 MTP module 作为 draft proposer,一次产出多 token
    - 与 EAGLE-like tree attention 结合验证

架构:
    主模型 forward -> hidden state h_1
    MTP module 1:输入 h_1 + embed(t_1) -> 预测 t_2
    MTP module 2:输入 h_2 + embed(t_2) -> 预测 t_3
    ...

推理流程(作为 specDec proposer):
    1. 主模型 forward 产出 hidden state + next token
    2. MTP module 串行产出 K 个 draft token
    3. 主模型一次 forward 验证整棵 draft sequence
    4. 接受最长合法前缀

优势 vs 独立 draft 模型:
    - 共享主模型权重(无额外大模型)
    - MTP module 很轻量(1 层 transformer)
    - 训练时已经学到多 token 预测,acceptance 高
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. 主模型(简化)
# ============================================================

class MainModel(nn.Module):
    """模拟 DeepSeek 主模型,返回 logits + 最后一层 hidden state"""

    def __init__(self, vocab_size: int = 1000, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden, nhead=4,
                                       dim_feedforward=hidden*2,
                                       batch_first=True)
            for _ in range(n_layers)
        ])
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.hidden = hidden

    def forward(self, token_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.embed(token_ids)
        for layer in self.layers:
            h = layer(h)
        return self.head(h), h  # logits, last_hidden


# ============================================================
# 2. MTP Module(单层)
# ============================================================

class MTPModule(nn.Module):
    """
    单个 MTP module:
        输入:主模型 hidden state + 上一个 token 的 embedding
        输出:下一个 token 的 logits + 自身的 hidden state(可串联下一 module)
    """

    def __init__(self, vocab_size: int, hidden: int = 256):
        super().__init__()
        # 融合 main hidden 和 token embedding
        self.fuse_norm = nn.LayerNorm(hidden * 2)
        self.fuse_proj = nn.Linear(hidden * 2, hidden)
        # 单层 transformer
        self.transformer = nn.TransformerEncoderLayer(
            d_model=hidden, nhead=4,
            dim_feedforward=hidden*2, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden)
        self.head = nn.Linear(hidden, vocab_size, bias=False)
        self.embed = nn.Embedding(vocab_size, hidden)  # 共享主模型 embed 实际

    def forward(self, main_hidden: torch.Tensor,
                prev_token_id: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        main_hidden: [hidden]  (主模型最后位置的 hidden)
        prev_token_id: [1] scalar tensor
        return: (logits [vocab], new_hidden [hidden])
        """
        tok_emb = self.embed(prev_token_id)  # [hidden]
        # 拼接 + 融合
        fused = torch.cat([main_hidden, tok_emb], dim=-1)  # [2*hidden]
        fused = self.fuse_norm(fused)
        h = self.fuse_proj(fused).unsqueeze(0)  # [1, hidden]
        h = self.transformer(h)
        h = self.norm(h).squeeze(0)
        logits = self.head(h)
        return logits, h


# ============================================================
# 3. 完整 MTP 推理(speculative decoding proposer)
# ============================================================

class MTPProposer:
    """
    串联多个 MTP module 产出 K 个 draft token。
    """

    def __init__(self, main_model: MainModel, num_mtp_modules: int = 4):
        self.main_model = main_model
        self.mtp_modules = nn.ModuleList([
            MTPModule(vocab_size=main_model.head.out_features,
                      hidden=main_model.hidden)
            for _ in range(num_mtp_modules)
        ])
        self.num_mtp = num_mtp_modules

    @torch.no_grad()
    def generate_draft(self, token_ids: List[int]) -> Tuple[List[int], torch.Tensor]:
        """
        主模型 forward + MTP modules 产出 K 个 draft token。
        return: (draft_tokens [K], main_logits)
        """
        # 1. 主模型 forward
        input_ids = torch.tensor(token_ids, dtype=torch.long)
        logits, hidden = self.main_model.forward(input_ids)
        next_token = logits[-1].argmax().item()

        # 2. MTP modules 串联产出 draft
        draft_tokens = [next_token]
        cur_hidden = hidden[-1]  # 主模型最后位置 hidden
        cur_token = torch.tensor([next_token], dtype=torch.long)

        for mtp in self.mtp_modules:
            mtp_logits, new_hidden = mtp.forward(cur_hidden, cur_token)
            next_draft = mtp_logits.argmax().item()
            draft_tokens.append(next_draft)
            cur_hidden = new_hidden
            cur_token = torch.tensor([next_draft], dtype=torch.long)

        return draft_tokens, logits


# ============================================================
# 4. Speculative Verification(主模型一次 forward 验证)
# ============================================================

@torch.no_grad()
def verify_draft(main_model: MainModel,
                 token_ids: List[int],
                 draft_tokens: List[int]) -> List[int]:
    """
    主模型对 [token_ids + draft_tokens] 一次 forward,
    比较每个位置的 argmax 是否与 draft 一致。
    接受最长合法前缀(简化:不做概率接受)。
    """
    all_ids = token_ids + draft_tokens
    input_ids = torch.tensor(all_ids, dtype=torch.long)
    logits, _ = main_model.forward(input_ids)

    # 主模型在每个位置预测的 next token
    predicted = logits[len(token_ids)-1:-1].argmax(dim=-1).tolist()

    # 找最长匹配前缀
    accepted = []
    for i, (draft, pred) in enumerate(zip(draft_tokens, predicted)):
        if draft == pred:
            accepted.append(draft)
        else:
            # 拒绝,用主模型该位置的预测替代
            accepted.append(pred)
            break
    else:
        # 全部接受,bonus token(主模型最后位置预测)
        bonus = logits[-1].argmax().item()
        accepted.append(bonus)

    return accepted


# ============================================================
# 5. 完整 MTP 推理循环
# ============================================================

def mtp_generate(proposer: MTPProposer, prompt: List[int],
                 max_new_tokens: int = 30) -> List[int]:
    output = list(prompt)
    total_accepted = 0
    total_steps = 0
    K = proposer.num_mtp

    while len(output) - len(prompt) < max_new_tokens:
        # 1. MTP 产出 draft
        draft_tokens, _ = proposer.generate_draft(output)

        # 2. 主模型验证
        accepted = verify_draft(proposer.main_model, output, draft_tokens)

        output.extend(accepted)
        total_accepted += len(accepted)
        total_steps += 1

        # EOS 检测(简化)
        if accepted[-1] == 2:
            break

    rate = total_accepted / max(1, total_steps * (K + 1))
    print(f"MTP: generated {len(output)-len(prompt)} tokens in {total_steps} steps, "
          f"acceptance rate = {rate:.2%}")
    return output[len(prompt):]


# ============================================================
# 6. 演示
# ============================================================

def demo():
    torch.manual_seed(42)
    main = MainModel(vocab_size=100, hidden=64, n_layers=2)
    proposer = MTPProposer(main, num_mtp_modules=3)

    # 让 MTP modules 模仿主模型(简化训练)
    for mtp in proposer.mtp_modules:
        mtp.head.weight.data = main.head.weight.data.clone()
        mtp.embed.weight.data = main.embed.weight.data.clone()

    prompt = [10, 20, 30, 40, 50]
    print(f"Prompt: {prompt}")
    out = mtp_generate(proposer, prompt, max_new_tokens=20)
    print(f"Generated: {out}")


if __name__ == "__main__":
    demo()
