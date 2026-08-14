"""
05_SpeculativeDecoding.py
=========================
vLLM Speculative Decoding:用小模型(draft)生成候选 token,大模型(target)一次 forward 验证。

核心算法(Leviathan 2023):
    1. draft 模型自回归生成 K 个候选 token: t_1, t_2, ..., t_K
    2. target 模型对 [prompt + t_1...t_K] 做一次 forward,得到每个位置的
       target 概率 p_i 和 draft 概率 q_i
    3. 对每个位置 i:
         r ~ Uniform(0, 1)
         if r < min(1, p_i[t_i] / q_i[t_i]):  接受 t_i
         else:                                  拒绝,从 (p - q)_+ 归一化分布采样新 token,停止
    4. bonus token:如果全部接受,从 target 第 K+1 位置再采样一个 token(白送)

理论保证:输出分布与纯 target 采样完全相同(无偏)。

vLLM 实际还支持:
    - Medusa:多头并行预测
    - Lookahead:Jacobi 迭代,无需 draft 模型
    - EAGLE-style tree attention(主要在 SGLang)
本文实现经典 draft-model 版本。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import torch
import torch.nn.functional as F


# ============================================================
# 1. Mock 模型
# ============================================================

class MockTargetModel:
    """模拟大模型 forward,返回 logits"""

    def __init__(self, vocab_size: int = 1000, hidden: int = 64):
        self.vocab_size = vocab_size
        self.embed = torch.nn.Embedding(vocab_size, hidden)
        self.head = torch.nn.Linear(hidden, vocab_size)

    @torch.no_grad()
    def forward(self, token_ids: List[int]) -> torch.Tensor:
        """token_ids -> logits [seq_len, vocab]"""
        ids = torch.tensor(token_ids, dtype=torch.long)
        h = self.embed(ids)
        return self.head(h)  # [seq_len, vocab]


class MockDraftModel(MockTargetModel):
    """小模型,结构与 target 一致但参数更小(此处简化)"""
    pass


# ============================================================
# 2. Speculative Decoding 核心算法
# ============================================================

def sample_from_probs(probs: torch.Tensor) -> int:
    """从概率分布采样"""
    return torch.multinomial(probs, num_samples=1).item()


def speculative_decode_step(
    target: MockTargetModel,
    draft: MockDraftModel,
    prompt: List[int],
    num_draft_tokens: int = 4,
    eos_token_id: int = 2,
) -> Tuple[List[int], int]:
    """
    一轮投机解码:返回 (新生成的 token 列表, 接受的 draft token 数)
    """
    # ---- 1. Draft 模型自回归生成 K 个候选 ----
    draft_tokens: List[int] = []
    draft_probs: List[torch.Tensor] = []  # 每个位置的 q 分布
    cur_seq = list(prompt)
    for _ in range(num_draft_tokens):
        logits = draft.forward(cur_seq)
        probs = F.softmax(logits[-1], dim=-1)
        tok = sample_from_probs(probs)
        draft_tokens.append(tok)
        draft_probs.append(probs)
        cur_seq.append(tok)
        if tok == eos_token_id:
            break

    K = len(draft_tokens)

    # ---- 2. Target 模型一次 forward 验证 ----
    target_logits = target.forward(prompt + draft_tokens)  # [prompt_len + K, vocab]
    target_probs = F.softmax(target_logits, dim=-1)

    # ---- 3. 逐位置接受/拒绝 ----
    accepted = 0
    output_tokens: List[int] = []
    prompt_len = len(prompt)

    for i in range(K):
        t = draft_tokens[i]
        p = target_probs[prompt_len + i]  # target 在该位置预测下个 token 的分布
        q = draft_probs[i]

        ratio = p[t].item() / max(q[t].item(), 1e-12)
        r = torch.rand(1).item()

        if r < min(1.0, ratio):
            # 接受
            output_tokens.append(t)
            accepted += 1
            if t == eos_token_id:
                return output_tokens, accepted
        else:
            # 拒绝:从 (p - q)_+ 归一化分布采样新 token
            adjusted = (p - q).clamp(min=0)
            adjusted = adjusted / adjusted.sum()
            new_tok = sample_from_probs(adjusted)
            output_tokens.append(new_tok)
            return output_tokens, accepted

    # ---- 4. Bonus token: 全部接受时,白送一个 target 在最后位置的预测 ----
    bonus_probs = target_probs[-1]  # target 在 prompt + K 位置的预测
    bonus_tok = sample_from_probs(bonus_probs)
    output_tokens.append(bonus_tok)

    return output_tokens, accepted


# ============================================================
# 3. 完整生成循环
# ============================================================

def generate(target: MockTargetModel,
             draft: MockDraftModel,
             prompt: List[int],
             max_new_tokens: int = 32,
             num_draft_tokens: int = 4,
             eos_token_id: int = 2) -> List[int]:
    output = list(prompt)
    total_accepted = 0
    total_steps = 0

    while len(output) - len(prompt) < max_new_tokens:
        new_tokens, accepted = speculative_decode_step(
            target, draft, output, num_draft_tokens, eos_token_id
        )
        output.extend(new_tokens)
        total_accepted += accepted
        total_steps += 1
        if new_tokens[-1] == eos_token_id:
            break

    print(f"SpecDec: generated {len(output)-len(prompt)} tokens in {total_steps} steps, "
          f"acceptance rate = {total_accepted / (total_steps * num_draft_tokens):.2%}")
    return output[len(prompt):]


# ============================================================
# 4. 演示
# ============================================================

def demo():
    torch.manual_seed(42)
    target = MockTargetModel(vocab_size=1000, hidden=64)
    draft = MockDraftModel(vocab_size=1000, hidden=32)

    # 简单复制 target 的部分权重作为 draft(演示用)
    # 实际中 draft 是独立训练的小模型
    draft.embed.weight.data = target.embed.weight.data.clone()
    draft.head.weight.data = target.head.weight.data.clone()
    draft.head.bias.data = target.head.bias.data.clone()

    prompt = [10, 20, 30, 40, 50]
    output = generate(target, draft, prompt,
                      max_new_tokens=20,
                      num_draft_tokens=4)
    print(f"Prompt: {prompt}")
    print(f"Generated: {output}")


if __name__ == "__main__":
    demo()
