"""
06_EAGLE.py
===========
SGLang EAGLE 投机解码:用 LLM 第二层 hidden state 训练一个小 head 预测多 token,
配合 draft tree + tree attention 实现高 acceptance rate。

对比 vLLM 的 Medusa:
    - Medusa:在最后 hidden state 上接多个 head,各预测下下个 token
    - EAGLE:用第二层 hidden state(更深语义)作为 draft 模型输入,自回归生成 draft tree
    - EAGLE-2:动态构建 draft tree(基于概率剪枝)
    - EAGLE-3:改进训练目标 + 更深特征融合

核心算法:
    1. Target LLM forward 一次,得到 hidden states
    2. EAGLE head(一个小 transformer)以 hidden state 为输入,自回归生成 K 个 draft token
    3. 构建候选 token 树(每个 draft token 派生多个候选)
    4. Target LLM 一次 forward 验证整棵树(tree attention)
    5. 接受最长合法前缀

Tree Attention:
    - 把 draft tree 编码为 attention mask
    - 每个候选 token 只能看到其祖先 token
    - 一次 forward 同时验证所有候选
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Mock Target Model
# ============================================================

class TargetModel(nn.Module):
    """模拟 LLM,返回 logits + hidden states"""

    def __init__(self, vocab_size: int = 1000, hidden: int = 128, n_layers: int = 4):
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
        """token_ids: [seq] -> (logits [seq, vocab], hidden [seq, hidden])"""
        h = self.embed(token_ids)
        for layer in self.layers:
            h = layer(h)
        return self.head(h), h


# ============================================================
# 2. EAGLE Draft Head
# ============================================================

class EAGLEHead(nn.Module):
    """
    EAGLE draft model:小 transformer,以 target 第二层 hidden state 为输入,
    自回归生成 draft token。

    输入:target hidden state + 上一个 draft token embedding
    输出:draft token logits
    """

    def __init__(self, vocab_size: int, hidden: int = 128, num_layers: int = 2):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden)
        # 融合 hidden state 和 token embedding
        self.fuse = nn.Linear(hidden * 2, hidden)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=hidden, nhead=4,
                                       dim_feedforward=hidden*2,
                                       batch_first=True)
            for _ in range(num_layers)
        ])
        self.head = nn.Linear(hidden, vocab_size, bias=False)

    def forward_step(self, hidden_state: torch.Tensor,
                     prev_token: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        单步:输入 (target hidden, prev token) -> 输出 (next token logits, draft hidden)
        """
        tok_emb = self.embed(prev_token)
        fused = self.fuse(torch.cat([hidden_state, tok_emb], dim=-1))
        h = fused.unsqueeze(0)  # [1, hidden]
        for layer in self.layers:
            h = layer(h)
        logits = self.head(h.squeeze(0))
        return logits, h.squeeze(0)


# ============================================================
# 3. Draft Tree 构建
# ============================================================

@dataclass
class DraftNode:
    token: int
    parent_id: int   # 在 tree 列表中的 index
    prob: float
    depth: int


def build_draft_tree(eagle: EAGLEHead,
                     target_hidden: torch.Tensor,
                     last_token: int,
                     num_draft_tokens: int = 4,
                     top_k_branches: int = 2) -> List[DraftNode]:
    """
    EAGLE-2 风格:动态构建 draft tree
        - 每步生成 top-k 个候选(不只 argmax)
        - 用概率剪枝控制树大小
    """
    tree: List[DraftNode] = [DraftNode(token=last_token, parent_id=-1, prob=1.0, depth=0)]
    cur_hidden = target_hidden[-1]  # 取最后位置的 hidden
    cur_token = torch.tensor([last_token], dtype=torch.long)

    frontier = [(0, cur_hidden, cur_token)]  # (node_id, hidden, token)
    for depth in range(1, num_draft_tokens + 1):
        new_frontier = []
        for node_id, h, tok in frontier:
            logits, new_h = eagle.forward_step(h, tok)
            probs = F.softmax(logits, dim=-1)
            topk_probs, topk_ids = probs.topk(top_k_branches)

            for p, t in zip(topk_probs[0], topk_ids[0]):
                if p.item() < 0.05:  # 概率阈值剪枝
                    continue
                child = DraftNode(token=t.item(), parent_id=node_id,
                                  prob=p.item(), depth=depth)
                tree.append(child)
                new_frontier.append((len(tree)-1, new_h, t))
        frontier = new_frontier
        if not frontier:
            break

    return tree


# ============================================================
# 4. Tree Attention
# ============================================================

def build_tree_attention_mask(tree: List[DraftNode]) -> torch.Tensor:
    """
    构造 attention mask:每个节点只能看到其祖先链。
    mask[i, j] = 0 if j 是 i 的祖先(含自身),否则 -inf
    """
    n = len(tree)
    mask = torch.full((n, n), float("-inf"))

    # 计算每个节点的祖先集
    ancestors: List[set] = []
    for i, node in enumerate(tree):
        anc = {i}
        if node.parent_id >= 0:
            anc |= ancestors[node.parent_id]
        ancestors.append(anc)
        for j in anc:
            mask[i, j] = 0

    return mask


def tree_verify(target: TargetModel,
                tree: List[DraftNode],
                prompt: List[int]) -> Tuple[List[int], int]:
    """
    Target LLM 一次 forward 验证整棵 draft tree。
    返回 (接受的 token 列表, 接受数)
    """
    # 把 prompt + tree tokens 作为输入
    all_tokens = prompt + [n.token for n in tree]
    input_ids = torch.tensor(all_tokens, dtype=torch.long)
    logits, _ = target.forward(input_ids)

    # 构造 tree attention mask(简化:只用因果 mask)
    # 实际 EAGLE 用 tree mask 限制每个 draft token 只看祖先
    tree_mask = build_tree_attention_mask(tree)

    # 验证:每个节点用 target logits 检查
    accepted: List[int] = []
    cur_parent = -1  # 从 root 开始
    accepted_count = 0

    # BFS 找最长合法路径
    target_probs = F.softmax(logits[len(prompt)-1:], dim=-1)  # tree 部分的 logits

    best_path: List[int] = []
    best_len = 0

    def dfs(node_idx: int, path: List[int]):
        nonlocal best_path, best_len
        node = tree[node_idx]
        if node.parent_id != cur_parent_recursive(node_idx, path):
            return
        # 检查这个节点的 token 是否被 target 接受
        t_prob = target_probs[node_idx, node.token].item()
        if t_prob < 0.01:  # 拒绝阈值
            return
        path.append(node.token)
        if len(path) > best_len:
            best_path = list(path)
            best_len = len(path)
        # 递归子节点
        for i, n in enumerate(tree):
            if n.parent_id == node_idx:
                dfs(i, path)
        path.pop()

    def cur_parent_recursive(idx, path):
        return -1 if not path else tree[idx].parent_id

    dfs(0, [])
    return best_path, len(best_path)


# ============================================================
# 5. 完整 EAGLE 推理
# ============================================================

def eagle_generate(target: TargetModel,
                   eagle: EAGLEHead,
                   prompt: List[int],
                   max_new_tokens: int = 20,
                   num_draft: int = 4) -> List[int]:
    output = list(prompt)
    total_accepted = 0
    total_steps = 0

    while len(output) - len(prompt) < max_new_tokens:
        # 1. Target forward 一次
        input_ids = torch.tensor(output, dtype=torch.long)
        logits, hidden = target.forward(input_ids)
        next_token = logits[-1].argmax().item()
        # 简化:不用采样,直接 argmax

        # 2. EAGLE 生成 draft tree
        tree = build_draft_tree(eagle, hidden, next_token,
                                num_draft_tokens=num_draft, top_k_branches=2)

        # 3. Tree verify
        accepted, n = tree_verify(target, tree, output)
        output.extend(accepted)
        total_accepted += n
        total_steps += 1

        if not accepted:
            break

    rate = total_accepted / max(1, total_steps * num_draft)
    print(f"EAGLE: generated {len(output)-len(prompt)} tokens in {total_steps} steps, "
          f"acceptance rate = {rate:.2%}")
    return output[len(prompt):]


# ============================================================
# 6. 演示
# ============================================================

def demo():
    torch.manual_seed(42)
    target = TargetModel(vocab_size=1000, hidden=128, n_layers=4)
    eagle = EAGLEHead(vocab_size=1000, hidden=128, num_layers=2)

    # 模拟训练:让 eagle 模仿 target 的下一 token(简化)
    # 实际 EAGLE 需要在数据集上训练
    eagle.embed.weight.data = target.embed.weight.data.clone()
    eagle.head.weight.data = target.head.weight.data.clone()

    prompt = [10, 20, 30, 40, 50]
    print(f"Prompt: {prompt}")
    out = eagle_generate(target, eagle, prompt, max_new_tokens=20, num_draft=4)
    print(f"Generated: {out}")


if __name__ == "__main__":
    demo()
