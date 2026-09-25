"""Kimi 系列解码器模型的可运行教学组件。

任务定义:
    实现面向因果语言建模的最小 Kimi 风格 Decoder-only 模型，支持训练、
    增量推理和每层缓存复用。该文件同时提供 Kimi K2 的 MLA/MoE 组件与
    Kimi Linear 的 KDA/MLA 混合组件。

代表架构/算法:
    Multi-head Latent Attention（MLA）、Kimi Delta Attention（KDA）、
    RoPE、SwiGLU、Top-k Mixture-of-Experts（MoE）。

核心思想与数据流:
    input_ids -> 词嵌入 -> Pre-Norm 注意力 -> 残差连接 -> MoE/SwiGLU
    -> RMSNorm -> 语言模型头。MLA 缓存完整 K/V，KDA 只缓存有限递归状态，
    因而增量生成阶段只需处理新输入 token。

数学目标:
    L_total = L_next_token + 0.01 * L_MoE_aux
    Attention(Q, K, V) = softmax(QK^T / sqrt(D))V
    KDA: retrieved = qS, delta = v - kS,
         S <- decay * S + gate * k^T delta

输入输出规范:
    input_ids: [B, T]，整数 token id。
    hidden_states: [B, T, H]，H 为隐藏维度。
    logits: [B, T, V]，V 为词表大小。
    MLA cache: (key, value)，每项 shape 为 [B, heads, T_cache, D]。
    KDA cache: (state, position)，state shape 为 [B, heads, D, D]。

说明:
    本实现用于 CPU smoke test、Shape 追踪和架构教学，不兼容官方 tokenizer、
    checkpoint 或推理 kernel。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelOutput:
    """训练和增量生成共用的解码器输出。

    属性:
        logits: 预测下一个 token 的未归一化分数，shape: [B, T, V]。
        loss: 可选的总损失，标量 Tensor。
        past_key_values: 可选的逐层缓存，MLA 或 KDA 的缓存类型取决于层。
        aux_loss: 可选的 MoE 负载均衡辅助损失，标量 Tensor。
    """

    logits: Tensor
    loss: Optional[Tensor] = None
    past_key_values: Optional[tuple[Any, ...]] = None
    aux_loss: Optional[Tensor] = None


@dataclass
class KimiConfig:
    """描述 MLA、KDA、混合注意力与 MoE 变体的统一配置。

    关键参数:
        hidden_size (int): 隐藏维度 H。
        num_layers (int): Decoder 层数。
        num_heads (int): 注意力头数。
        attention_type (str): ``mla``、``kda`` 或 ``hybrid``。
        kda_ratio (int): 混合模式中连续 KDA 层的数量。
        num_experts (int): MoE 路由专家数量。
        num_experts_per_tok (int): 每个 token 选取的 top-k 专家数。
    """

    vocab_size: int = 259
    hidden_size: int = 128
    intermediate_size: int = 256
    num_layers: int = 4
    num_heads: int = 4
    head_dim: Optional[int] = None
    max_position_embeddings: int = 2048
    rope_theta: float = 1_000_000.0
    rms_norm_eps: float = 1e-6
    q_lora_rank: int = 64
    kv_lora_rank: int = 64
    qk_rope_dim: int = 32
    attention_type: str = "mla"
    kda_ratio: int = 3
    use_moe: bool = True
    num_experts: int = 8
    num_experts_per_tok: int = 2
    shared_expert: bool = True
    dropout: float = 0.0
    tie_word_embeddings: bool = True

    def __post_init__(self) -> None:
        """在构造模块前校验维度、注意力类型和专家路由参数。"""
        if self.head_dim is None:
            if self.hidden_size % self.num_heads:
                raise ValueError("hidden_size must be divisible by num_heads")
            self.head_dim = self.hidden_size // self.num_heads
        if self.head_dim % 2:
            raise ValueError("head_dim must be even for RoPE")
        if self.qk_rope_dim <= 0 or self.qk_rope_dim % 2:
            raise ValueError("qk_rope_dim must be a positive even number")
        if self.qk_rope_dim > self.head_dim:
            raise ValueError("qk_rope_dim cannot exceed head_dim")
        if self.attention_type not in {"mla", "kda", "hybrid"}:
            raise ValueError("attention_type must be 'mla', 'kda', or 'hybrid'")
        if self.kda_ratio <= 0:
            raise ValueError("kda_ratio must be positive")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")


class ByteTokenizer:
    """不依赖外部词表的 UTF-8 字节级 tokenizer。

    特殊 token 占用 id 0、1、2，其余 UTF-8 字节映射到 ``[3, 258]``。
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        """将文本编码为 token id 序列。

        参数:
            text (str): 输入文本。
            add_bos (bool): 是否在开头添加 BOS。
            add_eos (bool): 是否在末尾添加 EOS。

        输入:
            text: Python 字符串。

        输出:
            list[int]: 长度为 ``len(text.encode("utf-8"))`` 加特殊 token 数量的
            一维 token id 序列。
        """
        ids = [byte + 3 for byte in text.encode("utf-8")]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """将字节 token id 序列还原为文本。

        参数:
            ids (Sequence[int]): 一维 token id 序列。

        输出:
            str: 忽略特殊 token 后解码得到的 UTF-8 文本；非法字节使用替换符。
        """
        values = [token - 3 for token in ids if 3 <= token < self.vocab_size]
        return bytes(values).decode("utf-8", errors="replace")


def build_training_batch(
    tokenizer: ByteTokenizer,
    text: str,
    batch_size: int,
    seq_len: int,
    step: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """构造确定性的重复文本因果语言模型 batch。

    参数:
        tokenizer (ByteTokenizer): 字节级 tokenizer。
        text (str): 用于生成训练样本的文本。
        batch_size (int): batch 大小 B。
        seq_len (int): 输入序列长度 T。
        step (int): 用于滚动样本起点的训练步数。
        device (torch.device): Tensor 所在设备。

    输出:
        tuple[Tensor, Tensor]: ``inputs`` 与 ``labels``，二者 shape 均为
        ``[B, T]``，其中 labels 是 inputs 向右平移一位的下一个 token。
    """
    if not text:
        raise ValueError("training text must not be empty")
    ids = tokenizer.encode(text, add_bos=True, add_eos=True)
    repeated = (ids * ((seq_len + 1) // len(ids) + 1))[: seq_len + 1]
    values = torch.tensor(repeated, dtype=torch.long, device=device)
    values = values.roll(shifts=step % max(len(values), 1))
    tokens = values.unsqueeze(0).expand(batch_size, -1).contiguous()
    return tokens[:, :-1], tokens[:, 1:]


class RMSNorm(nn.Module):
    """沿最后一维执行 RMSNorm。

    对输入 ``x`` 使用公式
    ``RMSNorm(x) = w * x / sqrt(mean(x^2) + eps)``，不改变输入 Shape。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """归一化输入张量。

        输入:
            x (Tensor): 任意前缀维度、最后一维为 H，shape: ``[..., H]``。

        输出:
            Tensor: 与输入相同 shape 的归一化结果，shape: ``[..., H]``。
        """
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps).to(x.dtype))


def rotate_half(x: Tensor) -> Tensor:
    """交换最后一维中前后两半并取负，保持输入 Shape 不变。"""
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    """为 ``[B, heads, T, D]`` 张量提供 RoPE 旋转位置编码缓存。

    位置角频率为 ``theta^(-2i / D)``，输出通过
    ``x * cos(position) + rotate_half(x) * sin(position)`` 得到。
    """

    def __init__(self, dim: int, max_position_embeddings: int, theta: float) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("rotary dimension must be even")
        self.dim = dim
        self.theta = theta
        self.register_buffer(
            "inv_freq",
            1.0 / theta ** (torch.arange(0, dim, 2).float() / dim),
            persistent=False,
        )
        self._set_cache(max_position_embeddings)

    def _set_cache(self, length: int) -> None:
        """按给定最大长度重建 RoPE 的正弦和余弦缓存。

        参数:
            length (int): 缓存位置数量。

        缓存 Shape:
            ``_cos`` 与 ``_sin`` 均为 ``[1, 1, length, dim]``。
        """
        positions = torch.arange(length, device=self.inv_freq.device).float()
        frequencies = torch.outer(positions, self.inv_freq)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("_cos", embeddings.cos()[None, None], persistent=False)
        self.register_buffer("_sin", embeddings.sin()[None, None], persistent=False)

    def forward(self, x: Tensor, position_offset: int = 0) -> Tensor:
        """将 RoPE 应用于输入张量并保持 Shape 不变。

        输入:
            x (Tensor): 多头张量，shape: ``[B, heads, T, D]``。
            position_offset (int): 当前序列在缓存中的起始位置。

        输出:
            Tensor: 旋转后张量，shape: ``[B, heads, T, D]``。
        """
        length = x.shape[-2]
        needed = position_offset + length
        if needed > self._cos.shape[-2]:
            self._set_cache(max(needed, self._cos.shape[-2] * 2))
        cos = self._cos[..., position_offset:needed, :].to(dtype=x.dtype)
        sin = self._sin[..., position_offset:needed, :].to(dtype=x.dtype)
        return x * cos + rotate_half(x) * sin


def repeat_kv(hidden_states: Tensor, repeats: int) -> Tensor:
    """复制 KV 头以实现分组查询注意力。

    输入:
        hidden_states: shape ``[B, heads_kv, T, D]``。
    输出:
        Tensor: shape ``[B, heads_kv * repeats, T, D]``。
    """
    if repeats == 1:
        return hidden_states
    batch, heads, length, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None].expand(
        batch, heads, repeats, length, head_dim
    )
    # [B, heads_kv, repeats, T, D] -> [B, heads_kv * repeats, T, D]
    return expanded.reshape(batch, heads * repeats, length, head_dim)


class MLAAttention(nn.Module):
    """紧凑的 Multi-head Latent Attention（MLA）实现。

    ``kv_down`` 将 K/V 内容压缩到 latent，再通过 ``k_up`` 与 ``v_up`` 恢复
    多头内容；查询和 key 的 RoPE 分量独立计算。注意力公式为
    ``Attention(Q, K, V) = softmax(QK^T / sqrt(D))V``。为使增量推理接口
    直观，缓存保存展开后的 K/V，而不是仅保存 latent。

    输入输出:
        hidden_states: ``[B, T, H]`` -> ``[B, T, H]``。
        MLA cache: 每项 ``[B, heads, T_cache, D]``。
    """

    def __init__(self, config: KimiConfig) -> None:
        super().__init__()
        assert config.head_dim is not None
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.qk_rope_dim = config.qk_rope_dim
        self.q_lora = nn.Linear(config.hidden_size, config.q_lora_rank, bias=False)
        self.q_up = nn.Linear(
            config.q_lora_rank, config.num_heads * config.head_dim, bias=False
        )
        self.kv_down = nn.Linear(
            config.hidden_size, config.kv_lora_rank, bias=False
        )
        self.k_up = nn.Linear(
            config.kv_lora_rank, config.num_heads * config.head_dim, bias=False
        )
        self.v_up = nn.Linear(
            config.kv_lora_rank, config.num_heads * config.head_dim, bias=False
        )
        self.q_rope = nn.Linear(config.hidden_size, config.num_heads * config.qk_rope_dim, bias=False)
        self.k_rope = nn.Linear(config.hidden_size, config.qk_rope_dim, bias=False)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(
            config.qk_rope_dim, config.max_position_embeddings, config.rope_theta
        )
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: Tensor,
        past_key_value: Optional[tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        """执行带因果掩码的 MLA。

        参数:
            hidden_states (Tensor): 输入隐藏状态，shape: ``[B, T, H]``。
            past_key_value (tuple, optional): 历史 K/V，单项 shape:
                ``[B, heads, T_cache, D]``。
            use_cache (bool): 是否返回拼接后的 K/V 缓存。

        输出:
            tuple[Tensor, tuple | None]: 输出 shape 为 ``[B, T, H]``；
            缓存中的 K/V shape 为 ``[B, heads, T_cache + T, D]``。
        """
        batch, length, _ = hidden_states.shape
        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]

        query = self.q_up(self.q_lora(hidden_states)).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # [B, T, q_lora_rank] -> [B, heads, T, D]
        key = self.k_up(self.kv_down(hidden_states)).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # [B, T, kv_lora_rank] -> [B, heads, T, D]
        value = self.v_up(self.kv_down(hidden_states)).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # [B, T, kv_lora_rank] -> [B, heads, T, D]
        query = self.q_norm(query)
        key = self.k_norm(key)

        query_rope = self.q_rope(hidden_states).view(
            batch, length, self.num_heads, self.qk_rope_dim
        ).transpose(1, 2)
        # [B, T, H] -> [B, heads, T, qk_rope_dim]
        key_rope = self.k_rope(hidden_states).view(
            batch, length, 1, self.qk_rope_dim
        ).transpose(1, 2).expand(-1, self.num_heads, -1, -1)
        # [B, T, qk_rope_dim] -> [B, heads, T, qk_rope_dim]
        query_rope = self.rotary(query_rope, past_length)
        key_rope = self.rotary(key_rope, past_length)
        query = torch.cat((query[..., :-self.qk_rope_dim], query_rope), dim=-1)
        key = torch.cat((key[..., :-self.qk_rope_dim], key_rope), dim=-1)
        # 内容分量与 RoPE 分量拼接，仍为 [B, heads, T, D]

        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)
            # [B, heads, T_cache, D] + [B, heads, T, D] -> [B, heads, T_cache + T, D]
        total_length = key.shape[-2]
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        # 注意力分数 QK^T / sqrt(D): [B, heads, T, D] x [B, heads, D, T_total]
        # -> [B, heads, T, T_total]
        if length > 1 or past_length == 0:
            query_positions = torch.arange(
                past_length, past_length + length, device=hidden_states.device
            )
            key_positions = torch.arange(total_length, device=hidden_states.device)
            future = key_positions[None, :] > query_positions[:, None]
            scores = scores.masked_fill(
                future[None, None], torch.finfo(scores.dtype).min
            )
        weights = F.softmax(scores.float(), dim=-1).to(query.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.matmul(weights, value).transpose(1, 2).contiguous()
        # [B, heads, T, T_total] x [B, heads, T_total, D]
        # -> [B, T, heads, D]
        output = self.o_proj(output.view(batch, length, -1))
        # [B, T, heads * D] -> [B, T, H]
        present = (key, value) if use_cache else None
        return output, present


class KDAttention(nn.Module):
    """带门控 Delta Rule 的有限状态递归注意力。

    状态 ``S`` 的 shape 为 ``[B, heads, D, D]``。对每个 token 执行:

    ``retrieved = q @ S``
    ``prediction = k @ S``
    ``delta = v - prediction``
    ``S <- decay * S + gate * outer(k, delta)``

    这是 Kimi Delta Attention 的可读教学近似，使用 token 循环展示递归更新，
    不等同于生产环境中的 chunkwise kernel。
    """

    def __init__(self, config: KimiConfig) -> None:
        super().__init__()
        assert config.head_dim is not None
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.num_heads * config.head_dim, bias=False)
        self.decay_proj = nn.Linear(config.hidden_size, config.num_heads, bias=True)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps)
        self.rotary = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.o_proj = nn.Linear(config.num_heads * config.head_dim, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: Tensor,
        past_state: Optional[tuple[Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Optional[tuple[Tensor]]]:
        """执行 KDA 递归更新并可选返回有限状态缓存。

        参数:
            hidden_states (Tensor): 输入隐藏状态，shape: ``[B, T, H]``。
            past_state (tuple, optional): ``(state, position)``；state shape:
                ``[B, heads, D, D]``。
            use_cache (bool): 是否返回更新后的状态和位置。

        输出:
            tuple[Tensor, tuple | None]: 输出 shape 为 ``[B, T, H]``；
            返回状态 shape 为 ``[B, heads, D, D]``。
        """
        batch, length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # [B, T, H] -> [B, heads, T, D]
        key = self.k_proj(hidden_states).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # [B, T, H] -> [B, heads, T, D]
        value = self.v_proj(hidden_states).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # [B, T, H] -> [B, heads, T, D]
        past_length = 0
        if past_state is not None:
            state = past_state[0]
            past_length = int(past_state[1]) if len(past_state) > 1 else 0
        else:
            state = torch.zeros(
                batch,
                self.num_heads,
                self.head_dim,
                self.head_dim,
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
        query = self.rotary(self.q_norm(query), past_length)
        key = self.rotary(self.k_norm(key), past_length)
        gate = torch.sigmoid(self.gate_proj(hidden_states)).view(
            batch, length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        # [B, T, H] -> [B, heads, T, D]
        decay = torch.sigmoid(self.decay_proj(hidden_states)).transpose(1, 2).unsqueeze(-1)
        # [B, T, heads] -> [B, heads, T, 1]

        outputs = []
        for index in range(length):
            q_t = query[:, :, index]
            k_t = key[:, :, index]
            v_t = value[:, :, index]
            # 当前 token: q_t、k_t、v_t 均为 [B, heads, D]
            retrieved = torch.einsum("bhd,bhde->bhe", q_t, state)
            prediction = torch.einsum("bhd,bhde->bhe", k_t, state)
            # 查询/键与状态 S 相乘: [B, heads, D] x [B, heads, D, D] -> [B, heads, D]
            delta = v_t - prediction
            update = torch.einsum("bhd,bhe->bhde", k_t, delta)
            # 外积 k^T delta: [B, heads, D] x [B, heads, D] -> [B, heads, D, D]
            state = (
                state * decay[:, :, index].unsqueeze(-1)
                + gate[:, :, index].unsqueeze(-1) * update
            )
            # 教学实现用 tanh 限制递归状态范围，降低随机初始化模型长序列生成时
            # 的数值发散风险；生产实现通常使用归一化递归 kernel。
            state = torch.tanh(state)
            outputs.append(retrieved + gate[:, :, index] * delta)
        output = torch.stack(outputs, dim=2).transpose(1, 2).contiguous()
        # [B, heads, T, D] -> [B, T, heads, D]
        output = self.o_proj(output.view(batch, length, -1))
        # [B, T, heads * D] -> [B, T, H]
        present = (state, past_length + length) if use_cache else None
        return output, present


class SwiGLU(nn.Module):
    """SwiGLU 前馈网络。

    计算公式为 ``Down(SiLU(Gate(x)) * Up(x))``，输入输出 Shape 保持为
    ``[B, T, H]``。
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """执行 SwiGLU 变换。

        输入:
            x (Tensor): 隐藏状态，shape: ``[..., H]``。
        输出:
            Tensor: 变换后的隐藏状态，shape: ``[..., H]``。
        """
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TopKMoE(nn.Module):
    """带 top-k 路由和共享专家的稠密教学版 MoE。

    路由器先计算 ``p = softmax(router(x))``，再选取 top-k 专家并归一化权重：
    ``y = sum_i w_i * Expert_i(x) + SharedExpert(x)``。为便于观察数据流，
    教学实现会计算所有小专家，但只有被选中的专家产生有效贡献。
    """

    def __init__(self, config: KimiConfig) -> None:
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(config.hidden_size, config.intermediate_size) for _ in range(config.num_experts)]
        )
        self.shared = (
            SwiGLU(config.hidden_size, config.intermediate_size)
            if config.shared_expert
            else None
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """执行专家路由并计算简化负载均衡损失。

        输入:
            x (Tensor): 隐藏状态，shape: ``[B, T, H]`` 或 ``[N, H]``。

        输出:
            tuple[Tensor, Tensor]: 路由结果 shape 与 x 相同；辅助损失为标量。
        """
        original_shape = x.shape
        flat = x.reshape(-1, original_shape[-1])
        # [B, T, H] -> [B*T, H]
        router_logits = self.router(flat)
        probabilities = F.softmax(router_logits, dim=-1)
        weights, indices = torch.topk(
            probabilities, self.num_experts_per_tok, dim=-1
        )
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        routed = torch.zeros_like(flat)
        # 计算所有小专家以保持数据流直观；selected * weights 只保留 top-k 贡献。
        for expert_id, expert in enumerate(self.experts):
            expert_output = expert(flat)
            selected = (indices == expert_id).to(expert_output.dtype)
            contribution = (selected * weights).sum(dim=-1, keepdim=True)
            routed = routed + contribution * expert_output
        if self.shared is not None:
            routed = routed + self.shared(flat)

        mean_probability = probabilities.mean(dim=0)
        assignments = F.one_hot(indices, self.num_experts).float().mean(dim=(0, 1))
        aux_loss = self.num_experts * (mean_probability * assignments).sum()
        routed = routed.reshape(original_shape)
        # [B*T, H] -> [B, T, H]
        return routed, aux_loss


class KimiBlock(nn.Module):
    """由 Pre-Norm 注意力和 MoE/SwiGLU 前馈组成的 Decoder block。

    输入和输出均为 ``[B, T, H]``。注意力缓存类型由当前层决定：
    MLA 层返回 K/V，KDA 层返回有限递归状态。
    """

    def __init__(self, config: KimiConfig, layer_index: int) -> None:
        super().__init__()
        if config.attention_type == "hybrid":
            attention_name = "mla" if (layer_index + 1) % (config.kda_ratio + 1) == 0 else "kda"
        else:
            attention_name = config.attention_type
        self.attention_name = attention_name
        self.input_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = (
            MLAAttention(config) if attention_name == "mla" else KDAttention(config)
        )
        self.feed_forward = (
            TopKMoE(config)
            if config.use_moe
            else SwiGLU(config.hidden_size, config.intermediate_size)
        )

    def forward(
        self,
        hidden_states: Tensor,
        past_key_value: Any = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Any, Optional[Tensor]]:
        """执行一个 Decoder block。

        参数:
            hidden_states (Tensor): 输入隐藏状态，shape: ``[B, T, H]``。
            past_key_value (Any, optional): 当前层的历史缓存。
            use_cache (bool): 是否返回当前层缓存。

        输出:
            tuple[Tensor, Any, Tensor | None]: 输出 shape 为 ``[B, T, H]``；
            同时返回缓存和可选的 MoE 辅助损失。
        """
        attention_output, present = self.attention(
            self.input_norm(hidden_states), past_key_value, use_cache
        )
        hidden_states = hidden_states + attention_output
        normalized = self.post_attention_norm(hidden_states)
        if isinstance(self.feed_forward, TopKMoE):
            feed_forward_output, aux_loss = self.feed_forward(normalized)
        else:
            feed_forward_output = self.feed_forward(normalized)
            aux_loss = None
        return hidden_states + feed_forward_output, present, aux_loss


class KimiForCausalLM(nn.Module):
    """支持缓存复用生成的 Kimi 风格 Decoder-only 因果语言模型。

    结构顺序:
        Embedding -> KimiBlock × N -> RMSNorm -> LM Head。

    训练目标:
        ``L_total = CrossEntropy(logits, labels) + 0.01 * L_MoE_aux``。
    """

    def __init__(self, config: KimiConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [KimiBlock(config, index) for index in range(config.num_layers)]
        )
        self.final_norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[tuple[Any, ...]] = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        """执行训练或增量推理前向计算。

        参数:
            input_ids (Tensor): token id，shape: ``[B, T]``。
            labels (Tensor, optional): 下一个 token 标签，shape: ``[B, T]``。
            past_key_values (tuple, optional): 每层历史缓存。
            use_cache (bool): 是否返回每层更新后的缓存。

        输出:
            ModelOutput: ``logits`` shape 为 ``[B, T, V]``；若提供 labels，
            ``loss`` 为标量；缓存 shape 由 MLA/KDA 层类型决定。
        """
        hidden_states = self.embedding(input_ids)
        # [B, T] -> [B, T, H]
        presents: list[Any] = []
        aux_losses: list[Tensor] = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            hidden_states, present, aux_loss = layer(hidden_states, past, use_cache)
            if use_cache:
                presents.append(present)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        logits = self.lm_head(self.final_norm(hidden_states))
        # [B, T, H] -> [B, T, V]
        loss = None
        if labels is not None:
            lm_loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), labels.reshape(-1)
            )
            aux_loss = (
                torch.stack(aux_losses).mean()
                if aux_losses
                else torch.zeros((), device=logits.device)
            )
            loss = lm_loss + 0.01 * aux_loss
        else:
            aux_loss = torch.stack(aux_losses).mean() if aux_losses else None
        return ModelOutput(
            logits=logits,
            loss=loss,
            past_key_values=tuple(presents) if use_cache else None,
            aux_loss=aux_loss,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 32,
        temperature: float = 0.8,
        eos_token_id: Optional[int] = None,
    ) -> Tensor:
        """使用每层缓存执行自回归生成。

        输入:
            input_ids (Tensor): 初始 prompt，shape: ``[B, T_prompt]``。
            max_new_tokens (int): 最多生成的 token 数。
            temperature (float): 采样温度。
            eos_token_id (int, optional): 遇到该 token 时提前停止。

        输出:
            Tensor: prompt 与生成结果拼接后的 token id，shape:
            ``[B, T_prompt + T_new]``。

        生成首轮处理完整 prompt；后续轮次只输入 ``output[:, -1:]``，
        并复用 MLA 的 KV cache 或 KDA 的递归状态。
        """
        self.eval()
        output = input_ids
        cache = None
        for _ in range(max_new_tokens):
            current = output if cache is None else output[:, -1:]
            result = self(current, past_key_values=cache, use_cache=True)
            cache = result.past_key_values
            logits = result.logits[:, -1] / max(temperature, 1e-5)
            next_token = torch.multinomial(F.softmax(logits, dim=-1), 1)
            output = torch.cat((output, next_token), dim=-1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return output


def format_prompt(prompt: str, thinking: bool = False) -> str:
    """构造带可选思考标记的最小 Kimi 风格对话模板。

    输入:
        prompt: 用户问题文本。
        thinking: 是否追加 ``<think>`` 标记。
    输出:
        str: 格式化后的 prompt 文本，tokenize 前为 Python 字符串。
    """
    suffix = "<think>\n" if thinking else ""
    return f"<|system|>\nYou are Kimi, an AI assistant.\n<|user|>\n{prompt}\n<|assistant|>\n{suffix}"


def build_training_batch_from_args(
    args: argparse.Namespace, model_device: torch.device
) -> tuple[Tensor, Tensor]:
    """使用共享字节 tokenizer 根据命令行参数构造训练 batch。

    输出:
        tuple[Tensor, Tensor]: inputs 与 labels，shape 均为 ``[B, T]``。
    """
    tokenizer = ByteTokenizer()
    return build_training_batch(
        tokenizer,
        args.text,
        args.batch_size,
        args.seq_len,
        args.step,
        model_device,
    )
