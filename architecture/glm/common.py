"""GLM/ChatGLM 教学模型的公共组件与可运行的因果语言模型。

任务定义:
    任务编号: GLM-COMMON；领域: decoder-only Transformer / causal language
    modeling。该文件为 GLM-130B、ChatGLM、ChatGLM2、ChatGLM3、GLM-4、
    GLM-4.5、GLM-4.6 和 GLM-4.7 教学实现提供共享模块。

代表架构与核心机制:
    输入 token ids [B, T] 经过 embedding 得到 hidden states [B, T, H]，
    随后通过 RMSNorm、带 RoPE 的 MHA/GQA/MQA、残差连接和 SwiGLU/MoE
    前馈层，最后输出 vocabulary logits [B, T, V]。GLM 风格 blank
    infilling 通过二维 position ids、prefix-visible attention 和 target
    区域 loss mask 表达；自回归生成时每层缓存 K/V，后续步骤只计算新 token。

核心公式:
    RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
    Attention(Q, K, V) = softmax(QKᵀ / sqrt(D) + mask) V
    RoPE(x, p) = x * cos(p * inv_freq) + rotate_half(x) * sin(p * inv_freq)
    SwiGLU(x) = W_down(SiLU(W_gate x) * W_up x)
    L_lm = CrossEntropy(logits[:, :-1], labels[:, 1:])
    L_total = L_lm + 0.01 * L_MoE_aux

说明:
    本实现面向 CPU 教学和小规模实验，强调可读的 Tensor Shape 与接口，
    不兼容任何已发布 GLM/ChatGLM 官方 checkpoint、tokenizer 或训练配方。
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelOutput:
    """因果语言模型的最小输出容器。

    Attributes:
        logits: 下一 token 的未归一化分数，shape: [B, T, V]。
        loss: 可选的语言模型损失，shape: []。
        past_key_values: 可选的逐层 KV cache，每层 K/V shape 为
            [B, n_kv_heads, T_cache, D]。
        aux_loss: 可选的 MoE 负载均衡损失，shape: []。
    """

    logits: Tensor
    loss: Optional[Tensor] = None
    past_key_values: Optional[tuple[tuple[Tensor, Tensor], ...]] = None
    aux_loss: Optional[Tensor] = None


@dataclass
class GLMConfig:
    """共享模型超参数，并在初始化时校验注意力和 MoE 约束。"""

    vocab_size: int = 259
    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: Optional[int] = None
    max_position_embeddings: int = 512
    max_block_position_embeddings: int = 128
    rope_theta: float = 10_000.0
    rope_scaling: float = 1.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    qkv_bias: bool = True
    tie_word_embeddings: bool = True
    use_2d_position_ids: bool = False
    use_qk_norm: bool = False
    use_moe: bool = False
    num_experts: int = 4
    num_experts_per_tok: int = 2

    def __post_init__(self) -> None:
        """补全 KV head 默认值并校验维度可整除关系。"""
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")


class RMSNorm(nn.Module):
    """沿最后一维执行 Root Mean Square Layer Normalization。

    公式为 ``x / sqrt(mean(x²) + eps) * weight``。

    Inputs:
        x (Tensor): 任意前导维度、最后一维为 hidden size，shape: [..., H]。
    Outputs:
        Tensor: 与输入同 shape 的归一化结果，shape: [..., H]。
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """计算 RMSNorm，使用 float 统计均方值以提高低精度稳定性。"""
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        return self.weight * (x * torch.rsqrt(variance + self.eps).to(x.dtype))


def rotate_half(x: Tensor) -> Tensor:
    """交换并旋转最后一维的前后半段，保持输入 Shape 不变。

    Inputs/Outputs:
        x: shape [..., D]，其中 D 必须为偶数；返回 shape [..., D]。
    """
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)  # [..., D/2] + [..., D/2] -> [..., D]


class RotaryEmbedding(nn.Module):
    """生成并应用旋转位置编码（RoPE）。

    位置坐标先按 ``position / scaling`` 缩放，再与 ``inv_freq`` 外积生成
    旋转角度；代码中的 ``_cos``、``_sin`` 对应公式中的 cos(θ)、sin(θ)。

    Inputs:
        query: shape [B, n_heads, T_q, D]。
        key: shape [B, n_kv_heads, T_q, D]。
        position_ids: 可选位置索引，shape [B, T_q] 或 [B, 2, T_q]。
    Outputs:
        旋转后的 ``(query, key)``，shape 分别与输入相同。
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        theta: float,
        scaling: float,
    ) -> None:
        super().__init__()
        if dim % 2:
            raise ValueError("rotary dimension must be even")
        self.dim = dim
        self.scaling = scaling
        self.register_buffer(
            "inv_freq",
            1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim)),
            persistent=False,
        )
        self._set_cache(max_position_embeddings)

    def _set_cache(self, length: int) -> None:
        """构造至少覆盖 ``length`` 个位置的 cos/sin cache。"""
        positions = torch.arange(length, device=self.inv_freq.device).float()
        positions = positions / self.scaling  # p' = p / scaling
        frequencies = torch.outer(positions, self.inv_freq)  # [T, D/2]
        embeddings = torch.cat((frequencies, frequencies), dim=-1)  # [T, D]
        self.register_buffer("_cos", embeddings.cos()[None, None], persistent=False)  # [1, 1, T, D]
        self.register_buffer("_sin", embeddings.sin()[None, None], persistent=False)  # [1, 1, T, D]

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        position_ids: Optional[Tensor] = None,
        position_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
        """按 position ids 和 cache 偏移旋转 query/key。"""
        batch, _, query_length, _ = query.shape
        if position_ids is None:
            positions = torch.arange(
                position_offset,
                position_offset + query_length,
                device=query.device,
            ).expand(batch, -1)
        elif position_ids.dim() == 3:
            positions = position_ids[:, 0]
        else:
            positions = position_ids
        needed = int(positions.max().item()) + 1
        if needed > self._cos.shape[-2]:
            self._set_cache(max(needed, 2 * self._cos.shape[-2]))
        cos = self._cos[0, 0, positions].unsqueeze(1).to(query.dtype)  # [B, 1, Tq, D]
        sin = self._sin[0, 0, positions].unsqueeze(1).to(query.dtype)  # [B, 1, Tq, D]
        query = query * cos + rotate_half(query) * sin  # [B, n_heads, Tq, D]
        key = key * cos + rotate_half(key) * sin  # [B, n_kv_heads, Tq, D]
        return query, key


def repeat_kv(hidden_states: Tensor, repeats: int) -> Tensor:
    """将 GQA/MQA 的 KV head 复制为 query head 数量。

    Inputs:
        hidden_states: shape [B, n_kv_heads, T, D]。
        repeats: ``n_heads // n_kv_heads``。
    Outputs:
        shape [B, n_kv_heads * repeats, T, D]。
    """
    if repeats == 1:
        return hidden_states
    batch, heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None].expand(  # [B, n_kv_heads, repeats, T, D]
        batch, heads, repeats, seq_len, head_dim
    )
    return expanded.reshape(batch, heads * repeats, seq_len, head_dim)  # [B, n_heads, T, D]


def _prefix_attention_mask(
    query_length: int,
    total_length: int,
    past_length: int,
    prefix_length: Optional[int | Tensor],
    device: torch.device,
) -> Tensor:
    """构造 prefix-visible + causal attention mask。

    prefix 内 token 互相可见；生成区只能看当前位置及其左侧位置。
    返回布尔 mask，shape 为 [1 或 B, T_q, T_k]。
    """
    query_positions = torch.arange(
        past_length, past_length + query_length, device=device
    )[None, :, None]  # [1, Tq, 1]
    key_positions = torch.arange(total_length, device=device)[None, None, :]  # [1, 1, Tk]
    allowed = key_positions <= query_positions  # [1, Tq, Tk]，基础因果 mask
    if prefix_length is not None:
        if isinstance(prefix_length, int):
            prefix = torch.full(
                (1, 1, 1), prefix_length, device=device, dtype=torch.long
            )
        else:
            prefix = prefix_length.to(device=device, dtype=torch.long).view(-1, 1, 1)
        allowed = allowed | (  # prefix 内额外开放双向可见性
            (query_positions < prefix) & (key_positions < prefix)
        )
    return allowed


class SelfAttention(nn.Module):
    """支持 MHA/GQA/MQA、RoPE、prefix mask 和 KV cache 的自注意力层。

    数学映射:
        ``scores = QKᵀ / sqrt(D)``；
        ``weights = softmax(scores + mask)``；
        ``output = weights @ V``。

    Inputs:
        hidden_states: shape [B, T_q, H]。
        position_ids: 可选位置索引，shape [B, T_q] 或 [B, 2, T_q]。
        past_key_value: 可选 ``(K_cache, V_cache)``，每项 shape
            [B, n_kv_heads, T_cache, D]。
    Outputs:
        attention_output: shape [B, T_q, H]；
        present: 可选最新 cache，K/V shape 为
            [B, n_kv_heads, T_cache + T_q, D]。
    """

    def __init__(self, config: GLMConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.q_proj = nn.Linear(
            config.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.qkv_bias,
        )
        self.v_proj = nn.Linear(
            config.hidden_size,
            self.num_kv_heads * self.head_dim,
            bias=config.qkv_bias,
        )
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = (
            RMSNorm(self.head_dim, config.rms_norm_eps)
            if config.use_qk_norm
            else nn.Identity()
        )
        self.k_norm = (
            RMSNorm(self.head_dim, config.rms_norm_eps)
            if config.use_qk_norm
            else nn.Identity()
        )
        self.rotary = RotaryEmbedding(
            self.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rope_scaling,
        )
        self.dropout = config.dropout

    def forward(
        self,
        hidden_states: Tensor,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[tuple[Tensor, Tensor]] = None,
        prefix_length: Optional[int | Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        """执行一次带可选 prefix mask 和 KV cache 的注意力计算。"""
        batch, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)  # [B, n_heads, Tq, D]
        key = self.k_proj(hidden_states).view(
            batch, query_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)  # [B, n_kv_heads, Tq, D]
        value = self.v_proj(hidden_states).view(
            batch, query_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)  # [B, n_kv_heads, Tq, D]
        query, key = self.q_norm(query), self.k_norm(key)
        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        query, key = self.rotary(query, key, position_ids, past_length)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)  # [B, n_kv_heads, T_cache + Tq, D]
            value = torch.cat((past_key_value[1], value), dim=-2)  # [B, n_kv_heads, T_cache + Tq, D]

        key_for_attention = repeat_kv(key, self.num_kv_groups)  # [B, n_heads, Tk, D]
        value_for_attention = repeat_kv(value, self.num_kv_groups)  # [B, n_heads, Tk, D]
        scores = torch.matmul(query, key_for_attention.transpose(-2, -1))  # [B, n_heads, Tq, Tk]
        scores = scores / math.sqrt(self.head_dim)  # QK^T / sqrt(D)
        allowed = _prefix_attention_mask(  # [1 or B, Tq, Tk]
            query_length,
            key.shape[-2],
            past_length,
            prefix_length,
            hidden_states.device,
        )
        scores = scores.masked_fill(~allowed[:, None], torch.finfo(scores.dtype).min)  # mask -> -inf
        weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)  # [B, n_heads, Tq, Tk]
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.matmul(weights, value_for_attention)  # [B, n_heads, Tq, D]
        output = output.transpose(1, 2).contiguous().view(batch, query_length, -1)  # [B, Tq, H]
        present = (key, value) if use_cache else None
        return self.o_proj(output), present  # [B, Tq, H]


class SwiGLU(nn.Module):
    """门控前馈网络，使用 SiLU 门分支调制线性上投影。

    数学公式:
        ``SwiGLU(x) = W_down(SiLU(W_gate x) * W_up x)``。

    Inputs/Outputs:
        输入 shape [..., H]，中间特征 shape [..., I]，输出 shape [..., H]。
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """计算门控前馈变换，并保持输入的所有前导维度不变。"""
        gate = F.silu(self.gate_proj(x))  # [..., H] -> [..., I]，对应 SiLU(W_gate x)
        up = self.up_proj(x)  # [..., H] -> [..., I]，对应 W_up x
        return self.down_proj(gate * up)  # [..., I] -> [..., H]，对应 W_down(...)


class MoEBlock(nn.Module):
    """按 token 路由的 top-k MoE 前馈层。

    router 先计算 ``p(e|x) = softmax(W_router x)``，再选择每个 token 的
    ``num_experts_per_tok`` 个 expert；各 expert 的 SwiGLU 输出按归一化后的
    router 权重加权求和。辅助损失为：
    ``L_aux = E * sum(mean(p_e) * mean(assign_e))``。

    Inputs:
        x: hidden states，shape [B, T, H]。
    Outputs:
        output: shape [B, T, H]；
        aux_loss: 标量负载均衡损失，shape []。
    """

    def __init__(self, config: GLMConfig) -> None:
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                SwiGLU(config.hidden_size, config.intermediate_size)
                for _ in range(config.num_experts)
            ]
        )
        self.num_experts = config.num_experts
        self.num_experts_per_tok = config.num_experts_per_tok

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """执行 token-level top-k 路由和 expert 加权聚合。"""
        flat = x.reshape(-1, x.shape[-1])  # [B, T, H] -> [B*T, H]
        router_logits = self.router(flat)  # [B*T, H] -> [B*T, E]
        weights, indices = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )  # weights/indices: [B*T, K]
        weights = F.softmax(weights.float(), dim=-1).to(flat.dtype)  # top-k 权重归一化
        result = torch.zeros_like(flat)  # [B*T, H]
        for expert_id, expert in enumerate(self.experts):
            token_index, slot = torch.where(indices == expert_id)
            if token_index.numel():
                result.index_add_(
                    0,
                    token_index,
                    expert(flat[token_index]) * weights[token_index, slot, None],  # [N_e, H]
                )
        probabilities = router_logits.softmax(dim=-1).mean(dim=0)  # [E]
        assignments = F.one_hot(indices[:, 0], self.num_experts).float().mean(dim=0)  # [E]
        aux_loss = self.num_experts * torch.sum(  # E * sum(mean(p_e) * mean(assign_e))
            probabilities * assignments
        )
        return result.view_as(x), aux_loss  # [B*T, H] -> [B, T, H]


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with attention and dense/MoE residual branches。

    数据流:
        ``x -> x + Attention(RMSNorm(x)) -> x + MLP(RMSNorm(x))``。

    Inputs/Outputs:
        ``x`` shape [B, T, H]，输出 shape [B, T, H]；同时返回可选逐层
        KV cache 和可选 MoE 辅助损失。
    """

    def __init__(self, config: GLMConfig) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = SelfAttention(config)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = (
            MoEBlock(config)
            if config.use_moe
            else SwiGLU(config.hidden_size, config.intermediate_size)
        )

    def forward(
        self,
        x: Tensor,
        position_ids: Optional[Tensor] = None,
        past_key_value: Optional[tuple[Tensor, Tensor]] = None,
        prefix_length: Optional[int | Tensor] = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]], Optional[Tensor]]:
        """执行一个 Transformer block，并保持残差分支的 Shape 不变。"""
        attention_output, present = self.self_attn(
            self.input_layernorm(x),
            position_ids,
            past_key_value,
            prefix_length,
            use_cache,
        )
        x = x + attention_output  # [B, T, H] + [B, T, H] -> [B, T, H]
        mlp_input = self.post_attention_layernorm(x)
        if isinstance(self.mlp, MoEBlock):
            mlp_output, aux_loss = self.mlp(mlp_input)
        else:
            mlp_output, aux_loss = self.mlp(mlp_input), None
        return x + mlp_output, present, aux_loss  # [B, T, H]


class GLMCausalLM(nn.Module):
    """共享的 GLM 风格 decoder-only causal language model。

    输入 token ids ``[B, T]``，输出 logits ``[B, T, V]`。模型支持普通因果
    语言建模，也支持通过 ``prefix_length`` 表达 blank infilling 的 prefix
    双向可见区域；训练时还可将各层 MoE 辅助损失加入总目标。
    """

    def __init__(self, config: GLMConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.block_position_embeddings = (
            nn.Embedding(config.max_block_position_embeddings, config.hidden_size)
            if config.use_2d_position_ids
            else None
        )
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.apply(self._init_weights)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """使用小标准差正态分布初始化线性层和 embedding。"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _default_position_ids(
        self, input_ids: Tensor, past_length: int
    ) -> Tensor:
        """为普通 causal LM 创建一维或二维默认 position ids。"""
        batch, seq_len = input_ids.shape
        absolute = torch.arange(
            past_length, past_length + seq_len, device=input_ids.device
        ).expand(batch, -1)  # [B, T]
        if not self.config.use_2d_position_ids:
            return absolute
        block = torch.zeros_like(absolute)  # [B, T]
        return torch.stack((absolute, block), dim=1)  # [B, 2, T]

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        prefix_length: Optional[int | Tensor] = None,
        past_key_values: Optional[Sequence[tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        """执行 decoder 前向传播。

        Inputs:
            input_ids: token ids，shape [B, T]。
            labels: 可选监督 token ids，shape [B, T]；``-100`` 位置忽略。
            position_ids: 一维或二维位置索引，shape [B, T] 或 [B, 2, T]。
            prefix_length: prefix 双向可见区域长度，可为标量或 shape [B]。
            past_key_values: 可选逐层 cache，每层 K/V shape
                [B, n_kv_heads, T_cache, D]。
        Outputs:
            ``ModelOutput.logits`` shape [B, T, V]；启用 cache 时，每层
            present 的 K/V shape 为 [B, n_kv_heads, T_cache + T, D]。
        """
        past_length = (
            0 if past_key_values is None else past_key_values[0][0].shape[-2]
        )
        if position_ids is None:
            position_ids = self._default_position_ids(input_ids, past_length)
        hidden_states = self.embed_tokens(input_ids)  # [B, T] -> [B, T, H]
        if self.block_position_embeddings is not None:
            block_positions = (
                position_ids[:, 1]
                if position_ids.dim() == 3
                else torch.zeros_like(input_ids)
            )  # [B, T]
            hidden_states = hidden_states + self.block_position_embeddings(
                block_positions.clamp_max(
                    self.config.max_block_position_embeddings - 1
                )
            )  # [B, T, H] + [B, T, H] -> [B, T, H]

        presents: list[tuple[Tensor, Tensor]] = []
        aux_losses = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            hidden_states, present, aux_loss = layer(
                hidden_states, position_ids, past, prefix_length, use_cache
            )
            if use_cache and present is not None:
                presents.append(present)
            if aux_loss is not None:
                aux_losses.append(aux_loss)

        logits = self.lm_head(self.norm(hidden_states))  # [B, T, H] -> [B, T, V]
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()  # [B, T-1, V]
            shift_labels = labels[:, 1:].contiguous()  # [B, T-1]
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
            )  # L_lm；展平为 [(B*(T-1)), V] 和 [B*(T-1)]
            if aux_losses:
                loss = loss + 0.01 * torch.stack(aux_losses).mean()  # L_total = L_lm + 0.01 * L_aux
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
        top_k: int = 40,
        top_p: float = 0.95,
        eos_token_id: Optional[int] = 2,
        prefix_length: Optional[int] = None,
    ) -> Tensor:
        """使用 temperature、top-k、top-p 和逐层 KV cache 自回归生成。

        Inputs:
            input_ids: prompt token ids，shape [B, T_prompt]。
        Outputs:
            包含 prompt 与新 token 的序列，shape [B, T_prompt + T_new]。
        """
        self.eval()
        generated = input_ids
        prompt_length = generated.shape[1] if prefix_length is None else prefix_length
        past_key_values = None
        for _ in range(max_new_tokens):
            current = (
                generated
                if past_key_values is None
                else generated[:, -1:]
            )  # 首轮 [B, T_prompt]，后续 [B, 1]
            output = self(
                current,
                past_key_values=past_key_values,
                prefix_length=prompt_length,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            logits = output.logits[:, -1]  # [B, T_current, V] -> [B, V]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits = logits.masked_fill(
                        logits < values[:, [-1]], float("-inf")
                    )  # 过滤低于第 k 大值的 logits
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True
                    )
                    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)  # [B, V]
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    logits.scatter_(
                        1,
                        sorted_indices,
                        sorted_logits.masked_fill(remove, float("-inf")),
                    )
                next_token = torch.multinomial(logits.softmax(dim=-1), 1)
            generated = torch.cat((generated, next_token), dim=-1)  # [B, T] + [B, 1] -> [B, T+1]
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return generated


class ByteTokenizer:
    """无依赖的 UTF-8 字节级 tokenizer。

    token ``0/1/2`` 分别保留给 PAD/BOS/EOS，真实字节 ``0..255`` 映射到
    ``3..258``，因此词表大小为 ``259``。编码结果是一维 token id 列表，
    由调用方堆叠为模型所需的 ``[B, T]``。
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = False
    ) -> list[int]:
        """将文本编码为 token ids，长度等于 UTF-8 字节数加特殊 token 数。"""
        ids = [byte + 3 for byte in text.encode("utf-8")]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        """过滤特殊 token 后将字节 token 解码为 UTF-8 文本。"""
        payload = bytes(token - 3 for token in ids if 3 <= token < self.vocab_size)
        return payload.decode("utf-8", errors="replace")


def build_causal_batch(
    tokenizer: ByteTokenizer,
    text: str,
    batch_size: int,
    seq_len: int,
    step: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Optional[int]]:
    """从循环文本中抽取固定长度的 teacher-forcing causal batch。

    Args:
        tokenizer: 提供 ``encode`` 的 tokenizer。
        text: 用于教学训练的文本。
        batch_size: batch 大小 B。
        seq_len: 每条序列长度 T。
        step: 当前训练步，用于移动窗口起点。
        device: 输出 tensor 所在设备。

    Returns:
        ``inputs`` 和 ``labels``，shape 均为 [B, T]；第三项为
        ``prefix_length=None``，表示使用普通因果 mask。
    """
    ids = torch.tensor(
        tokenizer.encode(text, add_bos=True, add_eos=True), device=device
    )  # [N]
    if ids.numel() < seq_len + 2:
        ids = ids.repeat(math.ceil((seq_len + 2) / ids.numel()))
    max_start = max(1, ids.numel() - seq_len - 1)
    starts = [
        (step * batch_size * seq_len + index * seq_len) % max_start
        for index in range(batch_size)
    ]
    inputs = torch.stack([ids[start : start + seq_len] for start in starts])  # [B, T]
    return inputs, inputs.clone(), None


def build_blank_infilling_batch(
    tokenizer: ByteTokenizer,
    text: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, int]:
    """构造 GLM blank-infilling 目标的 context/target batch。

    样本布局为 ``[BOS] + prefix + <BLANK> + suffix + target``。prefix 和
    suffix 组成可双向读取的 context，``labels`` 只在 target 区域保留 token，
    context 区域填充 ``-100``。返回的输入/标签 shape 均为 [B, T]。
    """

    payload = tokenizer.encode(text, add_bos=False, add_eos=False)
    if len(payload) < 8:
        payload = payload * math.ceil(8 / len(payload))
    split_start = max(2, len(payload) // 3)
    split_end = min(len(payload) - 2, split_start + max(2, len(payload) // 4))
    prefix = payload[:split_start]
    target = payload[split_start:split_end]
    suffix = payload[split_end:]
    blank = tokenizer.encode("<BLANK>", add_bos=False)
    sequence = [tokenizer.bos_token_id] + prefix + blank + suffix + target  # [N]
    sequence = sequence[: max(seq_len, len(sequence))]
    sequence = sequence[:seq_len]  # [<= T]
    if len(sequence) < seq_len:
        sequence.extend([tokenizer.pad_token_id] * (seq_len - len(sequence)))
    prefix_length = min(
        seq_len,
        1 + len(prefix) + len(blank) + len(suffix),
    )
    inputs = torch.tensor([sequence] * batch_size, device=device)  # [B, T]
    labels = torch.full_like(inputs, -100)  # [B, T]
    if prefix_length < seq_len:
        labels[:, prefix_length:] = inputs[:, prefix_length:]  # 仅 target 位置计算 loss
    return inputs, labels, prefix_length


def format_chat(
    messages: Sequence[dict[str, str]],
    system_prompt: Optional[str] = None,
    thinking: bool = False,
    tools: Optional[Sequence[dict]] = None,
) -> str:
    """将消息、工具 schema 和 thinking 标记组织为 ChatGLM 风格文本。

    输入消息是 role/content 字典序列；返回的字符串随后编码为一维 token id
    列表，再由推理入口构造成 shape [1, T] 的 ``input_ids``。
    """

    parts = ["[gMASK]", "<sop>"]
    if system_prompt:
        parts.extend(["<|system|>\n", system_prompt, "\n"])
    if tools:
        parts.extend(["<|tools|>\n", json.dumps(tools, ensure_ascii=False), "\n"])
    for message in messages:
        role = message["role"]
        parts.extend([f"<|{role}|>\n", message["content"], "\n"])
    parts.extend(["<|assistant|>\n"])
    if thinking:
        parts.append("<think>\n")
    return "".join(parts)


def parse_tool_call(text: str) -> Optional[dict]:
    """抽取首个 ``<tool_call>...</tool_call>`` 中的 JSON 对象。

    该函数只负责解析并返回字典，不执行命令、不校验工具权限，也不调用外部
    工具。解析失败或 JSON 顶层不是 object 时返回 ``None``。
    """

    match = re.search(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text, re.S)
    if not match:
        return None
    try:
        value = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def train_model(
    model: GLMCausalLM,
    text: str,
    steps: int,
    batch_size: int,
    seq_len: int,
    lr: float,
    device: str,
    checkpoint: str,
    blank_infilling: bool = False,
) -> None:
    """运行教学模型训练循环，并保存 ``model.state_dict()``。

    普通 causal batch 使用 shape [B, T]；blank-infilling batch 额外传入
    ``prefix_length``，让前缀区域使用 prefix-visible mask。模型输出的
    logits shape 为 [B, T, V]，loss 和可选 aux_loss 为标量。
    """
    target_device = torch.device(device)
    tokenizer = ByteTokenizer()
    model = model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        if blank_infilling:
            inputs, labels, prefix_length = build_blank_infilling_batch(  # inputs/labels: [B, T]
                tokenizer, text, batch_size, seq_len, target_device
            )
        else:
            inputs, labels, prefix_length = build_causal_batch(  # inputs/labels: [B, T]
                tokenizer, text, batch_size, seq_len, step, target_device
            )
        output = model(inputs, labels=labels, prefix_length=prefix_length)  # logits: [B, T, V]
        if output.loss is None:
            raise RuntimeError("training batch did not produce a loss")
        optimizer.zero_grad(set_to_none=True)
        output.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step + 1) % max(1, steps // 5) == 0:
            aux = 0.0 if output.aux_loss is None else output.aux_loss.item()
            print(
                f"step {step + 1:03d}/{steps}: "
                f"loss={output.loss.item():.4f}, aux={aux:.4f}"
            )
    torch.save({"model": model.state_dict()}, checkpoint)
    print(f"saved checkpoint to {checkpoint}")


def generation_cli(
    build_model: Callable[[], GLMCausalLM],
    default_prompt: str,
    default_text_prefix: str = "",
    chat: bool = False,
) -> None:
    """提供统一的 GLM 推理命令行入口。

    prompt 经过可选 ChatGLM 模板和字节 tokenizer 后变为 ``input_ids``
    shape [1, T]；``generate`` 返回包含 prompt 的 token 序列
    shape [1, T + T_new]，最后再解码为文本。
    """
    parser = argparse.ArgumentParser(description="Generate with a tiny GLM-style LM.")
    parser.add_argument("--prompt", default=default_prompt)
    parser.add_argument("--checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model = build_model().to(device)
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(state.get("model", state))
    tokenizer = ByteTokenizer()
    prompt = args.prompt
    if chat:
        prompt = format_chat(
            [{"role": "user", "content": prompt}],
            thinking=args.thinking,
        )
    prompt = default_text_prefix + prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)  # [1, T]
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(output_ids[0].tolist()))  # output_ids: [1, T + T_new]


def train_cli(
    build_model: Callable[[], GLMCausalLM],
    default_text: str,
    blank_infilling: bool = False,
) -> None:
    """解析统一训练参数并调用 ``train_model``。"""
    parser = argparse.ArgumentParser(description="Train a tiny GLM-style causal LM.")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--text", default=default_text)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--checkpoint", default="glm_tiny.pt")
    args = parser.parse_args()
    train_model(
        build_model(),
        args.text,
        args.steps,
        args.batch_size,
        args.seq_len,
        args.lr,
        args.device,
        args.checkpoint,
        blank_infilling=blank_infilling,
    )


if __name__ == "__main__":
    print("Import this module from one of the versioned GLM examples.")
