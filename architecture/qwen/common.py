"""Qwen 教学模型的公共组件与可运行的因果语言模型。

任务定义:
    任务编号: QWEN-COMMON；领域: decoder-only Transformer / causal language
    modeling。该文件为 Qwen1、Qwen1.5、Qwen2、Qwen2.5 和 Qwen3 教学模型
    提供共享的注意力、RoPE、SwiGLU、MoE、KV cache 和字节级 tokenizer。

代表架构与核心机制:
    输入 token ids [B, T] 经过 embedding 得到 [B, T, H]，随后依次通过
    RMSNorm、RoPE 注意力、残差连接和 SwiGLU/MoE 前馈层，最后映射为
    vocabulary logits [B, T, V]。当 n_kv_heads < n_heads 时，KV head
    按 GQA 的重复因子复制；自回归生成时每层缓存 K/V，后续步骤只计算新 token。

核心公式:
    RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
    Attention(Q, K, V) = softmax(QKᵀ / sqrt(d) + causal_mask) V
    RoPE(x) = x * cos(θ) + rotate_half(x) * sin(θ)
    SwiGLU(x) = W_down(SiLU(W_gate x) * W_up x)
    L_lm = CrossEntropy(logits[:, :-1], labels[:, 1:])

说明:
    本实现面向 CPU 教学和小规模实验，不兼容任何已发布 Qwen 官方 checkpoint。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelOutput:
    """因果语言模型的最小输出容器。

    Attributes:
        logits: 下一 token 的未归一化分数，shape: [B, T, V]。
        loss: 可选的语言模型损失，shape: []。
        past_key_values: 可选的逐层 KV cache，每层的 K/V shape 为
            [B, n_kv_heads, T_cache, head_dim]。
        aux_loss: 可选的 MoE 负载均衡损失，shape: []。
    """

    logits: Tensor
    loss: Optional[Tensor] = None
    past_key_values: Optional[tuple[tuple[Tensor, Tensor], ...]] = None
    aux_loss: Optional[Tensor] = None


@dataclass
class ModelConfig:
    """共享模型超参数，并在初始化时校验注意力和 MoE 约束。"""

    vocab_size: int = 259
    hidden_size: int = 128
    intermediate_size: int = 352
    num_layers: int = 4
    num_heads: int = 4
    num_kv_heads: Optional[int] = None
    max_position_embeddings: int = 512
    rope_theta: float = 1_000_000.0
    rope_scaling: float = 1.0
    rms_norm_eps: float = 1e-6
    dropout: float = 0.0
    qkv_bias: bool = False
    tie_word_embeddings: bool = True
    use_moe: bool = False
    num_experts: int = 4
    num_experts_per_tok: int = 2
    use_qk_norm: bool = False

    def __post_init__(self) -> None:
        """补全 KV head 默认值并校验维度可整除关系。"""
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
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
        """计算 RMSNorm，使用 float 统计方差以提高低精度稳定性。"""
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return self.weight * x


def rotate_half(x: Tensor) -> Tensor:
    """交换并旋转最后一维的前后半段，保持输入 shape 不变。

    Inputs/Outputs:
        x: shape [..., D]，其中 D 必须为偶数；返回 shape [..., D]。
    """
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)  # [..., D/2] + [..., D/2] -> [..., D]


class RotaryEmbedding(nn.Module):
    """生成并应用旋转位置编码（RoPE）。

    位置坐标先按 ``position / scaling`` 缩放，再与 ``inv_freq`` 外积生成
    旋转角度；代码中的 ``_cos``、``_sin`` 对应公式中的 cos(θ)、sin(θ)。

    Inputs:
        q: query，shape [B, n_heads, T, D]。
        k: key，shape [B, n_kv_heads, T, D]。
    Outputs:
        旋转后的 ``(q, k)``，shape 分别与输入相同。
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int,
        theta: float = 1_000_000.0,
        scaling: float = 1.0,
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
        positions = positions / self.scaling  # pos' = pos / scaling
        freqs = torch.outer(positions, self.inv_freq)  # [T, D/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, D]
        self.register_buffer("_cos", emb.cos()[None, None], persistent=False)  # [1, 1, T, D]
        self.register_buffer("_sin", emb.sin()[None, None], persistent=False)  # [1, 1, T, D]

    def forward(
        self, q: Tensor, k: Tensor, position_offset: int = 0
    ) -> tuple[Tensor, Tensor]:
        """按 cache 中的位置偏移旋转 query 和 key。"""
        needed = position_offset + q.shape[-2]
        if needed > self._cos.shape[-2]:
            self._set_cache(max(needed, 2 * self._cos.shape[-2]))
        cos = self._cos[..., position_offset:needed, :].to(dtype=q.dtype)  # [1, 1, Tq, D]
        sin = self._sin[..., position_offset:needed, :].to(dtype=q.dtype)  # [1, 1, Tq, D]
        q = (q * cos) + (rotate_half(q) * sin)  # [B, n_heads, Tq, D]
        k = (k * cos) + (rotate_half(k) * sin)  # [B, n_kv_heads, Tq, D]
        return q, k


def repeat_kv(hidden_states: Tensor, repeats: int) -> Tensor:
    """将 GQA 的 KV head 复制为 query head 数量。

    Inputs:
        hidden_states: shape [B, n_kv_heads, T, D]。
        repeats: ``n_heads // n_kv_heads``。
    Outputs:
        shape [B, n_kv_heads * repeats, T, D]。
    """
    if repeats == 1:
        return hidden_states
    batch, heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, heads, repeats, seq_len, head_dim
    )  # [B, n_kv_heads, repeats, T, D]
    return hidden_states.reshape(batch, heads * repeats, seq_len, head_dim)  # [B, n_heads, T, D]


class SelfAttention(nn.Module):
    """支持 MHA/GQA、RoPE、因果 mask 和 KV cache 的自注意力层。

    数学映射:
        ``scores = QKᵀ / sqrt(head_dim)``；
        ``weights = softmax(scores + causal_mask)``；
        ``output = weights @ V``。

    Inputs:
        hidden_states: shape [B, T_q, H]。
        past_key_value: 可选的 ``(K_cache, V_cache)``，每项 shape
            [B, n_kv_heads, T_cache, D]。
    Outputs:
        attention_output: shape [B, T_q, H]；
        present: 可选的最新 cache，K/V shape 为
            [B, n_kv_heads, T_cache + T_q, D]。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_kv_heads or config.num_heads
        self.head_dim = config.hidden_size // config.num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=config.qkv_bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=config.qkv_bias)
        self.o_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.use_qk_norm else nn.Identity()
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
        past_key_value: Optional[tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]]]:
        """执行一次带可选前缀 cache 的自注意力计算。"""
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
        query = self.q_norm(query)
        key = self.k_norm(key)

        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        query, key = self.rotary(query, key, past_length)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)  # [B, n_kv_heads, T_cache + Tq, D]
            value = torch.cat((past_key_value[1], value), dim=-2)  # [B, n_kv_heads, T_cache + Tq, D]
        present = (key, value) if use_cache else None

        key_for_attention = repeat_kv(key, self.num_kv_groups)  # [B, n_heads, Tk, D]
        value_for_attention = repeat_kv(value, self.num_kv_groups)  # [B, n_heads, Tk, D]
        scores = torch.matmul(query, key_for_attention.transpose(-2, -1))  # [B, n_heads, Tq, Tk]
        scores = scores / math.sqrt(self.head_dim)  # QK^T / sqrt(D)
        total_length = key.shape[-2]
        if query_length > 1 or past_length == 0:
            query_positions = torch.arange(
                past_length, past_length + query_length, device=hidden_states.device
            )
            key_positions = torch.arange(total_length, device=hidden_states.device)
            future = key_positions[None, :] > query_positions[:, None]
            scores = scores.masked_fill(  # causal mask: future positions -> -inf
                future[None, None], torch.finfo(scores.dtype).min
            )
        weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)  # [B, n_heads, Tq, Tk]
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.matmul(weights, value_for_attention)  # [B, n_heads, Tq, D]
        output = output.transpose(1, 2).contiguous().view(batch, query_length, -1)  # [B, Tq, H]
        return self.o_proj(output), present  # [B, Tq, H]


class SwiGLU(nn.Module):
    """门控前馈网络，使用 SiLU 门分支调制线性上投影。

    公式为 ``W_down(SiLU(W_gate x) * W_up x)``。

    Inputs/Outputs:
        输入 shape [..., H]，输出 shape [..., H]；中间特征为 [..., I]。
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        gate = F.silu(self.gate_proj(x))  # [..., I], 对应 SiLU(W_gate x)
        up = self.up_proj(x)  # [..., I], 对应 W_up x
        return self.down_proj(gate * up)  # [..., H], 对应 W_down(...)


class MoEBlock(nn.Module):
    """按 token 路由的 top-k MoE 前馈层。

    router 先计算 ``p(e|x) = softmax(W_router x)``，再选择每个 token 的
    ``num_experts_per_tok`` 个 expert；各 expert 的 SwiGLU 输出按归一化后的
    router 权重加权求和。返回的辅助损失对应 Switch-style 负载均衡项。

    Inputs:
        x: hidden states，shape [B, T, H]。
    Outputs:
        output: shape [B, T, H]；
        aux_loss: 标量负载均衡损失，shape []。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(config.hidden_size, config.intermediate_size) for _ in range(config.num_experts)]
        )
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        flat = x.reshape(-1, x.shape[-1])  # [B, T, H] -> [B*T, H]
        router_logits = self.router(flat)  # [B*T, H] -> [B*T, E]
        weights, indices = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)  # [B*T, K]
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
        # Switch-style auxiliary loss encourages balanced expert usage.
        probabilities = router_logits.softmax(dim=-1).mean(dim=0)  # [E]
        counts = F.one_hot(indices[:, 0], self.num_experts).float().mean(dim=0)  # [E]
        aux_loss = self.num_experts * torch.sum(probabilities * counts)  # E * sum(mean(p_e) * mean(assign_e))
        return result.view_as(x), aux_loss  # [B*T, H] -> [B, T, H]


class TransformerBlock(nn.Module):
    """Pre-norm Transformer block with attention and dense/MoE residual branches。

    数据流:
        ``x -> x + Attention(RMSNorm(x)) -> x + MLP(RMSNorm(x))``。

    Inputs/Outputs:
        ``x`` shape [B, T, H]，输出 shape [B, T, H]；同时返回可选逐层
        KV cache 和可选 MoE 辅助损失。
    """

    def __init__(self, config: ModelConfig) -> None:
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
        past_key_value: Optional[tuple[Tensor, Tensor]] = None,
        use_cache: bool = False,
    ) -> tuple[Tensor, Optional[tuple[Tensor, Tensor]], Optional[Tensor]]:
        """执行一个 Transformer block，并保留残差分支的 Shape 不变性。"""
        attention_output, present = self.self_attn(
            self.input_layernorm(x), past_key_value, use_cache
        )
        x = x + attention_output  # [B, T, H] + [B, T, H] -> [B, T, H]
        mlp_input = self.post_attention_layernorm(x)
        if isinstance(self.mlp, MoEBlock):
            mlp_output, aux_loss = self.mlp(mlp_input)
        else:
            mlp_output, aux_loss = self.mlp(mlp_input), None
        return x + mlp_output, present, aux_loss  # [B, T, H]


class QwenCausalLM(nn.Module):
    """共享的 Qwen 风格 decoder-only causal language model。

    输入 token ids ``[B, T]``，输出 logits ``[B, T, V]``。训练时使用
    ``logits[:, :-1]`` 预测 ``labels[:, 1:]``；MoE 模型还会将各层辅助损失
    以 ``0.01`` 的系数加入语言模型损失。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
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

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        past_key_values: Optional[Sequence[tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        """执行 decoder 前向传播。

        Inputs:
            input_ids: token ids，shape [B, T]。
            labels: 可选监督 token ids，shape [B, T]。
            past_key_values: 可选逐层 cache，每层 K/V shape
                [B, n_kv_heads, T_cache, D]。
        Outputs:
            ``ModelOutput.logits`` shape [B, T, V]；当启用 cache 时，
            每层 present 的 K/V shape 为 [B, n_kv_heads, T_cache + T, D]。
        """
        hidden_states = self.embed_tokens(input_ids)  # [B, T] -> [B, T, H]
        presents = []
        aux_losses = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            hidden_states, present, aux_loss = layer(hidden_states, past, use_cache)
            if use_cache:
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
            )
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
    ) -> Tensor:
        """使用 temperature、top-k、top-p 和逐层 KV cache 自回归生成。

        Inputs:
            input_ids: prompt token ids，shape [B, T_prompt]。
        Outputs:
            包含 prompt 与新 token 的序列，shape [B, T_prompt + T_new]。
        """
        self.eval()
        generated = input_ids
        past_key_values = None
        for _ in range(max_new_tokens):
            current = generated if past_key_values is None else generated[:, -1:]  # 首轮 [B, T]，后续 [B, 1]
            output = self(
                current,
                past_key_values=past_key_values,
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
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))  # 过滤低于第 k 大值的 logits
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)  # [B, V]
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    logits.scatter_(
                        1, sorted_indices, sorted_logits.masked_fill(remove, float("-inf"))
                    )
                next_token = torch.multinomial(logits.softmax(dim=-1), 1)
            generated = torch.cat((generated, next_token), dim=-1)  # [B, T] + [B, 1] -> [B, T+1]
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return generated


class ByteTokenizer:
    """无依赖的 UTF-8 字节级 tokenizer。

    token ``0/1/2`` 分别保留给 PAD/BOS/EOS，真实字节 ``0..255`` 映射到
    ``3..258``，因此词表大小为 ``259``。编码结果为一维 token id 列表，
    由调用方堆叠为模型所需的 ``[B, T]``。
    """

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
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


def build_training_batch(
    tokenizer: ByteTokenizer,
    text: str,
    batch_size: int,
    seq_len: int,
    step: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """从循环文本中抽取固定长度的 teacher-forcing batch。

    Args:
        tokenizer: 提供 ``encode`` 的 tokenizer。
        text: 用于教学训练的文本。
        batch_size: batch 大小 B。
        seq_len: 每条序列长度 T。
        step: 当前训练步，用于移动窗口起点。
        device: 输出 tensor 所在设备。

    Returns:
        ``inputs`` 和 ``labels``，二者 shape 均为 [B, T]；模型内部再将
        logits 与 labels 做一个 token 的 causal shift。
    """
    ids = torch.tensor(tokenizer.encode(text, add_bos=True, add_eos=True), device=device)  # [N]
    if ids.numel() < seq_len + 2:
        repeats = math.ceil((seq_len + 2) / ids.numel())
        ids = ids.repeat(repeats)  # [N] -> [N * repeats]
    max_start = ids.numel() - seq_len - 1
    starts = [(step * batch_size * seq_len + i * seq_len) % max_start for i in range(batch_size)]
    inputs = torch.stack([ids[start : start + seq_len] for start in starts])  # [B, T]
    # Keep labels aligned with inputs; QwenCausalLM.forward performs the
    # standard one-token causal shift for the loss.
    labels = torch.stack([ids[start : start + seq_len] for start in starts])  # [B, T]
    return inputs, labels
