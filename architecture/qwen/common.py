"""Small, self-contained building blocks shared by the Qwen teaching models.

The code intentionally targets clarity and CPU-sized experiments. It is not a
checkpoint-compatible implementation of any released Qwen model.
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
    """The subset of a causal LM output needed by the demos."""

    logits: Tensor
    loss: Optional[Tensor] = None
    past_key_values: Optional[tuple[tuple[Tensor, Tensor], ...]] = None
    aux_loss: Optional[Tensor] = None


@dataclass
class ModelConfig:
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
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads")
        if self.num_experts_per_tok > self.num_experts:
            raise ValueError("num_experts_per_tok cannot exceed num_experts")


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        variance = x.float().pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps).to(x.dtype)
        return self.weight * x


def rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


class RotaryEmbedding(nn.Module):
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
        positions = torch.arange(length, device=self.inv_freq.device).float()
        positions = positions / self.scaling
        freqs = torch.outer(positions, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("_cos", emb.cos()[None, None], persistent=False)
        self.register_buffer("_sin", emb.sin()[None, None], persistent=False)

    def forward(
        self, q: Tensor, k: Tensor, position_offset: int = 0
    ) -> tuple[Tensor, Tensor]:
        needed = position_offset + q.shape[-2]
        if needed > self._cos.shape[-2]:
            self._set_cache(max(needed, 2 * self._cos.shape[-2]))
        cos = self._cos[..., position_offset:needed, :].to(dtype=q.dtype)
        sin = self._sin[..., position_offset:needed, :].to(dtype=q.dtype)
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k


def repeat_kv(hidden_states: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return hidden_states
    batch, heads, seq_len, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, heads, repeats, seq_len, head_dim
    )
    return hidden_states.reshape(batch, heads * repeats, seq_len, head_dim)


class SelfAttention(nn.Module):
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
        batch, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(batch, query_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(hidden_states).view(batch, query_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        value = self.v_proj(hidden_states).view(batch, query_length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        query = self.q_norm(query)
        key = self.k_norm(key)

        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        query, key = self.rotary(query, key, past_length)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)
        present = (key, value) if use_cache else None

        key_for_attention = repeat_kv(key, self.num_kv_groups)
        value_for_attention = repeat_kv(value, self.num_kv_groups)
        scores = torch.matmul(query, key_for_attention.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        total_length = key.shape[-2]
        if query_length > 1 or past_length == 0:
            query_positions = torch.arange(
                past_length, past_length + query_length, device=hidden_states.device
            )
            key_positions = torch.arange(total_length, device=hidden_states.device)
            future = key_positions[None, :] > query_positions[:, None]
            scores = scores.masked_fill(future[None, None], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.matmul(weights, value_for_attention)
        output = output.transpose(1, 2).contiguous().view(batch, query_length, -1)
        return self.o_proj(output), present


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoEBlock(nn.Module):
    """Token-level top-k MoE used to make routing visible in a tiny model."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.router = nn.Linear(config.hidden_size, config.num_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(config.hidden_size, config.intermediate_size) for _ in range(config.num_experts)]
        )
        self.num_experts_per_tok = config.num_experts_per_tok
        self.num_experts = config.num_experts

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        flat = x.reshape(-1, x.shape[-1])
        router_logits = self.router(flat)
        weights, indices = torch.topk(router_logits, self.num_experts_per_tok, dim=-1)
        weights = F.softmax(weights.float(), dim=-1).to(flat.dtype)
        result = torch.zeros_like(flat)
        for expert_id, expert in enumerate(self.experts):
            token_index, slot = torch.where(indices == expert_id)
            if token_index.numel():
                result.index_add_(
                    0,
                    token_index,
                    expert(flat[token_index]) * weights[token_index, slot, None],
                )
        # Switch-style auxiliary loss encourages balanced expert usage.
        probabilities = router_logits.softmax(dim=-1).mean(dim=0)
        counts = F.one_hot(indices[:, 0], self.num_experts).float().mean(dim=0)
        aux_loss = self.num_experts * torch.sum(probabilities * counts)
        return result.view_as(x), aux_loss


class TransformerBlock(nn.Module):
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
        attention_output, present = self.self_attn(
            self.input_layernorm(x), past_key_value, use_cache
        )
        x = x + attention_output
        mlp_input = self.post_attention_layernorm(x)
        if isinstance(self.mlp, MoEBlock):
            mlp_output, aux_loss = self.mlp(mlp_input)
        else:
            mlp_output, aux_loss = self.mlp(mlp_input), None
        return x + mlp_output, present, aux_loss


class QwenCausalLM(nn.Module):
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
        hidden_states = self.embed_tokens(input_ids)
        presents = []
        aux_losses = []
        for index, layer in enumerate(self.layers):
            past = None if past_key_values is None else past_key_values[index]
            hidden_states, present, aux_loss = layer(hidden_states, past, use_cache)
            if use_cache:
                presents.append(present)
            if aux_loss is not None:
                aux_losses.append(aux_loss)
        logits = self.lm_head(self.norm(hidden_states))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
            )
            if aux_losses:
                loss = loss + 0.01 * torch.stack(aux_losses).mean()
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
        self.eval()
        generated = input_ids
        past_key_values = None
        for _ in range(max_new_tokens):
            current = generated if past_key_values is None else generated[:, -1:]
            output = self(
                current,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = output.past_key_values
            logits = output.logits[:, -1]
            if temperature <= 0:
                next_token = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    values, _ = torch.topk(logits, min(top_k, logits.shape[-1]))
                    logits = logits.masked_fill(logits < values[:, [-1]], float("-inf"))
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    logits.scatter_(
                        1, sorted_indices, sorted_logits.masked_fill(remove, float("-inf"))
                    )
                next_token = torch.multinomial(logits.softmax(dim=-1), 1)
            generated = torch.cat((generated, next_token), dim=-1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return generated


class ByteTokenizer:
    """A dependency-free tokenizer for runnable architecture demonstrations."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = False) -> list[int]:
        ids = [byte + 3 for byte in text.encode("utf-8")]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
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
    ids = torch.tensor(tokenizer.encode(text, add_bos=True, add_eos=True), device=device)
    if ids.numel() < seq_len + 2:
        repeats = math.ceil((seq_len + 2) / ids.numel())
        ids = ids.repeat(repeats)
    max_start = ids.numel() - seq_len - 1
    starts = [(step * batch_size * seq_len + i * seq_len) % max_start for i in range(batch_size)]
    inputs = torch.stack([ids[start : start + seq_len] for start in starts])
    # Keep labels aligned with inputs; QwenCausalLM.forward performs the
    # standard one-token causal shift for the loss.
    labels = torch.stack([ids[start : start + seq_len] for start in starts])
    return inputs, labels
