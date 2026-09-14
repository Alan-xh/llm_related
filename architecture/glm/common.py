"""Small, self-contained building blocks for GLM teaching implementations.

The code intentionally favors readable tensor transformations over speed or
checkpoint compatibility.  It demonstrates the parts that distinguish the
GLM family: autoregressive blank infilling, two-dimensional positions,
prefix-visible attention, KV caching, chat templates, and tool-call markers.
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
    logits: Tensor
    loss: Optional[Tensor] = None
    past_key_values: Optional[tuple[tuple[Tensor, Tensor], ...]] = None
    aux_loss: Optional[Tensor] = None


@dataclass
class GLMConfig:
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
        if self.num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        if self.hidden_size % self.num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.num_heads % self.num_kv_heads:
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
        return self.weight * (x * torch.rsqrt(variance + self.eps).to(x.dtype))


def rotate_half(x: Tensor) -> Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
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
        positions = torch.arange(length, device=self.inv_freq.device).float()
        positions = positions / self.scaling
        frequencies = torch.outer(positions, self.inv_freq)
        embeddings = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("_cos", embeddings.cos()[None, None], persistent=False)
        self.register_buffer("_sin", embeddings.sin()[None, None], persistent=False)

    def forward(
        self,
        query: Tensor,
        key: Tensor,
        position_ids: Optional[Tensor] = None,
        position_offset: int = 0,
    ) -> tuple[Tensor, Tensor]:
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
        cos = self._cos[0, 0, positions].unsqueeze(1).to(query.dtype)
        sin = self._sin[0, 0, positions].unsqueeze(1).to(query.dtype)
        return query * cos + rotate_half(query) * sin, key * cos + rotate_half(key) * sin


def repeat_kv(hidden_states: Tensor, repeats: int) -> Tensor:
    if repeats == 1:
        return hidden_states
    batch, heads, seq_len, head_dim = hidden_states.shape
    expanded = hidden_states[:, :, None].expand(batch, heads, repeats, seq_len, head_dim)
    return expanded.reshape(batch, heads * repeats, seq_len, head_dim)


def _prefix_attention_mask(
    query_length: int,
    total_length: int,
    past_length: int,
    prefix_length: Optional[int | Tensor],
    device: torch.device,
) -> Tensor:
    query_positions = torch.arange(
        past_length, past_length + query_length, device=device
    )[None, :, None]
    key_positions = torch.arange(total_length, device=device)[None, None, :]
    allowed = key_positions <= query_positions
    if prefix_length is not None:
        if isinstance(prefix_length, int):
            prefix = torch.full(
                (1, 1, 1), prefix_length, device=device, dtype=torch.long
            )
        else:
            prefix = prefix_length.to(device=device, dtype=torch.long).view(-1, 1, 1)
        allowed = allowed | (
            (query_positions < prefix) & (key_positions < prefix)
        )
    return allowed


class SelfAttention(nn.Module):
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
        batch, query_length, _ = hidden_states.shape
        query = self.q_proj(hidden_states).view(
            batch, query_length, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key = self.k_proj(hidden_states).view(
            batch, query_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        value = self.v_proj(hidden_states).view(
            batch, query_length, self.num_kv_heads, self.head_dim
        ).transpose(1, 2)
        query, key = self.q_norm(query), self.k_norm(key)
        past_length = 0 if past_key_value is None else past_key_value[0].shape[-2]
        query, key = self.rotary(query, key, position_ids, past_length)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)
            value = torch.cat((past_key_value[1], value), dim=-2)

        key_for_attention = repeat_kv(key, self.num_kv_groups)
        value_for_attention = repeat_kv(value, self.num_kv_groups)
        scores = torch.matmul(query, key_for_attention.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        allowed = _prefix_attention_mask(
            query_length,
            key.shape[-2],
            past_length,
            prefix_length,
            hidden_states.device,
        )
        scores = scores.masked_fill(~allowed[:, None], torch.finfo(scores.dtype).min)
        weights = F.softmax(scores.float(), dim=-1).to(scores.dtype)
        weights = F.dropout(weights, p=self.dropout, training=self.training)
        output = torch.matmul(weights, value_for_attention)
        output = output.transpose(1, 2).contiguous().view(batch, query_length, -1)
        present = (key, value) if use_cache else None
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
        flat = x.reshape(-1, x.shape[-1])
        router_logits = self.router(flat)
        weights, indices = torch.topk(
            router_logits, self.num_experts_per_tok, dim=-1
        )
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
        probabilities = router_logits.softmax(dim=-1).mean(dim=0)
        assignments = F.one_hot(indices[:, 0], self.num_experts).float().mean(dim=0)
        aux_loss = self.num_experts * torch.sum(probabilities * assignments)
        return result.view_as(x), aux_loss


class TransformerBlock(nn.Module):
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
        attention_output, present = self.self_attn(
            self.input_layernorm(x),
            position_ids,
            past_key_value,
            prefix_length,
            use_cache,
        )
        x = x + attention_output
        mlp_input = self.post_attention_layernorm(x)
        if isinstance(self.mlp, MoEBlock):
            mlp_output, aux_loss = self.mlp(mlp_input)
        else:
            mlp_output, aux_loss = self.mlp(mlp_input), None
        return x + mlp_output, present, aux_loss


class GLMCausalLM(nn.Module):
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
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _default_position_ids(
        self, input_ids: Tensor, past_length: int
    ) -> Tensor:
        batch, seq_len = input_ids.shape
        absolute = torch.arange(
            past_length, past_length + seq_len, device=input_ids.device
        ).expand(batch, -1)
        if not self.config.use_2d_position_ids:
            return absolute
        block = torch.zeros_like(absolute)
        return torch.stack((absolute, block), dim=1)

    def forward(
        self,
        input_ids: Tensor,
        labels: Optional[Tensor] = None,
        position_ids: Optional[Tensor] = None,
        prefix_length: Optional[int | Tensor] = None,
        past_key_values: Optional[Sequence[tuple[Tensor, Tensor]]] = None,
        use_cache: bool = False,
    ) -> ModelOutput:
        past_length = (
            0 if past_key_values is None else past_key_values[0][0].shape[-2]
        )
        if position_ids is None:
            position_ids = self._default_position_ids(input_ids, past_length)
        hidden_states = self.embed_tokens(input_ids)
        if self.block_position_embeddings is not None:
            block_positions = (
                position_ids[:, 1]
                if position_ids.dim() == 3
                else torch.zeros_like(input_ids)
            )
            hidden_states = hidden_states + self.block_position_embeddings(
                block_positions.clamp_max(
                    self.config.max_block_position_embeddings - 1
                )
            )

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

        logits = self.lm_head(self.norm(hidden_states))
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.shape[-1]),
                shift_labels.view(-1),
                ignore_index=-100,
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
        prefix_length: Optional[int] = None,
    ) -> Tensor:
        self.eval()
        generated = input_ids
        prompt_length = generated.shape[1] if prefix_length is None else prefix_length
        past_key_values = None
        for _ in range(max_new_tokens):
            current = (
                generated
                if past_key_values is None
                else generated[:, -1:]
            )
            output = self(
                current,
                past_key_values=past_key_values,
                prefix_length=prompt_length,
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
                    logits = logits.masked_fill(
                        logits < values[:, [-1]], float("-inf")
                    )
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(
                        logits, descending=True
                    )
                    cumulative = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                    remove = cumulative > top_p
                    remove[:, 1:] = remove[:, :-1].clone()
                    remove[:, 0] = False
                    logits.scatter_(
                        1,
                        sorted_indices,
                        sorted_logits.masked_fill(remove, float("-inf")),
                    )
                next_token = torch.multinomial(logits.softmax(dim=-1), 1)
            generated = torch.cat((generated, next_token), dim=-1)
            if eos_token_id is not None and bool((next_token == eos_token_id).all()):
                break
        return generated


class ByteTokenizer:
    """A dependency-free tokenizer for tiny CPU-sized architecture demos."""

    pad_token_id = 0
    bos_token_id = 1
    eos_token_id = 2
    vocab_size = 259

    def encode(
        self, text: str, add_bos: bool = True, add_eos: bool = False
    ) -> list[int]:
        ids = [byte + 3 for byte in text.encode("utf-8")]
        if add_bos:
            ids.insert(0, self.bos_token_id)
        if add_eos:
            ids.append(self.eos_token_id)
        return ids

    def decode(self, ids: Sequence[int]) -> str:
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
    ids = torch.tensor(
        tokenizer.encode(text, add_bos=True, add_eos=True), device=device
    )
    if ids.numel() < seq_len + 2:
        ids = ids.repeat(math.ceil((seq_len + 2) / ids.numel()))
    max_start = max(1, ids.numel() - seq_len - 1)
    starts = [
        (step * batch_size * seq_len + index * seq_len) % max_start
        for index in range(batch_size)
    ]
    inputs = torch.stack([ids[start : start + seq_len] for start in starts])
    return inputs, inputs.clone(), None


def build_blank_infilling_batch(
    tokenizer: ByteTokenizer,
    text: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, int]:
    """Build [context, target] examples for the GLM blank-infilling objective."""

    payload = tokenizer.encode(text, add_bos=False, add_eos=False)
    if len(payload) < 8:
        payload = payload * math.ceil(8 / len(payload))
    split_start = max(2, len(payload) // 3)
    split_end = min(len(payload) - 2, split_start + max(2, len(payload) // 4))
    prefix = payload[:split_start]
    target = payload[split_start:split_end]
    suffix = payload[split_end:]
    blank = tokenizer.encode("<BLANK>", add_bos=False)
    sequence = [tokenizer.bos_token_id] + prefix + blank + suffix + target
    sequence = sequence[: max(seq_len, len(sequence))]
    sequence = sequence[:seq_len]
    if len(sequence) < seq_len:
        sequence.extend([tokenizer.pad_token_id] * (seq_len - len(sequence)))
    prefix_length = min(
        seq_len,
        1 + len(prefix) + len(blank) + len(suffix),
    )
    inputs = torch.tensor([sequence] * batch_size, device=device)
    labels = torch.full_like(inputs, -100)
    if prefix_length < seq_len:
        labels[:, prefix_length:] = inputs[:, prefix_length:]
    return inputs, labels, prefix_length


def format_chat(
    messages: Sequence[dict[str, str]],
    system_prompt: Optional[str] = None,
    thinking: bool = False,
    tools: Optional[Sequence[dict]] = None,
) -> str:
    """Use a readable ChatGLM-style role format for the byte tokenizer."""

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
    """Extract the first JSON object following a tool-call marker."""

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
    target_device = torch.device(device)
    tokenizer = ByteTokenizer()
    model = model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for step in range(steps):
        if blank_infilling:
            inputs, labels, prefix_length = build_blank_infilling_batch(
                tokenizer, text, batch_size, seq_len, target_device
            )
        else:
            inputs, labels, prefix_length = build_causal_batch(
                tokenizer, text, batch_size, seq_len, step, target_device
            )
        output = model(inputs, labels=labels, prefix_length=prefix_length)
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
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    output_ids = model.generate(
        input_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        eos_token_id=tokenizer.eos_token_id,
    )
    print(tokenizer.decode(output_ids[0].tolist()))


def train_cli(
    build_model: Callable[[], GLMCausalLM],
    default_text: str,
    blank_infilling: bool = False,
) -> None:
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
