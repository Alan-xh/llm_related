"""
09_StructuredOutput.py
======================
vLLM 结构化输出(Guided Decoding):约束 LLM 输出符合 JSON Schema / 正则 / CFG / Choice。

核心机制:
    在采样前,把"语法合法的下一个 token 集合"作为 mask 应用到 logits 上,
    非法 token 的 logit 设为 -inf,从而保证采样结果永远合法。

后端实现:
    - outlines:基于 FSM(finite state machine),把正则/JSON 编译成状态机
    - xgrammar:基于 CFG,JSON schema 加速
    - llguidance:更通用的 CFG 引擎

本文实现:
    1. 简单 Choice 约束
    2. 正则约束(基于字符级 FSM)
    3. JSON Schema 约束(简化)
    4. CFG 约束示例
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Callable
import re
import json
import torch


# ============================================================
# 1. LogitsProcessor 接口
# ============================================================

class LogitsProcessor:
    """vLLM 的 LogitsProcessor 接口"""

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        """根据已生成 token_ids,修改并返回 logits"""
        raise NotImplementedError


# ============================================================
# 2. Choice 约束(从候选列表中选)
# ============================================================

class ChoiceLogitsProcessor(LogitsProcessor):
    """
    约束输出必须是 choices 之一。
    实现:在生成开始时,确定哪些 token 是某个 choice 的合法前缀。
    """

    def __init__(self, choices: List[str], tokenizer_encode: Callable[[str], List[int]]):
        self.choices = choices
        self.tokenizer_encode = tokenizer_encode
        # 每个 choice 编码成 token 序列
        self.choice_token_ids = [tokenizer_encode(c) for c in choices]
        self.state = 0  # 已匹配的 token 数

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        # 当前在哪个 choice 上
        valid_ids: Set[int] = set()
        for ct in self.choice_token_ids:
            if len(ct) > self.state:
                # 已生成的 token 必须匹配该 choice 的前缀
                if token_ids[-self.state:] == ct[:self.state] if self.state else True:
                    valid_ids.add(ct[self.state])
        mask = torch.full_like(logits, float("-inf"))
        for tid in valid_ids:
            mask[tid] = 0
        return logits + mask


# ============================================================
# 3. 正则约束(基于 FSM)
# ============================================================

class RegexFSM:
    """
    把正则编译成有限状态机(FSM),每个状态对应一组合法字符。
    简化版:只支持字符级(实际 outlines 用 byte-level)
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.compiled = re.compile(pattern)
        self.state = ""  # 当前已生成字符串

    def allowed_next_chars(self) -> Set[str]:
        """返回当前状态下,所有合法的下一个字符"""
        allowed = set()
        # 暴力枚举 ASCII 字符
        for c in map(chr, range(32, 127)):
            if self.compiled.fullmatch(self.state + c, partial=True):
                allowed.add(c)
        return allowed

    def advance(self, char: str):
        self.state += char


class RegexLogitsProcessor(LogitsProcessor):
    """
    把 FSM 的合法字符集转成 token mask。
    简化:假设 tokenizer 是 char-level(实际需要处理 byte-pair)
    """

    def __init__(self, pattern: str, char_to_token_id: Dict[str, int]):
        self.fsm = RegexFSM(pattern)
        self.char_to_id = char_to_token_id

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        allowed_chars = self.fsm.allowed_next_chars()
        allowed_ids = {self.char_to_id[c] for c in allowed_chars
                       if c in self.char_to_id}
        mask = torch.full_like(logits, float("-inf"))
        for tid in allowed_ids:
            mask[tid] = 0
        return logits + mask


# ============================================================
# 4. JSON Schema 约束(简化版)
# ============================================================

class JSONSchemaLogitsProcessor(LogitsProcessor):
    """
    约束输出符合给定 JSON Schema。
    简化:只演示 {"key": value} 这种结构。

    实际 xgrammar 把 schema 编译成增量 parser,
    每生成一个 token 推进 parser 状态。
    """

    def __init__(self, schema: Dict):
        self.schema = schema
        self.target_str = self._schema_to_template(schema)
        self.position = 0  # 在 target_str 中的位置
        self.in_value = False  # 是否在写 value
        self.value_type = None

    def _schema_to_template(self, schema: Dict) -> str:
        """把 schema 转成模板字符串(实际 xgrammar 用增量 parser)"""
        if schema.get("type") == "object":
            props = schema.get("properties", {})
            return "{" + ",".join(f'"{k}":__' for k in props) + "}"
        return ""

    def __call__(self, token_ids: List[int], logits: torch.Tensor) -> torch.Tensor:
        # 简化:强制按模板输出,非 __ 位置必须是模板字符
        # __ 位置可以自由生成(但受 schema type 约束)
        # 真实实现:每个 __ 对应一个 schema type,对应 token mask
        return logits  # 这里简化


# ============================================================
# 5. 演示:Regex 约束生成
# ============================================================

def demo_regex():
    """演示用正则约束生成 phone number: \\d{3}-\\d{4}"""
    pattern = r"\d{3}-\d{4}"

    # char-level tokenizer
    chars = [chr(i) for i in range(32, 127)]
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for c, i in char_to_id.items()}
    vocab_size = len(chars)

    proc = RegexLogitsProcessor(pattern, char_to_id)

    # 模拟生成
    torch.manual_seed(0)
    generated = ""
    for step in range(8):
        logits = torch.randn(vocab_size) * 0.1  # 模拟 LLM logits
        masked = proc([], logits)
        # greedy argmax
        next_id = masked.argmax().item()
        # 处理 -inf
        if masked[next_id].item() == float("-inf"):
            print(f"step {step}: no valid token!")
            break
        next_char = id_to_char[next_id]
        generated += next_char
        proc.fsm.advance(next_char)
        print(f"step {step}: picked '{next_char}', current='{generated}'")

    print(f"\nFinal: '{generated}', matches: {bool(re.fullmatch(pattern, generated))}")


def demo_choice():
    """演示 Choice 约束"""

    # 模拟 tokenizer:用单词字符
    vocab = ["red", "blue", "green", "yellow", "cat", "dog"]
    word_to_ids = {w: [i] for i, w in enumerate(vocab)}
    encode = lambda s: [vocab.index(s)] if s in vocab else []

    proc = ChoiceLogitsProcessor(["red", "blue", "green"], encode)
    logits = torch.zeros(len(vocab))

    # 第一次:合法 token = red[0], blue[0], green[0] (即 0, 1, 2)
    masked = proc([], logits.clone())
    valid = (masked > float("-inf")).nonzero().flatten().tolist()
    print(f"Step 0 valid token ids: {valid} (expect [0, 1, 2] for red/blue/green)")

    # 假设选了 red (id=0)
    proc.state = 1
    # 模拟后续(简化:choice 已完整)
    print("Selected: red")


def demo_json():
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
    proc = JSONSchemaLogitsProcessor(schema)
    print(f"JSON schema template: {proc.target_str}")
    print(f"  (实际 xgrammar 会用增量 parser 逐 token 推进)")


if __name__ == "__main__":
    print("=" * 60)
    print("Demo 1: Regex Constrained Generation")
    print("=" * 60)
    demo_regex()
    print()
    print("=" * 60)
    print("Demo 2: Choice Constrained Generation")
    print("=" * 60)
    demo_choice()
    print()
    print("=" * 60)
    print("Demo 3: JSON Schema Constraint")
    print("=" * 60)
    demo_json()
