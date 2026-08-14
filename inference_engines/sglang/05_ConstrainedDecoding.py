"""
05_ConstrainedDecoding.py
=========================
SGLang 约束解码:xgrammar 深度集成,支持 JSON Schema / 正则 / CFG。

对比 vLLM 的 outlines:
    - SGLang 与 xgrammar 团队深度合作,JSON schema 预编译为 DFA
    - Composable Grammar:JSON schema + 正则 + 自由文本混合,parser 跨 token 共享状态
    - 跳过 logits 的语法 prefilter:在不读 logits 的情况下排除大量非法 token

核心机制:
    1. 把 grammar(JSON schema / regex / CFG)编译成增量 parser
    2. 维护 parser 状态(当前在哪个 production rule)
    3. 每生成一个 token,计算合法的下一个 token 集合
    4. 用 bitmap 表示合法 token,apply 到 logits

xgrammar 加速:
    - JSON schema -> 编译成 BNF -> DFA
    - 跨 token 共享 parser 状态(不重新 parse 整个序列)
    - 用 bitset 加速 mask 计算
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Any
import json
import re


# ============================================================
# 1. Grammar 抽象基类
# ============================================================

class Grammar:
    """所有 grammar 的基类"""

    def get_allowed_tokens(self, current_state: Any,
                            vocab: List[str]) -> Set[int]:
        """根据当前 parser 状态,返回合法 token id 集合"""
        raise NotImplementedError

    def advance(self, current_state: Any, token: str) -> Any:
        """消费一个 token,返回新状态"""
        raise NotImplementedError

    def initial_state(self) -> Any:
        raise NotImplementedError


# ============================================================
# 2. Regex Grammar(基于 FSM)
# ============================================================

class RegexGrammar(Grammar):
    """
    正则编译成 FSM。
    简化:字符级,每个状态对应已生成的字符串。
    实际 xgrammar 用 byte-level + DFA minimization。
    """

    def __init__(self, pattern: str):
        self.pattern = pattern
        self.compiled = re.compile(pattern)

    def initial_state(self) -> str:
        return ""

    def get_allowed_tokens(self, current_state: str, vocab: List[str]) -> Set[int]:
        allowed = set()
        for i, tok in enumerate(vocab):
            # 简化:每个 token 是单字符
            if len(tok) == 1 and self.compiled.fullmatch(current_state + tok, partial=True):
                allowed.add(i)
        return allowed

    def advance(self, current_state: str, token: str) -> str:
        return current_state + token


# ============================================================
# 3. JSON Schema Grammar(基于增量 parser)
# ============================================================

class JSONSchemaGrammar(Grammar):
    """
    JSON Schema 约束:用增量 parser 跟踪当前在 schema 中的位置。

    简化实现:状态 = (在模板中的位置, 当前期望的类型)
    实际 xgrammar 把 schema 编译成 BNF grammar,再用 DFA 推进。
    """

    def __init__(self, schema: Dict):
        self.schema = schema
        # 把 schema 转成模板字符串 + 类型标注
        self.template, self.types = self._compile_schema(schema)

    def _compile_schema(self, schema: Dict) -> tuple:
        """
        编译 schema 为模板。
        简化:只支持 object + string/integer 字段。
        实际 xgrammar 支持完整 JSON schema 规范。
        """
        if schema.get("type") == "object":
            props = schema.get("properties", {})
            parts = []
            types = []
            parts.append("{")
            for i, (k, v) in enumerate(props.items()):
                if i > 0:
                    parts.append(",")
                parts.append(f'"{k}":')
                parts.append("__VALUE__")  # 占位
                types.append(v.get("type", "string"))
            parts.append("}")
            return "".join(parts), types
        return "", []

    def initial_state(self) -> tuple:
        return (0, 0)  # (template 位置, 当前 value 类型 index)

    def get_allowed_tokens(self, state: tuple, vocab: List[str]) -> Set[int]:
        pos, val_idx = state
        allowed = set()

        if pos >= len(self.template):
            return allowed

        # 在 __VALUE__ 位置:允许符合类型的 token
        if self.template[pos:pos+9] == "__VALUE__":
            vtype = self.types[val_idx] if val_idx < len(self.types) else "string"
            for i, tok in enumerate(vocab):
                if self._is_valid_for_type(tok, vtype):
                    allowed.add(i)
        else:
            # 在固定字符位置:只允许该字符
            expected_char = self.template[pos]
            for i, tok in enumerate(vocab):
                if tok == expected_char:
                    allowed.add(i)
        return allowed

    def advance(self, state: tuple, token: str) -> tuple:
        pos, val_idx = state
        if pos >= len(self.template):
            return state

        if self.template[pos:pos+9] == "__VALUE__":
            # 在 value 位置消费 token
            # 简化:遇到引号或逗号表示 value 结束
            if token in [",", "}"]:
                pos += 9  # 跳过 __VALUE__
                val_idx += 1
                # 继续消费这个 token
                if token == self.template[pos:pos+1]:
                    pos += 1
            # else: 继续 value,不前进
        else:
            if token == self.template[pos:pos+1]:
                pos += 1
        return (pos, val_idx)

    def _is_valid_for_type(self, tok: str, vtype: str) -> bool:
        if vtype == "integer":
            return tok.isdigit() or tok == "-"
        elif vtype == "string":
            # string 必须以 " 开头
            return tok == '"' or tok.isalnum()
        return True


# ============================================================
# 4. Composable Grammar(组合)
# ============================================================

class ComposableGrammar(Grammar):
    """
    组合多个 grammar:JSON schema + 正则 + 自由文本
    实际 xgrammar 支持 CFG 上下文无关文法的组合
    """

    def __init__(self, grammars: List[Grammar]):
        self.grammars = grammars

    def initial_state(self) -> List:
        return [g.initial_state() for g in self.grammars]

    def get_allowed_tokens(self, state: List, vocab: List[str]) -> Set[int]:
        # 取所有 grammar 允许 token 的交集
        allowed = None
        for g, s in zip(self.grammars, state):
            t = g.get_allowed_tokens(s, vocab)
            allowed = t if allowed is None else (allowed & t)
        return allowed or set()

    def advance(self, state: List, token: str) -> List:
        return [g.advance(s, token) for g, s in zip(self.grammars, state)]


# ============================================================
# 5. Guided Decoding 主循环
# ============================================================

def guided_generate(grammar: Grammar,
                    vocab: List[str],
                    prompt: str,
                    max_tokens: int = 50,
                    mock_logits_fn=None) -> str:
    """
    用 grammar 约束的生成主循环。
    mock_logits_fn: callable,返回每个 token 的 logits(模拟 LLM)
    """
    state = grammar.initial_state()
    generated = ""

    for _ in range(max_tokens):
        # 1. 获取合法 token 集合
        allowed = grammar.get_allowed_tokens(state, vocab)
        if not allowed:
            print(f"No valid token, stopping. generated='{generated}'")
            break

        # 2. 模拟 LLM 输出 logits,选合法 token 中 logits 最大的
        if mock_logits_fn:
            logits = mock_logits_fn()
            masked = [(i, logits[i]) for i in allowed]
            best = max(masked, key=lambda x: x[1])[0]
        else:
            best = next(iter(allowed))

        tok = vocab[best]
        generated += tok

        # 3. 推进 parser 状态
        state = grammar.advance(state, tok)

        # 4. 完成判断
        if isinstance(grammar, RegexGrammar) and \
                grammar.compiled.fullmatch(generated):
            break
        if isinstance(grammar, JSONSchemaGrammar) and \
                state[0] >= len(grammar.template):
            break

    return generated


# ============================================================
# 6. 演示
# ============================================================

def demo():
    # char-level vocab
    vocab = list('0123456789-":,{} abcdefghijklmnopqrstuvwxyz') + ['"', '}']

    # ---- Regex 约束:电话号码 ----
    phone_pattern = r"\d{3}-\d{4}"
    print(f"--- Regex: phone number {phone_pattern} ---")
    grammar = RegexGrammar(phone_pattern)
    out = guided_generate(grammar, vocab, "", max_tokens=8)
    is_match = bool(re.fullmatch(phone_pattern, out))
    print(f"Generated: '{out}', matches: {is_match}")

    # ---- JSON Schema 约束 ----
    print("\n--- JSON Schema ---")
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"}
        }
    }
    grammar = JSONSchemaGrammar(schema)
    print(f"Template: {grammar.template}")
    out = guided_generate(grammar, vocab, "", max_tokens=30)
    print(f"Generated: '{out}'")

    # ---- Composable: JSON + Regex ----
    print("\n--- Composable: JSON + Regex ---")
    # 组合(简化演示)
    composite = ComposableGrammar([JSONSchemaGrammar(schema)])
    out = guided_generate(composite, vocab, "", max_tokens=30)
    print(f"Generated: '{out}'")


if __name__ == "__main__":
    demo()
