"""
04_FrontendDSL.py
=================
SGLang 前端 DSL:Python 嵌入式 DSL,把 LLM 调用表达成可被 runtime 优化的程序。

核心 primitives:
    - gen(name, **kwargs):采样一段 token,存入变量
    - select(name, choices):在候选中选择(对每个候选跑一次 logits)
    - fork(n):并行分支,共享前缀(由 RadixAttention 自动复用)
    - image / video:多模态输入
    - gen_regex / gen_json:结构化输出

runtime 优化:
    - Primitive fusion:连续多个 gen 合并成一次 forward
    - Branch writeback:fork 分支的 KV 直接落到 RadixCache
    - Static analysis:静态分析每个变量位置,便于 select 复用

示例(完整 agent 流程):
    @sgl.function
    def agent(s, question):
        s += "You are a helpful assistant.\\n"
        s += "Question: " + question + "\\n"
        with s.fork(n=5):
            s += "Reasoning: " + sgl.gen("reason", max_tokens=128)
        s += "Best answer: " + sgl.select("reason", choices=[...])

本文实现:
    - State 容器(类似 sgl.State)
    - 各 primitive 函数
    - 简化 runtime(无真实 LLM,用 mock)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable
import random


# ============================================================
# 1. State:DSL 的运行时容器
# ============================================================

class State:
    """
    类似 SGLang 的 State,作为 DSL 的"上下文"。
    累积生成的文本 + 变量 + fork 分支。
    """

    def __init__(self, runtime: "MockRuntime"):
        self.runtime = runtime
        self.text: str = ""            # 累积文本
        self.variables: Dict[str, Any] = {}  # 命名变量
        self.fork_branches: List["State"] = []

    def __iadd__(self, other: str):
        """支持 s += "text" 语法"""
        self.text += other
        return self

    def gen(self, name: str, max_tokens: int = 32,
            stop: Optional[List[str]] = None) -> "State":
        """采样一段 token"""
        generated = self.runtime.generate(self.text, max_tokens=max_tokens, stop=stop)
        self.variables[name] = generated
        self.text += generated
        return self

    def select(self, name: str, choices: List[str]) -> "State":
        """
        在候选中选择:对每个候选计算 log-prob,选最高的
        实际 SGLang 复用前缀 KV,只需对每个候选的最后 token forward
        """
        best_choice = self.runtime.select(self.text, choices)
        self.variables[name] = best_choice
        self.text += best_choice
        return self

    def fork(self, n: int) -> "ForkContext":
        """并行分支:共享当前前缀,各分支独立生成"""
        return ForkContext(self, n)


class ForkContext:
    """with s.fork(n=3): 上下文管理器"""

    def __init__(self, parent: State, n: int):
        self.parent = parent
        self.n = n
        self.branches: List[State] = []

    def __enter__(self):
        for _ in range(self.n):
            branch = State(self.parent.runtime)
            branch.text = self.parent.text  # 共享前缀
            self.branches.append(branch)
        return self.branches

    def __exit__(self, *args):
        # 把分支结果回写到 parent
        self.parent.fork_branches = self.branches
        # 简化:选择第一个分支作为主路径(实际 SGLang 保留所有分支)
        if self.branches:
            self.parent.text = self.branches[0].text
            for b in self.branches:
                for k, v in b.variables.items():
                    self.parent.variables.setdefault(k, []).append(v)


# ============================================================
# 2. Mock Runtime(模拟 LLM)
# ============================================================

class MockRuntime:
    """模拟 LLM 推理 runtime"""

    def __init__(self, vocab_words: Optional[List[str]] = None):
        self.vocab_words = vocab_words or [
            "yes", "no", "maybe", "the", "answer", "is", "correct",
            "wrong", "because", "therefore", "however", "in", "conclusion"
        ]
        self.rng = random.Random(0)

    def generate(self, prompt: str, max_tokens: int = 32,
                 stop: Optional[List[str]] = None) -> str:
        """模拟生成"""
        words = []
        for _ in range(max_tokens):
            w = self.rng.choice(self.vocab_words)
            if stop and w in stop:
                break
            words.append(w)
        return " ".join(words)

    def select(self, prompt: str, choices: List[str]) -> str:
        """模拟选择:返回随机一个(实际用 log-prob 比较)"""
        return self.rng.choice(choices)


# ============================================================
# 3. @sgl.function 装饰器
# ============================================================

def function(func: Callable) -> Callable:
    """
    @sgl.function 装饰器:
        把 Python 函数变成 SGLang 程序。
        实际实现会做 trace + IR 编译,这里简化为直接调用。
    """
    def wrapper(*args, **kwargs):
        runtime = kwargs.pop("runtime", None) or MockRuntime()
        state = State(runtime)
        func(state, *args, **kwargs)
        return state
    return wrapper


# ============================================================
# 4. 示例程序
# ============================================================

@function
def multi_choice_qa(s, question: str):
    """多分支推理 + 选择最佳答案"""
    s += "You are a helpful assistant.\n"
    s += "Question: " + question + "\n"
    # 5 个并行分支,共享前缀
    with s.fork(n=5):
        for branch in s.fork_branches:
            branch += "Reasoning: "
            branch.gen("reason", max_tokens=20)
    s += "Best answer: "
    s.select("best", choices=["A", "B", "C", "D"])


@function
def json_extraction(s, text: str):
    """结构化抽取"""
    s += "Extract info from: " + text + "\n"
    s += "Name: "
    s.gen("name", max_tokens=5, stop=["\n"])
    s += " Age: "
    s.gen("age", max_tokens=3, stop=["\n"])


@function
def simple_chat(s, user_input: str):
    """简单对话"""
    s += "User: " + user_input + "\n"
    s += "Assistant: "
    s.gen("response", max_tokens=20)


# ============================================================
# 5. 演示
# ============================================================

def demo():
    runtime = MockRuntime()

    print("--- Simple Chat ---")
    state = simple_chat(user_input="Hello, how are you?", runtime=runtime)
    print(f"Full text:\n{state.text}")
    print(f"Variables: {state.variables}")

    print("\n--- Multi-Choice QA with Fork ---")
    state = multi_choice_qa(question="What is 2+2?", runtime=runtime)
    print(f"Final text:\n{state.text}")
    print(f"Variables: {state.variables}")
    if state.fork_branches:
        print(f"Fork branches generated: {len(state.fork_branches)}")
        for i, b in enumerate(state.fork_branches):
            print(f"  Branch {i} reasoning: {b.variables.get('reason', 'N/A')[:50]}...")

    print("\n--- JSON Extraction ---")
    state = json_extraction(text="John is 30 years old", runtime=runtime)
    print(f"Variables: {state.variables}")


if __name__ == "__main__":
    demo()
