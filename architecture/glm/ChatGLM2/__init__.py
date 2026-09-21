"""ChatGLM2 MQA 对话教学实现。

默认使用 4 个 query head 和 1 个 KV head，单层 KV cache shape 为
[B, 1, T_cache, D]。
"""
