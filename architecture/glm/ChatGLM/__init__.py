"""ChatGLM 双语对话教学实现。

模型主干输入 token ids shape 为 [B, T]，输出 logits shape 为 [B, T, V]；
对话角色模板由公共 ``format_chat`` 组织。
"""
