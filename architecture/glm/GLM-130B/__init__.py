"""GLM-130B blank-infilling 教学实现。

二维 position ids shape 为 [B, 2, T]，prefix-visible attention 和
``-100`` loss mask 用于表达空白填充目标。
"""
