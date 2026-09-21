"""Grounding DINO 教学实现。

视觉 query ``[B,Q,D]`` 与文本 token 特征 ``[B,T,D]`` 做余弦相似度，
输出 token-level logits ``[B,Q,T]``、框 ``[B,Q,4]`` 和 objectness。
"""
