"""原始 DETR 教学实现。

核心机制为固定 object queries、Transformer encoder-decoder、Hungarian
matching 和 set-based loss；模型输入 ``[B,3,H,W]``，输出
``[B,Q,K+1]`` 分类 logits 与 ``[B,Q,4]`` 归一化 ``cxcywh`` 框。
"""
