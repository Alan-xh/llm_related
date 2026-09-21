"""DETR 系列教学架构包。

任务范围:
    提供 DETR、Deformable-DETR、DAB-DETR、DN-DETR、DINO、RT-DETR 和
    Grounding-DINO 的紧凑 PyTorch 实现。各版本统一使用图像
    ``[B,3,H,W]``、预测 logits ``[B,Q,K+1]`` 和归一化框
    ``[B,Q,4]`` 的基础契约；Grounding-DINO 额外使用文本 token ``[B,T]``。

公共组件:
    ``common.py`` 提供 backbone、位置编码、Hungarian matching、集合损失、
    合成数据、训练循环和推理解码工具。
"""
