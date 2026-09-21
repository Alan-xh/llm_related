"""DN-DETR 教学实现。

训练时将带噪标签/框编码为 ``[B,G*M,D]`` denoising query，并与正常
``[B,Q,D]`` query 拼接；推理时仅保留正常 query 分支。
"""
