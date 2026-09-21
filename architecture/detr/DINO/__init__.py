"""DINO 教学实现。

先从 encoder proposal ``[B,L,D]`` 按前景分数选择 top-k query，再加入
contrastive denoising query，输出正常检测分支和训练期 DN 分支。
"""
