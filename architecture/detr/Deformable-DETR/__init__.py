"""Deformable-DETR 教学实现。

核心机制为多尺度特征 ``[B,D,H_l,W_l]``、reference point 和少量
``grid_sample`` 采样点；最终输出 ``[B,Q,K+1]`` 与 ``[B,Q,4]``。
"""
