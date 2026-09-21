"""RT-DETR 教学实现。

混合多尺度 encoder 产生 ``[B,S,D]`` memory，并依据
``max_class_prob * predicted_iou`` 选择 top-k query 后执行 decoder。
"""
