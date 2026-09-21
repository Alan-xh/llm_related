"""DAB-DETR 教学实现。

每个 query 携带动态 ``cxcywh`` anchor，使用 box sine embedding 注入
decoder，并按 ``sigmoid(inv_sigmoid(b)+Δb)`` 逐层更新框位置。
"""
