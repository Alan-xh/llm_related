# DAB-DETR

## 来源

- 论文：[DAB-DETR: Dynamic Anchor Boxes are Better Queries for DETR](https://arxiv.org/abs/2201.12329)
- 代码参考：[IDEA-Research/DAB-DETR](https://github.com/IDEA-Research/DAB-DETR)

## 数据流与 Shape

```text
image [B,3,H,W]
  -> Tiny CNN + sine position -> memory [B,L,d]
  -> dynamic anchor boxes [B,N,4]
  -> box sine embedding [B,N,d]
  -> Transformer decoder + iterative box refinement
  -> logits [B,N,K+1], boxes [B,N,4]
  -> Hungarian matching + set loss
```

每个 query 不再只有一条自由的内容向量，而是同时携带归一化
`cx,cy,w,h` anchor；每层 decoder 用预测增量更新 anchor。

## 核心公式

```text
q_pos = BoxSineEmbedding(anchor)
b^(l+1) = sigmoid(inv_sigmoid(b^l) + Delta b^l)
L = L_cls + lambda_L1 * L_box + lambda_giou * L_giou
```

## 教学实现边界

本实现保留动态 anchor、box positional encoding 和逐层 refinement，使用小型
CNN/Transformer 与合成矩形数据；不复刻官方 ResNet、COCO pipeline、多尺度
训练和 checkpoint 兼容。

## 运行

```bash
python architecture/detr/DAB-DETR/model.py
python architecture/detr/DAB-DETR/train.py --steps 2 --checkpoint dab_detr_tiny.pt
python architecture/detr/DAB-DETR/inference.py --checkpoint dab_detr_tiny.pt
```
