# RT-DETR

## 来源

- 论文：[DETRs Beat YOLOs on Real-time Object Detection](https://arxiv.org/abs/2304.08069)
- 官方代码：[PaddlePaddle/PaddleDetection](https://github.com/PaddlePaddle/PaddleDetection)

## 数据流与 Shape

```text
image [B,3,H,W]
  -> multi-scale CNN [B,d,H/8,W/8], [B,d,H/16,W/16], [B,d,H/32,W/32]
  -> hybrid encoder -> flattened memory [B,S,d]
  -> encoder class/box/IoU quality
  -> top-k IoU-aware query selection [B,N,d]
  -> Transformer decoder
  -> logits [B,N,K+1], boxes [B,N,4]
```

其中 `S` 是多尺度空间位置总数。query 由 encoder proposal 直接初始化，
减少了传统 DETR 对大量 decoder query 的依赖。

## 核心公式

```text
quality_i = max_c p_i(c) * IoU_i
Q = TopK_i(quality_i)
L = L_set + lambda_iou * L_IoU
```

教学代码返回 `pred_iou` 和 encoder proposals，主集合损失仍使用 Hungarian
matching；推理不使用 NMS。

## 教学实现边界

本实现展示多尺度特征、轻量 hybrid encoder、IoU-aware top-k selection 和
decoder；以 `grid`/Transformer 友好的 PyTorch 模块替代官方高性能 backbone、
部署 engine、TensorRT 加速和 COCO 评测。

## 运行

```bash
python architecture/detr/RT-DETR/model.py
python architecture/detr/RT-DETR/train.py --steps 2 --checkpoint rt_detr_tiny.pt
python architecture/detr/RT-DETR/inference.py --checkpoint rt_detr_tiny.pt
```
