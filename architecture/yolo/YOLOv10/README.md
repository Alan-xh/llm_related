# YOLOv10

YOLOv10 的核心教学点是端到端、NMS-free 的检测思路：训练时保留 one-to-many
分支帮助密集监督，同时提供 one-to-one 分支用于推理，从而减少对后处理 NMS 的
依赖。本实现用两个同构的小 head 展示这个接口。

## 来源

- 论文：[YOLOv10: Real-Time End-to-End Object Detection](https://arxiv.org/abs/2405.14458)
- 代码：[THU-MIG/yolov10](https://github.com/THU-MIG/yolov10)

## 数据流与 Shape

```text
image                  [B, 3, 64, 64]
shared CSP + neck      P3/P4/P5
one_to_many heads      three [B, 5+C, H_i, W_i] tensors
one_to_one heads       three [B, 5+C, H_i, W_i] tensors
inference              decode one_to_one, skip NMS
```

## 核心公式

```text
L_train = L_one_to_many
L_infer = decode(one_to_one)
```

这里的 `one_to_one` 分支只是端到端接口的最小抽象；代码没有实现官方完整的
consistent dual assignments、rank-guided assignment 或完整损失配方。命令行推理
通过 `nms_free=True` 保留置信度最高的候选，不调用 NMS。

## 运行

```bash
python architecture/yolo/YOLOv10/model.py
python architecture/yolo/YOLOv10/train.py --steps 5 --checkpoint yolov10_tiny.pt
python architecture/yolo/YOLOv10/inference.py --checkpoint yolov10_tiny.pt
```

