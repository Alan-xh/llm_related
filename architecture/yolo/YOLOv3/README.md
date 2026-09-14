# YOLOv3

YOLOv3 引入三个尺度的检测层，以覆盖小、中、大目标。本教学实现保留三层
feature pyramid 和显式的经典 anchor 配置；为了让 box 解码代码更短，实际 head
使用统一的 anchor-free 距离表示。

## 来源

- 论文：[YOLOv3: An Incremental Improvement](https://arxiv.org/abs/1804.02767)
- 代码参考：[pjreddie/darknet](https://github.com/pjreddie/darknet)

## 数据流与 Shape

```text
image             [B, 3, 64, 64]
backbone P3       [B, 4W, 8, 8]
backbone P4       [B, 8W, 4, 4]
backbone P5       [B, 8W, 2, 2]
neck              P3/P4/P5 top-down fusion
outputs           [B, 5+C, 8, 8], [B, 5+C, 4, 4], [B, 5+C, 2, 2]
```

## 核心公式

```text
score = sigmoid(objectness) * sigmoid(class_logits)
L = 5 * SmoothL1(box, target) + BCE(objectness) + BCE(class)
```

多尺度的 stride 分别为 `8, 16, 32`。`decode_predictions` 将距离回归恢复为
`xyxy`，再执行按类别 NMS。

## 运行

```bash
python architecture/yolo/YOLOv3/model.py
python architecture/yolo/YOLOv3/train.py --steps 5 --checkpoint yolov3_tiny.pt
python architecture/yolo/YOLOv3/inference.py --checkpoint yolov3_tiny.pt
```

