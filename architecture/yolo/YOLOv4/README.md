# YOLOv4

YOLOv4 的教学重点是 CSP 风格的部分跨阶段连接、SPP/PAN 类特征融合以及更强的
训练/工程组合。本目录用轻量 CSP block 和 top-down neck 展示这些结构关系。

## 来源

- 论文：[YOLOv4: Optimal Speed and Accuracy of Object Detection](https://arxiv.org/abs/2004.10934)
- 代码参考：[AlexeyAB/darknet](https://github.com/AlexeyAB/darknet)

## 数据流与 Shape

```text
image             [B, 3, 64, 64]
CSP backbone P3   [B, 4W, 8, 8]
CSP backbone P4   [B, 8W, 4, 4]
SPP/CSP P5        [B, 8W, 2, 2]
PAN-like neck     [P3, P4, P5]
heads             [B, 5+C, 8, 8], [B, 5+C, 4, 4], [B, 5+C, 2, 2]
```

## 核心公式

```text
P4 = Conv(P4 + Upsample(P5))
P3 = Conv(P3 + Upsample(P4))
L = lambda_box * L_box + L_obj + L_cls
```

代码使用 SiLU、BatchNorm 和可读性优先的简化损失，没有实现完整的
CIoU、Mosaic、Self-Adversarial Training 等官方训练细节。

## 运行

```bash
python architecture/yolo/YOLOv4/model.py
python architecture/yolo/YOLOv4/train.py --steps 5 --checkpoint yolov4_tiny.pt
python architecture/yolo/YOLOv4/inference.py --checkpoint yolov4_tiny.pt
```

