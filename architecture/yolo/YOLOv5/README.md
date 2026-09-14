# YOLOv5

YOLOv5 将检测模型做成配置化、易训练和易部署的工程系统。本实现聚焦其中最容易
观察的结构：C3/CSP block、SPPF 和多尺度 PAN 风格融合，并提供统一的训练/推理
命令行。

## 来源

- 代码：[ultralytics/yolov5](https://github.com/ultralytics/yolov5)
- 文档：[Ultralytics YOLO Docs](https://docs.ultralytics.com/)

## 数据流与 Shape

```text
image             [B, 3, 64, 64]
C3 backbone P3    [B, 4W, 8, 8]
C3 backbone P4    [B, 8W, 4, 4]
SPPF backbone P5  [B, 8W, 2, 2]
PAN neck          [P3, P4, P5]
prediction        [B, 5+C, H/stride, W/stride]
```

## 核心公式

```text
SPPF(x) = Conv([x, MaxPool(x), MaxPool^2(x), MaxPool^3(x)])
score = sigmoid(obj) * sigmoid(cls)
```

检测损失沿用公共教学版 box/objectness/class 三项。`YOLOv5Config` 中的
`depth_multiple` 和 `width_multiple` 用于说明官方配置缩放概念；本最小实现采用
固定小宽度，避免 CPU 烟测过慢。

## 运行

```bash
python architecture/yolo/YOLOv5/model.py
python architecture/yolo/YOLOv5/train.py --steps 5 --checkpoint yolov5_tiny.pt
python architecture/yolo/YOLOv5/inference.py --checkpoint yolov5_tiny.pt
```

