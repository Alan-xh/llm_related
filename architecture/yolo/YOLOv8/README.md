# YOLOv8

YOLOv8 的教学重点是 anchor-free 检测、解耦检测头和 Distribution Focal Loss
风格的离散距离回归。每个网格位置预测四条边的离散分布，而不是直接预测四个
anchor 参数。

## 来源

- 代码：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)
- 文档：[Ultralytics YOLO Docs](https://docs.ultralytics.com/models/yolov8/)

## 数据流与 Shape

默认 `reg_max=8`：

```text
image             [B, 3, 64, 64]
CSP backbone      P3/P4/P5
decoupled head    [B, 4*8 + 1 + C, 8, 8]
                  [B, 4*8 + 1 + C, 4, 4]
                  [B, 4*8 + 1 + C, 2, 2]
DFL decode        [B, points, 4] distances
```

## 核心公式

```text
p_j = softmax(z)_j
d = sum(j * p_j), j = 0 ... reg_max-1
box = [cx-d_l, cy-d_t, cx+d_r, cy+d_b]
```

公共 `yolo_loss(..., regression_bins=8)` 会先把离散分布转换为期望距离，再计算
教学版 SmoothL1。真实训练还包含更细致的正样本分配和 IoU/DFL 组合损失。

## 运行

```bash
python architecture/yolo/YOLOv8/model.py
python architecture/yolo/YOLOv8/train.py --steps 5 --checkpoint yolov8_tiny.pt
python architecture/yolo/YOLOv8/inference.py --checkpoint yolov8_tiny.pt
```

