# YOLO 系列架构

本目录提供 YOLOv1、YOLOv3、YOLOv4、YOLOv5、YOLOv8、YOLOv10 和 YOLO26 的轻量
PyTorch 教学实现。代码重点是把各代模型的结构变化放进统一的目标检测数据流中，便于
对照阅读和在 CPU 上运行；它们不是官方仓库的 checkpoint 兼容实现，也没有复现完整
数据增强、数据集标注格式、EMA、分布式训练或部署算子。

## 目录

| 版本 | 主要机制 | 输出/推理特点 |
| --- | --- | --- |
| [`YOLOv1/`](./YOLOv1/) | 单网格、anchor-based、直接回归 | `[B, S, S, C+10]`，推理后解码框 |
| [`YOLOv3/`](./YOLOv3/) | Darknet 风格多尺度预测 | 三个 stride 的检测层 |
| [`YOLOv4/`](./YOLOv4/) | CSP backbone、SPP/PAN 思路 | 多尺度特征融合 |
| [`YOLOv5/`](./YOLOv5/) | C3、SPPF、PAN 和工程化训练接口 | 小模型训练/推理入口 |
| [`YOLOv8/`](./YOLOv8/) | anchor-free、解耦头和 DFL | 4×`reg_max` 分布回归通道 |
| [`YOLOv10/`](./YOLOv10/) | one-to-many + one-to-one 双头 | one-to-one 分支可不做 NMS |
| [`YOLO26/`](./YOLO26/) | 端到端双头、渐进式监督教学抽象 | 默认使用 one-to-one 分支推理 |

除 YOLOv1 外，教学版检测头统一输出三层：

```text
images                 [B, 3, H, W]
backbone features      [B, 4W, H/8, W/8]
                       [B, 8W, H/16, W/16]
                       [B, 8W, H/32, W/32]
neck features          同上，经过 top-down 融合
head output            [B, 5+C, H/stride, W/stride]
                       或 [B, 4*reg_max+1+C, H/stride, W/stride]
decoded detections     [N, 6] = [x1, y1, x2, y2, score, class_id]
```

这里的训练 target 是教学用 `[tx, ty, tw, th, objectness, one_hot_classes]` 网格张量。
`tx, ty, tw, th` 在 `[0, 1]` 范围内，公共训练循环用一个彩色矩形合成数据集验证前向、
反向和 checkpoint 保存，不需要下载数据。

## 共同公式

普通多尺度头将每个网格位置解码为：

```text
d = softplus(box_logits)
c = (grid_x + 0.5, grid_y + 0.5) * stride
box = [c_x - d_l, c_y - d_t, c_x + d_r, c_y + d_b] * stride
score = sigmoid(objectness) * sigmoid(class_logits)
```

YOLOv8 的 DFL 风格回归把每条边表示为 `reg_max` 个离散 bin：

```text
d = sum_j softmax(z_j) * j,  j = 0 ... reg_max-1
```

教学版基础损失为：

```text
L = 5 * SmoothL1(box, target) + BCE(objectness, target)
    + BCE(class_logits, one_hot_class)
```

YOLOv1 使用单独的简化版损失，包含坐标、objectness、no-object 和类别项。YOLOv10/
YOLO26 的双头训练对 one-to-many 分支计算主损失，YOLO26 额外以较小权重计算
one-to-one 分支的渐进式监督。

## 来源

- YOLOv1：`You Only Look Once: Unified, Real-Time Object Detection`
  ([arXiv](https://arxiv.org/abs/1506.02640))
- YOLOv3：`YOLOv3: An Incremental Improvement`
  ([arXiv](https://arxiv.org/abs/1804.02767))
- YOLOv4：`YOLOv4: Optimal Speed and Accuracy of Object Detection`
  ([arXiv](https://arxiv.org/abs/2004.10934))
- YOLOv5：Ultralytics YOLOv5
  ([GitHub](https://github.com/ultralytics/yolov5))
- YOLOv8：Ultralytics YOLO
  ([GitHub](https://github.com/ultralytics/ultralytics))
- YOLOv10：`YOLOv10: Real-Time End-to-End Object Detection`
  ([arXiv](https://arxiv.org/abs/2405.14458))
- YOLO26：Ultralytics YOLO26 文档与实现
  ([Docs](https://docs.ultralytics.com/models/yolo26/))

## 运行

从仓库根目录运行结构检查：

```bash
python architecture/yolo/YOLOv1/model.py
python architecture/yolo/YOLOv3/model.py
python architecture/yolo/YOLOv8/model.py
python architecture/yolo/YOLOv10/model.py
python architecture/yolo/YOLO26/model.py
```

训练五步合成数据：

```bash
python architecture/yolo/YOLOv8/train.py --steps 5 --checkpoint yolov8_tiny.pt
```

加载 checkpoint 并推理：

```bash
python architecture/yolo/YOLOv8/inference.py \
  --checkpoint yolov8_tiny.pt --confidence 0.25
python architecture/yolo/YOLOv10/inference.py --confidence 0.25
python architecture/yolo/YOLO26/inference.py --confidence 0.25
```

公共模块 `architecture.yolo.common` 提供 `decode_predictions`、`nms`、
`yolo_loss`、`yolo_v1_loss` 和合成数据训练工具。根目录的 `model.py`、`train.py`、
`inference.py` 保留一个简洁的多尺度 YOLO 风格兼容入口。

