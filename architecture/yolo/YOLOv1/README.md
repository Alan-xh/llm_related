# YOLOv1

YOLOv1 把整张图划分为 `S×S` 网格，每个网格预测 `B` 个边界框和类别概率。
本实现使用 `S=7`、`B=2`，保留“单次前向直接做集合回归”的核心教学接口。

## 来源

- 论文：[You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640)
- 代码参考：[pjreddie/darknet](https://github.com/pjreddie/darknet)

## 数据流与 Shape

```text
image                       [B, 3, 64, 64]
CNN backbone                [B, 8W, 2, 2]
adaptive average pooling    [B, 8W, 7, 7]
prediction                  [B, 7, 7, C + 2*5]
```

每个网格单元的布局为：

```text
[class_logits(C), box_1(x,y,w,h,confidence), box_2(x,y,w,h,confidence)]
```

`x,y` 是相对当前网格的偏移，`w,h` 是相对整张图的比例。推理时先取
`class_probability * confidence`，再将中心点和宽高转换为图像坐标。

## 核心公式

```text
L = lambda_coord * L_xywh
  + L_obj
  + lambda_noobj * L_noobj
  + L_class
```

代码中的 `yolo_v1_loss` 是可读性优先的 MSE 教学版本，默认
`lambda_coord=5`、`lambda_noobj=0.5`，没有实现原论文完整的 responsible-box
IoU 分配。

## 运行

```bash
python architecture/yolo/YOLOv1/model.py
python architecture/yolo/YOLOv1/train.py --steps 5 --checkpoint yolov1_tiny.pt
python architecture/yolo/YOLOv1/inference.py --checkpoint yolov1_tiny.pt
```

