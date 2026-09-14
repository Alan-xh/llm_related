# YOLO26

本目录把 YOLO26 表达为一个适合架构学习的当前代端到端检测抽象：共享多尺度
特征、one-to-many 密集监督、one-to-one 无 NMS 推理，以及额外的渐进式辅助监督。
它不是官方模型权重或算子级复刻，目的是让代码中的训练/推理契约清楚可见。

## 来源

- 官方模型文档：[Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- 官方实现：[ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

## 数据流与 Shape

```text
image                  [B, 3, 64, 64]
shared CSP + neck      P3/P4/P5
one_to_many heads      [B, 5+C, 8, 8], [B, 5+C, 4, 4], [B, 5+C, 2, 2]
one_to_one heads       同上
inference              decode one_to_one, no NMS
```

## 核心公式

```text
L = L_many + 0.25 * L_one
```

其中 `L_many` 提供密集正样本信号，`L_one` 让推理分支直接学习稀疏输出。这个
教学版本将 `YOLO26Config.progressive_loss` 设为 `True`，并复用可读的
box/objectness/class loss；官方实现中的训练细节、任务头和部署优化没有在这里展开。

## 运行

```bash
python architecture/yolo/YOLO26/model.py
python architecture/yolo/YOLO26/train.py --steps 5 --checkpoint yolo26_tiny.pt
python architecture/yolo/YOLO26/inference.py --checkpoint yolo26_tiny.pt
```

