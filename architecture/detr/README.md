# DETR 系列架构

本目录按 DETR 系列的主要架构演进拆分。每个版本目录包含对应的
`model.py`、`train.py`、`inference.py` 和张量流动说明；公共的
Hungarian matching、框工具、合成矩形数据和训练工具位于
[`common.py`](./common.py)。

| 目录 | 代表机制 | 主要创新技术 |
| --- | --- | --- |
| [`DETR/`](./DETR/) | 集合预测 | object queries + Hungarian matching，去除 anchor 与 NMS |
| [`Deformable-DETR/`](./Deformable-DETR/) | 多尺度可变形注意力 | 只对参考点附近的稀疏位置采样，降低注意力开销 |
| [`DAB-DETR/`](./DAB-DETR/) | 动态锚框查询 | 让 query 携带动态 anchor box，并逐层迭代框位置 |
| [`DN-DETR/`](./DN-DETR/) | 去噪训练 | 加入带噪标签/框的 denoising query，加快收敛 |
| [`DINO/`](./DINO/) | 改进去噪锚框 | 改进 query selection、对比式去噪和框迭代 |
| [`RT-DETR/`](./RT-DETR/) | 实时检测 Transformer | 高效混合编码器 + IoU-aware query selection |
| [`Grounding-DINO/`](./Grounding-DINO/) | 开放词汇检测 | 文本-图像跨模态融合，语言引导目标定位 |

## 教学实现约定

- 所有示例默认使用小型 CNN/Transformer 和 `64x64` 合成矩形数据，便于在 CPU
  上观察完整的前向、匹配、损失和反向传播。
- 检测框统一使用归一化 `cx, cy, w, h`；模型输出分类 logits 和框回归，
  推理解码为 `[x1, y1, x2, y2, score, label]`。
- 每个版本保留论文中的关键机制，但不复刻官方 backbone、COCO 数据管道、
  CUDA 算子、分布式训练、checkpoint 兼容和评测脚本。

## 快速运行

```bash
python architecture/detr/DETR/model.py
python architecture/detr/DETR/train.py --steps 2 --checkpoint detr_tiny.pt
python architecture/detr/DETR/inference.py --checkpoint detr_tiny.pt
```

其余版本使用相同的 `model.py`、`train.py`、`inference.py` 入口；版本 README
会列出它们的特有参数和 Tensor Shape。
