# DETR

## 论文与官方代码

- 论文：[End-to-End Object Detection with Transformers](https://arxiv.org/abs/2005.12872)
- 官方代码：[facebookresearch/detr](https://github.com/facebookresearch/detr)
- 论文要点：把目标检测建模为集合预测，用固定数量的 object queries 和 Hungarian matching 产生一组无重复预测，推理阶段不需要 anchor 和 NMS。

## 数据流

```text
图像 [B, 3, H, W]
  -> CNN backbone
  -> 特征图 [B, C, H/32, W/32]
  -> 1x1 conv 投影 + 2D positional encoding
  -> 展平为 [B, HW/1024, d]
  -> Transformer encoder
  -> Transformer decoder(object queries)
  -> [B, N, d]
  -> 分类头 / 边界框头
  -> 类别 [B, N, K+1]、框 [B, N, 4]
  -> Hungarian matching + set-based loss
```

其中 `N` 是 query 数量（原论文通常为 100），`K` 是目标类别数，`K+1` 包含 no-object；框采用归一化 `cx, cy, w, h`。

## Tensor Shape

以 ResNet-50、`d=256`、单尺度 stride=32 为例：

| 张量 | Shape | 说明 |
| --- | --- | --- |
| 输入图像 | `[B, 3, H, W]` | batch 内可 padding |
| backbone 特征 | `[B, 2048, H/32, W/32]` | 最后一层特征 |
| 投影后特征 | `[B, 256, H/32, W/32]` | `1x1` 卷积 |
| encoder 输入/输出 | `[B, L, 256]` | `L=(H/32)(W/32)` |
| query embedding | `[N, 256]` | 学习得到 |
| decoder 输出 | `[B, N, 256]` | 每个 query 一个候选 |
| 分类 logits | `[B, N, K+1]` | 含 no-object |
| 边界框 | `[B, N, 4]` | 归一化 `cxcywh` |
| 目标标签/框 | `[B, M]` / `[B, M, 4]` | 每张图有效目标数可不同 |

## 核心公式

给定预测集合 `y_hat` 和标注集合 `y`，先求一一匹配：

```text
sigma* = argmin_sigma sum_i C(y_i, y_hat_{sigma(i)})
C = -log p_hat(c_i) + lambda_L1 ||b_i - b_hat||_1
    + lambda_giou (1 - GIoU(b_i, b_hat))
```

匹配后计算集合损失：

```text
L = L_cls + lambda_L1 L_box + lambda_giou L_giou
L_cls = -sum_i log p_hat_{sigma*(i)}(c_i)
```

训练使用 Hungarian algorithm；未匹配 query 的类别为 `no-object`，通常对该类别使用较低权重。

## 教学实现边界

- 实现最小闭环：CNN backbone、2D 位置编码、Transformer encoder-decoder、object queries、分类/框回归头、Hungarian matching 和损失。
- 默认支持单尺度特征、固定 query 数量和 COCO 风格 `[x, y, w, h]` 标注；为便于阅读，可使用 ResNet-50 或轻量 CNN。
- 不复刻官方全部数据增强、分布式训练、混合精度、checkpoint 兼容、COCO 评估细节和多尺度变体。
- 不把 NMS、anchor 生成或候选框筛选加入主流程；它们不是原始 DETR 的核心路径。

## 运行命令

```bash
python architecture/detr/DETR/model.py
python architecture/detr/DETR/train.py --steps 2 --checkpoint detr_tiny.pt
python architecture/detr/DETR/inference.py --checkpoint detr_tiny.pt
```

本仓库命令使用 CPU 友好的合成矩形数据。官方仓库的分布式训练入口则是：

```bash
python -m torch.distributed.launch --nproc_per_node=8 \
  --use_env main.py --coco_path /path/to/coco
```
