# SAM 系列架构

本目录提供 8 个可运行的 PyTorch 教学实现。代码共享 [`common.py`](./common.py)，每个版本目录都包含：

```text
model.py       模型结构、损失函数或关键数据变换
train.py       合成矩形数据上的最小训练循环
inference.py   单图、视频或多目标推理入口
README.md      来源、数据流、Shape、公式和运行命令
```

| 目录 | 任务 | 主要教学重点 |
| --- | --- | --- |
| [`SAM/`](./SAM/) | 点/框 prompt 图像分割 | ViT、随机 Fourier prompt、Two-Way Transformer、动态 mask token |
| [`MobileSAM/`](./MobileSAM/) | 轻量图像分割 | depthwise-separable student 与 SAM 接口复用 |
| [`FastSAM/`](./FastSAM/) | 实例候选选择 | 全图候选 mask、prompt 排序与候选 IoU |
| [`SAM-HQ/`](./SAM-HQ/) | 高质量掩码 | 高分辨率特征注入 mask decoder |
| [`SAM2/`](./SAM2/) | 图像/视频分割 | streaming memory、memory attention 和跨帧状态 |
| [`SAM2.1/`](./SAM2.1/) | SAM 2 改进接口 | 更大的有界 memory 与 checkpoint-compatible API 形态 |
| [`SAM3/`](./SAM3/) | 概念分割 | 文本、示例框、点 prompt、presence head |
| [`SAM3.1/`](./SAM3.1/) | 多目标概念分割 | shared image pass、对象 token multiplex 和身份维度 |

## 共同数据流

```text
image / video frame
        ↓
image encoder → image embedding [B,C,H/8,W/8]
        ↓                         ↑
prompt encoder → sparse/dense prompt embeddings
        ↓
Two-Way Transformer → mask tokens → hypernetwork masks + IoU scores
                                      ↑
                         video memory / concept tokens
```

默认教学配置为 `image_size=64`、`embed_dim=64`、`patch_size=8`，因此 CPU smoke test 很快。它们不是官方模型的参数规模、权重格式或训练数据复刻。

## 统一接口

单图模型：

```python
masks, scores = model(
    images,                         # [B,3,H,W]
    point_coords=points,             # [B,N,2], pixel [x,y]
    point_labels=labels,             # [B,N], 1 foreground, 0 background, -1 padding
    boxes=boxes,                     # optional [B,M,4], xyxy
)
# masks: [B,K,H,W], scores: [B,K]
```

SAM 2 使用 `predict_frame(...)` 返回 `(masks, scores, state)`；SAM 3 返回 `masks`、`iou_scores`、`presence_logits` 的字典；SAM 3.1 的 `multiplex(...)` 返回 `[B,O,K,H,W]` 的对象维度。

## 推荐阅读顺序

```text
SAM → SAM-HQ / MobileSAM / FastSAM → SAM2 → SAM2.1 → SAM3 → SAM3.1
```

各版本的论文、官方/作者代码仓库、Shape 表和公式见对应 README。所有入口都可以从仓库根目录直接运行，例如：

```bash
python architecture/SAM/SAM/model.py
python architecture/SAM/SAM2/inference.py --frames 4
python architecture/SAM/SAM3.1/inference.py --prompts rectangle object
```
