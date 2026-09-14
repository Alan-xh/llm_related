# SAM 2

## 来源

- 论文：Ravi et al., *SAM 2: Segment Anything in Images and Videos*，2024，<https://arxiv.org/abs/2408.00714>
- 官方实现：<https://github.com/facebookresearch/sam2>

## 数据流

```text
current frame → image encoder → frame feature
prompt + previous mask → prompt encoder
frame feature + session memory → memory attention
                         ↓
                   mask decoder → mask + object score
                         ↓
                   memory append → bounded VideoMemory
```

单图是 memory 为空的特殊情况。视频路径使用 `predict_frame` 逐帧更新状态，并限制最多保存 `max_memory_frames` 个 memory key。

## Shape 表

| 节点 | Shape |
| --- | --- |
| Frame | `[B,3,64,64]` |
| Frame feature | `[B,64,8,8]` |
| Memory key | `[B,1,64]` |
| Decoder output | `[B,3,64,64]`, `[B,3]` |
| Video state | keys/masks/frame_indices 三个有界列表 |

## 核心公式

memory attention 使用：

```text
Q = flatten(current_feature)
K,V = concat(previous_memory_keys)
feature' = LayerNorm(Q + Attention(Q,K,V))
```

掩码训练损失仍为 `20 * BCE + Dice + IoU-MSE`。

## 运行

```bash
python architecture/SAM/SAM2/model.py
python architecture/SAM/SAM2/train.py --steps 5 --checkpoint sam2_tiny.pt
python architecture/SAM/SAM2/inference.py --frames 4 --checkpoint sam2_tiny.pt
```

该实现展示 streaming state 和交互式 re-prompt 的接口，不复刻官方 Hiera、memory encoder 或视频数据引擎。
