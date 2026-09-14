# SAM 2.1

## 来源

- 官方发布与说明：<https://github.com/facebookresearch/sam2>
- 版本定位：SAM 2 的改进 checkpoint、训练/微调和 demo 接口；本目录沿用 SAM 2 的公开论文和 memory 设计。

## 数据流

```text
frame → image encoder → [B,C,H/8,W/8]
prompt / previous mask → prompt encoder
current feature ↔ bounded streaming memory → mask decoder
                                      ↓
                              masks + quality scores
```

与 `SAM2` 的教学差异是使用更大的默认 memory 窗口 `max_memory_frames=6`，并通过 `improved=True` 标记版本配置；主调用方式保持兼容。

## Shape 与公式

默认输入 `[B,3,64,64]`，frame feature `[B,64,8,8]`，memory key `[B,1,64]`，输出 masks `[B,3,64,64]` 和 scores `[B,3]`。

```text
feature' = LayerNorm(feature + MultiHeadAttention(feature, memory, memory))
L = 20 * BCE(mask, target) + Dice(mask, target) + MSE(score, true_iou)
```

## 运行

```bash
python architecture/SAM/SAM2.1/model.py
python architecture/SAM/SAM2.1/train.py --steps 5 --checkpoint sam21_tiny.pt
python architecture/SAM/SAM2.1/inference.py --frames 4 --checkpoint sam21_tiny.pt
```

本目录关注 checkpoint/config 组织和稳定的推理接口，不伪造官方权重或宣称与官方 checkpoint 可直接加载。
