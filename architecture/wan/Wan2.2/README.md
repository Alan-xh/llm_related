# Wan2.2

Wan2.2 教学实现把 Wan2.2 的两个关键方向压缩成一个 CPU 可运行示例：高噪声/低噪声
阶段的双专家扩散 Transformer，以及高压缩视频 VAE 驱动的 TI2V 流程。它不兼容官方
Wan checkpoint，重点是观察条件、路由和张量 Shape。

## 来源

- 论文：[Wan2.2: Training and Inference of Video Generation Models](https://arxiv.org/abs/2503.20314)
- 开源代码：[Alibaba Wan2.2](https://github.com/Wan-Video/Wan2.2)

## 数据流与 Shape

```text
video                    [B, 3, T, H, W]
high-compression VAE     [B, 4, ceil(T/2), ceil(H/8), ceil(W/8)]
3D patchify               [B, N, 4 * pt * ph * pw]
text encoder              [B, 64, 64]
TI2V condition            first-frame latent padded to video latent grid
MoE DiT                   high-noise expert <-> low-noise expert
VAE decode                [B, 3, T, H, W]
```

当 `t >= 0.5` 时以 high-noise expert 为主，否则以 low-noise expert 为主；
learned router 作为平滑门控的一部分。Flow Matching 目标为：

```text
x_t = (1 - t) * z_0 + t * epsilon
u_t = epsilon - z_0
L = mean(|| u_theta(x_t, t, text, image) - u_t ||^2)
```

## 运行

```bash
python architecture/wan/Wan2.2/model.py
python architecture/wan/Wan2.2/train.py --steps 1 --checkpoint wan22_tiny.pt
python architecture/wan/Wan2.2/inference.py --steps 4 --output wan22.pt
```

推理入口保存 `[B, 3, T, H, W]` tensor；第一帧来自 TI2V 条件，其余帧由 MoE DiT 积分生成。
