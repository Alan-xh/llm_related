# Wan2.1

Wan2.1 教学实现把视频生成拆成四个容易观察的部分：3D causal VAE、时空 patch
token、带文本 cross-attention 的 Flow Matching DiT，以及 T2V/I2V 采样入口。代码是
CPU 可运行的 tiny 模型，不兼容官方 Wan checkpoint，也不追求官方画质。

## 来源

- 论文：[Wan: Open and Advanced Large-Scale Video Generative Models](https://arxiv.org/abs/2411.17420)
- 开源代码：[Alibaba Wan2.1](https://github.com/Wan-Video/Wan2.1)

## 数据流与 Shape

```text
video                  [B, 3, T, H, W]
causal video VAE        [B, 4, ceil(T/2), ceil(H/4), ceil(W/4)]
3D patchify             [B, N, 4 * pt * ph * pw]
T5-like text encoder    [B, 64, 64]
Flow Matching DiT       [B, 4, ceil(T/2), ceil(H/4), ceil(W/4)]
VAE decode              [B, 3, T, H, W]
```

默认 patch size 是 `(1, 2, 2)`。`t=0` 表示干净 latent，`t=1` 表示高斯噪声：

```text
x_t = (1 - t) * z_0 + t * epsilon
u_t = epsilon - z_0
L = mean(|| u_theta(x_t, t, text) - u_t ||^2)
x_{t+dt} = x_t + dt * u_theta(x_t, t, text)
```

## 运行

```bash
python architecture/wan/Wan2.1/model.py
python architecture/wan/Wan2.1/train.py --steps 1 --checkpoint wan21_tiny.pt
python architecture/wan/Wan2.1/inference.py --mode t2v --steps 4 --output wan21.pt
python architecture/wan/Wan2.1/inference.py --mode i2v --prompt "red motion"
```

推理入口保存的是 `[B, 3, T, H, W]` 的 PyTorch tensor，便于继续接入视频编码器或可视化工具。
