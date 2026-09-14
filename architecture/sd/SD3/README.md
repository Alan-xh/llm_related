# Stable Diffusion 3

本例用 MMDiT 风格 diffusion Transformer 替代 U-Net：图像 latent token 和文本 token 先独立
投影，再拼接进入共享 attention；训练使用 flow matching，推理使用 Euler 积分。

## 来源

- 论文：[Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206)
- 开源实现：[Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)

## 数据流与 Shape

```text
image                  [B, 3, 32, 32]
VAE latent             [B, 4, 4, 4]
image tokens           [B, 16, 4] -> [B, 16, 128]
text encoder           [B, 64, 80] -> [B, 64, 128]
joint attention        concat -> [B, 80, 128]
image output           [B, 16, 4] -> [B, 4, 4, 4]
```

## 核心公式

令 `t=0` 为 clean data、`t=1` 为 Gaussian noise：

```text
x_t = (1 - t) * z_0 + t * epsilon
u_t = epsilon - z_0
L = mean(|| u_theta(x_t, t, c) - u_t ||^2)
x_{t+dt} = x_t + dt * u_theta(x_t, t, c)
```

采样从 `t=1` 反向积分到 `t=0`，并在每一步使用：

```text
guided = u_uncond + s * (u_cond - u_uncond)
```

## 运行

```bash
python architecture/sd/SD3/model.py
python architecture/sd/SD3/train.py --steps 5 --checkpoint sd3_tiny.pt
python architecture/sd/SD3/inference.py --steps 8 --guidance-scale 4.5 --output sd3.png
```
