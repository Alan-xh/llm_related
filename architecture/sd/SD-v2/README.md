# Stable Diffusion v2

本例在 v1 风格 latent U-Net 上展示 SD-v2 的主要接口变化：更宽的 OpenCLIP-like 文本上下文、
v-prediction，以及可选的深度条件输入。深度条件在示例中默认为零张量，便于先观察主干数据流。

## 来源

- 论文：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- 开源实现：[Stability-AI/stablediffusion](https://github.com/Stability-AI/stablediffusion)

## 数据流与 Shape

```text
image                    [B, 3, 32, 32]
VAE latent z_0           [B, 4, 4, 4]
optional depth           [B, 4, 4, 4]
U-Net input              concat(z_t, depth) -> [B, 8, 4, 4]
OpenCLIP-like context    [B, 64, 80]
U-Net output             [B, 4, 4, 4]
```

与 v1 相比，配置将文本通道改为 80，并把噪声目标换成 velocity：

```text
v_t = sqrt(alpha_bar_t) * epsilon - sqrt(1 - alpha_bar_t) * z_0
L = mean(|| v_theta(z_t, t, c) - v_t ||^2)
```

DDIM sampler 会把 `v` 还原成 `epsilon` 和 `z_0` 后执行确定性更新。

## 运行

```bash
python architecture/sd/SD-v2/model.py
python architecture/sd/SD-v2/train.py --steps 5 --checkpoint sd_v2_tiny.pt
python architecture/sd/SD-v2/inference.py --prompt "a blue diagonal" --steps 8
python architecture/sd/SD-v2/inference.py --checkpoint sd_v2_tiny.pt --output sample.png
```
