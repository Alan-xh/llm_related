# Stable Diffusion v1

本例实现 latent diffusion 的最小闭环：VAE 潜空间、CLIP-like 文本上下文、带 cross-attention
的 U-Net、epsilon prediction、DDIM 和 classifier-free guidance。

## 来源

- 论文：[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- 开源实现：[CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)

## 数据流与 Shape

```text
image                    [B, 3, 32, 32]
TinyVAE encode           [B, 4, 4, 4]
byte tokenizer           [B, 64]
TinyTextEncoder          [B, 64, 64]
U-Net input              [B, 4, 4, 4]
cross-attention context  [B, 64, 64]
U-Net output             [B, 4, 4, 4]
TinyVAE decode            [B, 3, 32, 32]
```

U-Net 的下采样路径为 `4 -> 32 -> 64` channels，注意力在高、低两个分辨率上注入文本；
上采样路径与 skip connection 拼接后恢复 latent shape。

## 核心公式

令 `alpha_bar_t` 为 scheduler 的累计 alpha，训练前向过程为：

```text
z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon
L = mean(|| epsilon_theta(z_t, t, c) - epsilon ||^2)
guided = epsilon_uncond + s * (epsilon_cond - epsilon_uncond)
```

## 运行

```bash
python architecture/sd/SD-v1/model.py
python architecture/sd/SD-v1/train.py --steps 5 --checkpoint sd_v1_tiny.pt
python architecture/sd/SD-v1/inference.py --prompt "a red square" --steps 8
python architecture/sd/SD-v1/inference.py --checkpoint sd_v1_tiny.pt --output sample.png
```

代码使用随机初始化的 tiny VAE 和文本编码器，输出用于检查流程，不代表官方模型的生成质量。
