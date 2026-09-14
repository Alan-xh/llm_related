# Stable Diffusion XL

SDXL 教学实现把双文本编码器、pooled text embedding、time ids，以及 base/refiner 两阶段
推理压缩到一个小型 latent U-Net 中。

## 来源

- 论文：[SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952)
- 开源实现：[Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)

## 数据流与 Shape

```text
image                     [B, 3, 32, 32]
VAE latent                [B, 4, 4, 4]
text encoder 1            [B, 64, 64]
text encoder 2            [B, 64, 80]
concat token context      [B, 64, 144]
concat pooled embeddings  [B, 144]
time ids                  [B, 6]
added conditioning        [B, 144 + 6] -> [B, 128]
base/refiner U-Net        [B, 4, 4, 4] -> [B, 4, 4, 4]
```

两个文本编码器的 token features 在 cross-attention 前拼接；pooled embedding 和 6 个 size/
crop/target-size 数值经过 MLP 后加入 timestep embedding。推理先运行 base scheduler，再用
独立的 refiner scheduler 细化 latent。

## 核心公式

```text
z_t = sqrt(alpha_bar_t) * z_0 + sqrt(1 - alpha_bar_t) * epsilon
h_t = MLP(timestep_t) + MLP([pooled_text, time_ids])
guided = u + s * (c - u)
```

## 运行

```bash
python architecture/sd/SDXL/model.py
python architecture/sd/SDXL/train.py --steps 5 --checkpoint sdxl_tiny.pt
python architecture/sd/SDXL/inference.py --steps 6 --refiner-steps 2 --output sdxl.png
python architecture/sd/SDXL/inference.py --checkpoint sdxl_tiny.pt --prompt "a green circle"
```

该实现的 base 和 refiner 都是随机初始化的教学网络；两阶段接口用于展示结构，不等同于官方
base/refiner checkpoint。
