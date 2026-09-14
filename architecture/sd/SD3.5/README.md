# Stable Diffusion 3.5

本例复用 SD3 的 MMDiT 数据流，但使用更宽、更深的 tiny 配置，并提供 guidance/distillation
embedding hook，以对应 SD3.5 系列中质量、速度和模型尺寸之间的工程取舍。

## 来源

- 官方模型页：[Stable Diffusion 3.5](https://stability.ai/stable-image)
- 开源实现：[Stability-AI/generative-models](https://github.com/Stability-AI/generative-models)

## 数据流与 Shape

```text
image                  [B, 3, 32, 32]
VAE latent             [B, 4, 4, 4]
image tokens           [B, 16, 4] -> [B, 16, 160]
text encoder           [B, 64, 96] -> [B, 64, 160]
joint attention        concat -> [B, 80, 160]
guidance embedding     [B, 1] -> [B, 128]
image velocity         [B, 4, 4, 4]
```

训练目标仍是 flow matching；`distilled=True` 可以作为少步数变体的配置开关，
推理入口则显式把 guidance scale 传入 denoiser 的 embedding hook。

```text
x_t = (1 - t) * z_0 + t * epsilon
u_t = epsilon - z_0
L = mean(|| u_theta(x_t, t, c) - u_t ||^2)
x_next = x_t + (t_next - t) * u_theta(x_t, t, c)
```

## 运行

```bash
python architecture/sd/SD3.5/model.py
python architecture/sd/SD3.5/train.py --steps 5 --checkpoint sd35_tiny.pt
python architecture/sd/SD3.5/inference.py --steps 6 --guidance-scale 4.5 --output sd35.png
```

这里的 MMDiT、VAE 和文本编码器均为随机初始化教学模块，不兼容官方 SD3.5 checkpoint。
