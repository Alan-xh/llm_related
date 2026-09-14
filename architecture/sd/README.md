# Stable Diffusion 系列架构

本目录按 Stable Diffusion 的主要版本拆分，重点记录 VAE、文本编码器、U-Net/DiT、scheduler 和 classifier-free guidance。
每个版本都是可在 CPU 上运行的极小 PyTorch 教学实现：使用合成图像、字节级 tokenizer 和随机初始化权重，
用于观察数据流与关键模块，不兼容 Stability AI 官方 checkpoint，也不是官方训练配方的复刻。

公共组件位于 [`common.py`](./common.py)，包括 8 倍下采样 VAE、文本编码器、交叉注意力、DDIM、
v-prediction 和 flow matching Euler solver。每个版本目录仍保留独立的 `model.py`、`train.py` 和
`inference.py`，方便单独阅读与运行。

| 目录 | 主要方向 | 主要创新技术 |
| --- | --- | --- |
| [`SD-v1/`](./SD-v1/) | latent diffusion + U-Net | VAE 潜空间扩散 + CLIP cross-attention 条件控制 |
| [`SD-v2/`](./SD-v2/) | 文本编码器与分辨率改进 | OpenCLIP、v-prediction、深度条件和更高分辨率 |
| [`SDXL/`](./SDXL/) | 双文本编码器与高分辨率生成 | 双文本编码器、更大 U-Net、base/refiner 两阶段 |
| [`SD3/`](./SD3/) | 多模态扩散 Transformer | MMDiT 分离文本/图像流 + flow matching |
| [`SD3.5/`](./SD3.5/) | SD3.5 系列改进 | 改进 MMDiT 与训练数据，并提供多规模/Turbo 变体 |

## 统一数据流

```text
image [B, 3, H, W]
  -> TinyVAE encoder                         [B, 4, H/8, W/8]
  -> add noise / flow interpolation          [B, 4, H/8, W/8]
prompt
  -> byte tokenizer -> text encoder          [B, T, C]
  -> U-Net cross-attention or MMDiT joint attention
  -> epsilon/v/velocity prediction            [B, 4, H/8, W/8]
  -> DDIM or Euler flow solver
  -> TinyVAE decoder                          [B, 3, H, W]
```

训练入口只优化 denoiser，VAE 和文本编码器冻结。默认 `H=W=32`、`T=64`，
因此默认 latent shape 是 `[B, 4, 4, 4]`。推理时同时计算空 prompt 和条件 prompt，
使用 classifier-free guidance：

```text
u = f(z_t, empty_prompt)
c = f(z_t, prompt)
guided = u + s * (c - u)
```

## 快速运行

```bash
python architecture/sd/SD-v1/model.py
python architecture/sd/SD-v1/train.py --steps 2 --checkpoint sd_v1_tiny.pt
python architecture/sd/SD-v1/inference.py --checkpoint sd_v1_tiny.pt --output sd_v1.png

python architecture/sd/SD-v2/inference.py --steps 4 --output sd_v2.png
python architecture/sd/SDXL/inference.py --steps 4 --refiner-steps 1 --output sdxl.png
python architecture/sd/SD3/inference.py --steps 4 --output sd3.png
python architecture/sd/SD3.5/inference.py --steps 4 --output sd35.png
```

如果希望从包路径导入公共组件，可以使用 `from architecture.sd.common import TinyVAE`。
