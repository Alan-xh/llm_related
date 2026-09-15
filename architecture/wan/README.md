# Wan 系列架构

本目录按 Wan 视频生成模型的主要版本拆分，重点记录视频 VAE、时空 patch、扩散 Transformer、文本条件和视频采样流程。

| 目录 | 主要方向 | 主要创新技术 |
| --- | --- | --- |
| [`Wan2.1/`](./Wan2.1/) | 文生视频、图生视频与视频 VAE | 3D causal VAE + Flow Matching DiT + T5 文本条件 |
| [`Wan2.2/`](./Wan2.2/) | MoE 扩散 Transformer 与高压缩视频生成 | 高噪声/低噪声专家 MoE + 高压缩 VAE + TI2V |

两个版本目录都包含 `model.py`、`train.py`、`inference.py` 和版本 README。
实现面向架构教学与 CPU smoke test，不兼容官方 checkpoint；推理脚本默认保存
`[B, 3, T, H, W]` 的 PyTorch 视频 tensor。

```bash
python architecture/wan/Wan2.1/model.py
python architecture/wan/Wan2.1/inference.py --mode t2v --steps 4
python architecture/wan/Wan2.2/model.py
python architecture/wan/Wan2.2/inference.py --steps 4
```
