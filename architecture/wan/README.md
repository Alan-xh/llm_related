# Wan 系列架构

本目录按 Wan 视频生成模型的主要版本拆分，重点记录视频 VAE、时空 patch、扩散 Transformer、文本条件和视频采样流程。

| 目录 | 主要方向 | 主要创新技术 |
| --- | --- | --- |
| [`Wan2.1/`](./Wan2.1/) | 文生视频、图生视频与视频 VAE | 3D causal VAE + Flow Matching DiT + T5 文本条件 |
| [`Wan2.2/`](./Wan2.2/) | MoE 扩散 Transformer 与高压缩视频生成 | 高噪声/低噪声专家 MoE + 高压缩 VAE + TI2V |
