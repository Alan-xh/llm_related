# Deformable-DETR

## 论文与官方代码

- 论文：[Deformable DETR: Deformable Transformers for End-to-End Object Detection](https://arxiv.org/abs/2010.04159)
- 官方代码：[fundamentalvision/Deformable-DETR](https://github.com/fundamentalvision/Deformable-DETR)
- 论文要点：在多尺度特征图上围绕参考点采样少量位置，替代对全部空间位置做全局注意力，从而降低计算量并改善 DETR 的收敛速度。

## 数据流

```text
图像 [B, 3, H, W]
  -> CNN backbone + multi-scale feature maps
  -> 各层 1x1 conv 统一到 d 维
  -> flatten + level embedding + positional encoding
  -> multi-scale deformable encoder
  -> object queries / two-stage proposals
  -> deformable decoder
  -> 分类头 / 框回归头
  -> Hungarian matching + 分类、L1、GIoU 损失
```

典型 ResNet-50 配置使用 stride `8, 16, 32, 64` 的四层特征。每个 query 带有参考点；decoder 每层根据参考点预测少量采样偏移，并可迭代更新参考框。

## Tensor Shape

设四个尺度为 `l=1..4`，统一通道 `d=256`，每个注意力头采样 `K_s=4` 个点：

| 张量 | Shape | 说明 |
| --- | --- | --- |
| 第 `l` 层特征 | `[B, 256, H_l, W_l]` | stride 可为 8/16/32/64 |
| 第 `l` 层展平特征 | `[B, H_lW_l, 256]` | 保留 level 顺序 |
| 多尺度 encoder 输入 | `[B, S, 256]` | `S=sum_l H_lW_l` |
| reference points | `[B, L_q, 2]` 或 `[B, L_q, 4]` | 点或框，坐标归一化 |
| sampling offsets | `[B, L_q, heads, 4, K_s, 2]` | 每层、每头、每点偏移 |
| attention weights | `[B, L_q, heads, 4, K_s]` | 采样点权重 |
| decoder 输出 | `[B, N, 256]` | `N` 为 query 数量 |
| 分类 logits | `[B, N, K+1]` | 含 no-object |
| 边界框 | `[B, N, 4]` | 归一化 `cxcywh` |

这里的 `L_q` 是 query 数量，`S` 是所有尺度空间位置总数；`heads` 是多头注意力头数。

## 核心公式

多尺度可变形注意力对 query `q` 的输出为：

```text
MSDeformAttn(q, p_q, x) =
  sum_{l=1}^4 sum_{k=1}^{K_s}
    A_{q,l,k} W x_l(phi_l(p_q) + Delta p_{q,l,k})
```

其中 `p_q` 是参考点，`Delta p` 是预测偏移，`A` 是归一化采样权重，`phi_l` 将归一化坐标映射到第 `l` 层特征图；实际实现使用双线性插值。

匹配与检测损失仍采用集合预测：

```text
sigma* = argmin_sigma sum_i [
  -log p_hat_{sigma(i)}(c_i)
  + lambda_L1 ||b_i - b_hat_{sigma(i)}||_1
  + lambda_giou (1 - GIoU(b_i, b_hat_{sigma(i)}))
]
L = L_cls + lambda_L1 L_box + lambda_giou L_giou
```

## 教学实现边界

- 实现核心路径：CNN 多尺度特征、level embedding、reference points、稀疏采样的 multi-scale deformable attention、decoder、Hungarian matching 和检测损失。
- 可先用 `grid_sample` 表达双线性采样，明确展示“少量点 × 多尺度”的数据流；生产级 CUDA kernel、MSDeformAttn 优化和完整 checkpoint 兼容不在范围内。
- 默认实现四尺度、固定 query 和 COCO 风格标注；可选实现 iterative box refinement，two-stage proposal 仅作为扩展。
- 不复刻官方全部训练 recipe、分布式策略、混合精度、COCO 工具链和速度指标；Shape 以单卡、batch-first 表达。

## 运行命令

```bash
python architecture/detr/Deformable-DETR/model.py
python architecture/detr/Deformable-DETR/train.py --steps 2 --checkpoint deformable_detr_tiny.pt
python architecture/detr/Deformable-DETR/inference.py --checkpoint deformable_detr_tiny.pt
```

本仓库用 `grid_sample` 表达稀疏双线性采样。官方仓库的编译与训练入口则是：

```bash
cd /path/to/Deformable-DETR
cd models/ops && sh make.sh
GPUS_PER_NODE=8 ./tools/run_dist_launch.sh 8 ./configs/r50_deformable_detr.sh
```
