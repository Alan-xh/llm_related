# Conditional DiT (Diffusion Transformer) 技术架构与接口文档

## 1. 架构总览

Conditional DiT (Conditional Diffusion Transformer) 抛弃了传统扩散模型中基于 2D 卷积的 U-Net 骨干，改用全 Transformer 架构对带有噪声的图像 Token 进行全局上下文建模。其核心工作流分为：图像 Patch 化、条件特征融合、adaLN-Zero 块内调制以及 Patch 还原。

```
                       [ 噪声图像 x_t ] (B, C, H, W)
                              │
                              ▼
                      [ Patchify & Linear ]
                              │
                              ▼
                      [ Token 序列 x ] ───(+)─── [ 可学习位置编码 pos_embed ]
                        (B, N, dim)         │
                                            ▼
┌───────────────────────────────────────────┴──────────────────────────────────────────┐
│                             DiT Block (重复 Depth 次)                                 │
│                                                                                      │
│   [ 时间步 t ] ──> TimeEmbedding ──┐                                                 │
│                                    ├──(+)──> [ 条件向量 c ]                           │
│   [ 类别 y ]   ──> ClassEmbedding ──┘          (B, dim)                              │
│                                                   │                                  │
│                                                   ▼                                  │
│                                         [ adaLN Modulation ]                         │
│                                                   │ (6 x [B, dim])                   │
│                                                   ▼                                  │
│   x_in ──> LayerNorm ──> (1+scale)*x + shift ──> Attention ──> *gate ──(+)──> x_mid  │
│     │                                                                   ▲            │
│     └───────────────────────────────────────────────────────────────────┘            │
│                                                                                      │
│   x_mid ──> LayerNorm ──> (1+scale)*x + shift ──>   MLP    ──> *gate ──(+)──> x_out  │
│     │                                                                   ▲            │
│     └───────────────────────────────────────────────────────────────────┘            │
└───────────────────────────────────────────┬──────────────────────────────────────────┘
                                            │ (B, N, dim)
                                            ▼
                                  [ LayerNorm & Linear ]
                                            │
                                            ▼
                                   [ Unpatchify 还原 ]
                                            │
                                            ▼
                                 [ 预测噪声 \hat{\epsilon} ] (B, C, H, W)

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

以下为图像尺寸 `[B, 3, 32, 32]`，Patch 尺寸 $p=4$（即 Patch 数 $N=(32/4)^2=64$，Patch 扁平维度 $p^2 \cdot C = 4 \times 4 \times 3 = 48$），隐层维度 `dim=128` 时的完整张量维度演变：

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input (`x_t`)** | `[B, 3, 32, 32]` | - | 带有高斯噪声的输入图像 |
| **Input (`t`, `y`)** | `[B]` | - | 扩散时间步离散索引与类别 Label |
| **Patchify** | `[B, 3, 32, 32]` | `[B, 64, 48]` | 空间分割：`[B, C, (H/p)*p, (W/p)*p]` $\rightarrow$ `[B, N, p*p*C]` |
| **Patch Embedding** | `[B, 64, 48]` | `[B, 64, 128]` | 线性通道投影：将 $48$ 维 Patch 投影至内部 `dim=128` 维度 |
| **Position Add** | `[B, 64, 128]` | `[B, 64, 128]` | 注入可学习位置编码：`x + pos_embed` (`[1, 64, 128]`) |
| **Time Embedding** | `[B]` | `[B, 128]` | 正弦函数映射后再通过 2 层 MLP 得到时间步 Embedding |
| **Class Embedding** | `[B]` | `[B, 128]` | 查表映射 `nn.Embedding(10, 128)` 获得类别向量 |
| **Condition Agg (c)** | `[B, 128]` | `[B, 128]` | 条件特征元素级相加：`c = t_emb + y_emb` |
| **adaLN Chunk** | `[B, 128]` | 6 $\times$ `[B, 128]` | 通过 Linear 层生成 `shift, scale, gate` 参数组 |
| **Self-Attention** | `[B, 64, 128]` | `[B, 64, 128]` | Q,K,V 投影在 4 个 Attention Head 上计算全局自注意力 |
| **DiT Block Output** | `[B, 64, 128]` | `[B, 64, 128]` | 经过调制后自注意力与 MLP 块残差累加 |
| **Proj Out** | `[B, 64, 128]` | `[B, 64, 48]` | LayerNorm 后线性投影还原回 Patch 像素通道数 |
| **Unpatchify** | `[B, 64, 48]` | `[B, 3, 32, 32]` | 张量维度重塑：`[B, (H/p)*(W/p), p*p*C]` $\rightarrow$ `[B, C, H, W]` |

---

## 3. 核心公式与代码映射

| 数学概念 / 论文公式 | 代码变量 / 实现位置 | 物理/工程含义说明 |
| --- | --- | --- |
| **加噪公式**：$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ | `q_sample()` 中的 `sqrt_acp * x0 + sqrt_omc * noise` | 前向扩散过程，根据 Schedule 步数 $t$ 快速封闭求解加噪图像 |
| **正弦位置编码**：$\sin/\cos(t / 10000^{2i/d})$ | `TimeEmbedding.forward()` 中的 `torch.sin(t_emb)`, `torch.cos(t_emb)` | 将离散的时间步 $t$ 编码为连续高维空间表示 |
| **adaLN 调制**：$\gamma \cdot \text{LN}(x) + \beta$ | `DiTBlock.forward()` 中的 `self.norm1(x) * (1 + scale_msa[:, None, :]) + shift_msa[:, None, :]` | 利用时间与类别综合向量 $c$ 对 LayerNorm 输出进行缩放 $\gamma=(1+\text{scale})$ 与平移 $\beta=\text{shift}$ |
| **门控残差控制**：$x + \alpha \cdot f(x)$ | `DiTBlock.forward()` 中的 `x + gate_msa[:, None, :] * self.attn(...)` | 引入可学习门控权重 $\alpha=\text{gate}$，在初始化为 0 时使分支初始接近恒等映射 |
| **扩散损失**：$\mathcal{L}_{\text{simple}} = \Vert{}\epsilon - \hat{\epsilon}_\theta\Vert{}^2$ | `compute_diffusion_loss()` 中的 `F.mse_loss(pred_noise, noise)` | 均方误差损失，直接优化模型预测加入的高斯噪声 |