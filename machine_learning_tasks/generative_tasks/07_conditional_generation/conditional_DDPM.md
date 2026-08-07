# Class-conditional DDPM 技术架构与接口文档

## 1. 架构总览

条件去噪扩散概率模型 (Class-conditional DDPM) 旨在通过外部条件（如类别标签 $y$）控制图像生成的类别分布。该架构主要分为三大部分：**前向加噪过程 (Forward Process)**、**条件控制嵌入 (Condition Embedding)** 和 **反向去噪 U-Net 网络 (Reverse U-Net)**。

```
                       [ 类别标签 y ] -> ClassEmbedding --\
                                                          + (元素相加) -> [ 条件嵌入 cond ]
                       [ 时间步长 t ] -> TimeEmbedding --/                     |
                                                                               v (空间广播注入)
[ 真实图像 x0 ] -> q_sample(x0, t) -> [ 含噪图像 xt ] ----> [ Down1 -> Down2 ] -+-> [ Bottleneck Mid ] -> [ Up2 -> Up1 ] -> [ 预测噪声 ε_theta ]

```

* **前向过程 (`q_sample`)**：根据马尔可夫链性质，直接利用 closed-form 扩展公式计算 $t$ 时刻的含噪图像 $x_t$。
* **条件融合机制**：正弦时间位置编码与类别嵌入做向量加和（`t_emb + c_emb`），将标量/离散控制量统一转换为连续向量 $cond \in \mathbb{R}^{128}$。
* **特征注入**：在 U-Net 瓶颈之前将条件向量进行空间广播，并加和至特征图（`h2 + cond[:, :, None, None]`），使网络在去噪推断中显式感知条件约束。

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

以下为输入样本批次大小为 $B$，通道数 $C=3$，分辨率 $H=W=32$，条件维度 $D=128$ 时，模型内部张量维度演变全流程：

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Input x0** | `[B, 3, 32, 32]` | - | 原始无噪图像输入 |
| **Input t / y** | `[B]` / `[B]` | - | 时间步长标量索引与离散类别标签 |
| **TimeEmbedding** | `[B]` | `[B, 128]` | 正弦位置编码 + Linear 映射至连续向量空间 |
| **ClassEmbedding** | `[B]` | `[B, 128]` | 离散类别索引转换（nn.Embedding 查表） |
| **Condition Fusion** | `[B, 128]`, `[B, 128]` | `[B, 128]` | 向量点对点相加：`t_emb + c_emb` |
| **q_sample (Forward)** | `[B, 3, 32, 32]` | `[B, 3, 32, 32]` | 前向加噪公式：$x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ |
| **Encoder Stage 1 (Down1)** | `[B, 3, 32, 32]` | `[B, 64, 32, 32]` | 卷积升维，保持空间分辨率 |
| **Encoder Stage 2 (Down2)** | `[B, 64, 32, 32]` | `[B, 128, 16, 16]` | 步长 `stride=2` 卷积下采样，分辨率减半 |
| **Cond Injection** | `[B, 128, 16, 16]` | `[B, 128, 16, 16]` | `cond` 扩展为 `[B, 128, 1, 1]` 广播叠加至特征图 |
| **Bottleneck (Mid)** | `[B, 128, 16, 16]` | `[B, 128, 16, 16]` | 瓶颈层特征提取与表征拟合 |
| **Decoder Stage 2 (Up2)** | `[B, 128, 16, 16]` | `[B, 64, 32, 32]` | 转置卷积上采样，分辨率翻倍 |
| **Decoder Stage 1 (Up1)** | `[B, 64, 32, 32]` | `[B, 3, 32, 32]` | 卷积将通道数还原至原始图像通道数 $C=3$ |

---

## 3. 核心公式与代码映射

| 数学推导公式 | 代码变量 / 算子实现 | 物理/工程含义 |
| --- | --- | --- |
| $PE_{(t, 2i)} = \sin\left(\frac{t}{10000^{2i/d}}\right)$ | `torch.sin(emb)` | 时间步 $t$ 连续位置正弦编码 |
| $PE_{(t, 2i+1)} = \cos\left(\frac{t}{10000^{2i/d}}\right)$ | `torch.cos(emb)` | 时间步 $t$ 连续位置余弦编码 |
| $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ | `sqrt_acp * x0 + sqrt_omc * noise` | 单步闭式加噪计算当前含噪图像 $x_t$ |
| $h_{cond} = h + (E_t(t) + E_c(y))$ | `h2 + cond[:, :, None, None]` | 将时间与类别联合条件注入网络特征图 |
| $\mathcal{L}_{\text{simple}}(\theta) = \Vert{}\epsilon - \epsilon_\theta(x_t, t, y)\Vert{}^2$ | `F.mse_loss(pred_noise, noise)` | 预测高斯噪声与真实噪声的 MSE 损失 |