# SimCLR 对比学习 Pipeline 技术架构与接口文档

## 1. 架构总览

SimCLR (A Simple Framework for Contrastive Learning of Visual Representations) 是一种无监督判别式对比学习架构。其核心理念是通过数据增强构建“正样本对”，并通过拉近同源正样本对的距离、拉远异源负样本对的距离来学习无监督表征。

### 数据流与系统拓扑结构 (ASCII Architecture)

```
                       +-------------------+
                       | Raw Image Batch   |  x: [B, 3, 64, 64]
                       +---------+---------+
                                 |
              +------------------+------------------+
              |                                     |
              v (Augmentation t)                    v (Augmentation t')
     +-----------------+                   +-----------------+
     | View 1 (v1)     | [B, 3, 64, 64]    | View 2 (v2)     | [B, 3, 64, 64]
     +--------+--------+                   +--------+--------+
              |                                     |
              +------------------+------------------+
                                 | Concatenation (Dim 0)
                                 v
                       +-------------------+
                       | Batch Concat      |  x_concat: [2*B, 3, 64, 64]
                       +---------+---------+
                                 |
                                 v
                       +-------------------+
                       | ResNet18 Backbone |  f(·) Encoder
                       +---------+---------+
                                 |
                                 v
                       +-------------------+
                       | Feature h         |  h: [2*B, 512]
                       +---------+---------+
                                 |
                                 v
                       +-------------------+
                       | MLP Projector g(·)|  2-Layer MLP
                       +---------+---------+
                                 |
                                 v
                       +-------------------+
                       | Projection Vector |  z: [2*B, 128]
                       +---------+---------+
                                 |
                                 v
                       +-------------------+
                       | NT-Xent Loss      |  Cosine Sim & Softmax CrossEntropy
                       +-------------------+

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

下表追踪了一个标准 Mini-Batch (假设 $B=32$, $C=3$, $H=64$, $W=64$) 在整个模型管道中的 Shape 演变全过程：

| 节点/模块 (Module Node) | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Raw Input** ($x$) | - | `[32, 3, 64, 64]` | 输入合成图像 Batch |
| **Augmentation** ($v_1, v_2$) | `[32, 3, 64, 64]` | `[32, 3, 64, 64]` × 2 | 生成两个独立的随机增强视图 |
| **Batch Concat** ($x_{concat}$) | `[32, 3, 64, 64]` × 2 | `[64, 3, 64, 64]` | 沿 Batch 维度拼接，$2N = 64$ |
| **Conv1 + MaxPool** | `[64, 3, 64, 64]` | `[64, 64, 16, 16]` | $7 \times 7$ 卷积 (s=2) + MaxPool (s=2) 下采样 4 倍 |
| **ResNet Stage 1** | `[64, 64, 16, 16]` | `[64, 64, 16, 16]` | 残差块提取特征，分辨率保持 |
| **ResNet Stage 2** | `[64, 64, 16, 16]` | `[64, 128, 8, 8]` | 步长 2 卷积下采样，通道加倍 |
| **ResNet Stage 3** | `[64, 128, 8, 8]` | `[64, 256, 4, 4]` | 步长 2 卷积下采样，通道加倍 |
| **ResNet Stage 4** | `[64, 256, 4, 4]` | `[64, 512, 2, 2]` | 步长 2 卷积下采样，通道加倍 |
| **Adaptive AvgPool** | `[64, 512, 2, 2]` | `[64, 512, 1, 1]` | 全局自适应均值池化压缩空间维度 |
| **Flatten (Backbone Output $h$)** | `[64, 512, 1, 1]` | `[64, 512]` | 展平为一维表征向量 $h$ |
| **MLP Projector** | `[64, 512]` | `[64, 128]` | 经过线性映射与激活变换输出 $z$ |
| **L2 Normalization** | `[64, 128]` | `[64, 128]` | $z_{norm} = z / \Vert{}z\Vert{}_2$，使其分布在单位超球面上 |
| **Similarity Matrix ($S$)** | `[64, 128]` | `[64, 64]` | 全对全点积计算余弦相似度并除以温度参数 $\tau$ |
| **NT-Xent Loss Output** | `[64, 64]` | `[]` (Scalar) | 与 Target 匹配计算 CrossEntropy 标量 Loss |

---

## 3. 核心公式与代码映射

| 数学定义 / 论文公式 | 代码实现变量 / 算子 | 详细物理/几何含义 |
| --- | --- | --- |
| **Cosine Similarity**<br>

<br>$\text{sim}(u, v) = \frac{u^T v}{\Vert{}u\Vert{}_2 \Vert{}v\Vert{}_2}$ | `F.normalize(z, p=2, dim=1)`<br>

<br>`torch.matmul(z_norm, z_norm.t())` | 特征归一化后计算矩阵乘法，获取特征空间内夹角的余弦值 |
| **Temperature Scaling**<br>

<br>$\frac{\text{sim}(z_i, z_j)}{\tau}$ | `/ self.temperature` | 温度系数 $\tau$ 控制对困难负样本的惩罚粒度，极小化分布的熵 |
| **Mask Diagonal**<br>

<br>$\text{mask}_{i=j} = -\infty$ | `sim_matrix.masked_fill(mask_self, -1e9)` | 屏蔽样本与自身的对比，防止模型学习平凡解 (Trivial Solution) |
| **NT-Xent Loss**<br>

<br>$\ell_{i,j} = -\log \frac{\exp(\text{sim}_{i,j}/\tau)}{\sum_k \exp(\text{sim}_{i,k}/\tau)}$ | `F.cross_entropy(sim_matrix, pos_targets)` | 利用 CrossEntropy 将对比学习任务转换为一个 $2N-1$ 分类的预测问题 |