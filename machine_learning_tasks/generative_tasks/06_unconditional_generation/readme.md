## 去噪扩散概率模型(DDPM)

在每一轮训练过程中，包含以下内容

1. 每一个训练样本选择一个随机步长 t
2. 将 time step t 对应的高斯噪音应用到图片中
3. 将 time step 转化为对应 embedding

DDPM的前向过程是一个**马尔可夫链**，它逐步向原始数据 $x_0$ 中添加高斯噪声，经过 $T$ 步后，数据分布趋近于标准正态分布。

---

### 1. 单步转移公式

从第 $t-1$ 步到第 $t$ 步，噪声添加过程定义为：

$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t \mathbf{I})$

其中：
- $\beta_t \in (0, 1)$ 是预先定义的方差调度(常量)（variance schedule），通常很小（如从 $10^{-4}$ 到 $0.02$）。
- 均值为 $\sqrt{1 - \beta_t} \, x_{t-1}$
- 方差为 $\beta_t \mathbf{I}$

---

### 2. 从 $x_0$ 到 $x_t$ 的闭合形式（重参数化）

利用高斯分布的可加性，我们可以直接一步得到 $x_t$ 的分布，而不需要逐步迭代。

定义：
$\alpha_t = 1 - \beta_t, \quad \bar{\alpha}_t = \prod_{i=1}^{t} \alpha_i$

则有：
$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} \, x_0, (1 - \bar{\alpha}_t) \mathbf{I})$

使用重参数化技巧（采样形式）：
$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$

这个公式非常关键，因为它允许我们在训练时**直接从 $x_0$ 采样任意时刻的 $x_t$**，效率极高。

---

### 3. 后验条件分布（用于反向过程训练）

在反向过程中，我们需要已知 $x_0$ 时 $x_{t-1}$ 的条件分布，其表达式为：

$q(x_{t-1} | x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t \mathbf{I})$

其中：
$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0$

$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$

这个后验分布用于推导训练目标时的变分下界（VLB）。

---

### 4. 最终性质（当 $T \to \infty$）

如果噪声调度设计得当，当 $T$ 足够大时：
$q(x_T | x_0) \approx \mathcal{N}(0, \mathbf{I})$
即原始数据的信息被完全破坏，变成纯高斯噪声。

---

### 总结关键公式表

| 描述 | 公式 |
|------|------|
| 单步转移 | $q(x_t \vert x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t \mathbf{I})$ |
| 任意步边缘分布 | $q(x_t \vert x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t}x_0, (1-\bar{\alpha}_t)\mathbf{I})$ |
| 采样形式 | $x_t = \sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon$ |
| 反向条件后验均值 | $\tilde{\mu}_t = \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t + \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0$ |
| 反向条件后验方差 | $\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t$ |

---

### 核心公式

**采样公式：**
$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$

**其中噪声分布：**
$\epsilon \sim \mathcal{N}(0, \mathbf{I})$
（注意：$x_t$ 一般是向量。如果 $x_t$ 是标量形式，那 $\mathcal{N}(0,1)$ 也是对的。）

**参数定义：**
$\alpha_t = 1 - \beta_t$
$\bar{\alpha}_t = \prod_{i=1}^t \alpha_i = \alpha_1 \alpha_2 \cdots \alpha_t$

---

### 公式含义

- 这个公式允许你**从原始图像 $x_0$ 直接采样出任意时刻 $t$ 的加噪图像 $x_t$**，而不需要从 $x_0 \to x_1 \to \cdots \to x_t$ 逐步迭代。
- 它是通过重参数化技巧（reparameterization trick）得到的，利用了高斯噪声的可加性。

---

### 完整写法（含分布形式）

如果你需要写成**条件分布**的形式，它等价于：

$q(x_t \mid x_0) = \mathcal{N}\big(x_t;\, \sqrt{\bar{\alpha}_t} \, x_0,\; (1 - \bar{\alpha}_t) \mathbf{I}\big)$

采样时就是：
$x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$

---

### 完全对齐的整理

| 项目 | 公式 |
|------|------|
| 前向采样 | $x_t = \sqrt{\bar{\alpha}_t} \, x_0 + \sqrt{1 - \bar{\alpha}_t} \, \epsilon$ |
| 噪声分布 | $\epsilon \sim \mathcal{N}(0, \mathbf{I})$ |
| 单步衰减系数 | $\alpha_t = 1 - \beta_t$ |
| 累积衰减系数 | $\bar{\alpha}_t = \prod_{i=1}^t \alpha_i$ |

