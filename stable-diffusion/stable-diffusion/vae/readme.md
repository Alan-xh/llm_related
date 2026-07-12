### 概率分布
#### 正态分布 (Normal Distribution)

1.  **`torch.randn(*size)`**
    *   这是最直接的方式，用于生成服从**标准正态分布**（均值为0，标准差为1）的张量。
    *   **示例**：生成一个形状为 (3, 4) 的张量。
        ```python
        import torch
        x = torch.randn(3, 4)
        ```

2.  **`torch.normal(mean, std)`**
    *   当需要指定**不同的均值或标准差**时使用。它非常灵活，可以接受标量或张量作为参数。
    *   **示例**：生成10个元素，每个元素来自均值不同、标准差固定的正态分布。
        ```python
        # mean 是 [1, 2, 3, ..., 10] 的向量，std 是标量 1.0
        x = torch.normal(mean=torch.arange(1., 11.), std=1.0)
        ```

3.  **`torch.nn.init.normal_(tensor, mean=0.0, std=1.0)`**
    *   这是一个**原地操作**（函数名后的 `_` 表示原地修改），用于将已有的张量 `tensor` 的值用正态分布的随机数填充。常用于初始化神经网络的权重。

#### 均匀分布 (Uniform Distribution)

1.  **`torch.rand(*size)`**
    *   这是最常用的方式，生成服从 **[0, 1) 区间内的均匀分布**的张量。
    *   **示例**：生成一个形状为 (3, 4) 的张量。
        ```python
        x = torch.rand(3, 4)
        ```

2.  **`torch.nn.init.uniform_(tensor, a=0.0, b=1.0)`**
    *   这是一个**原地操作**，将已有张量的值用区间 `[a, b)` 上的均匀分布随机数填充。这也是一个常用的权重初始化方法。

#### 补充：如果追求更精细的控制

PyTorch 还提供了 `torch.distributions` 包。如果你需要进行更复杂的概率操作（如计算对数概率密度、实现可微采样等），这个包会更合适。

*   **正态分布**：`from torch.distributions import Normal; dist = Normal(loc=0.0, scale=1.0)`
*   **均匀分布**：`from torch.distributions import Uniform; dist = Uniform(low=0.0, high=1.0)`

> `dist` 对象提供了 `sample()`, `rsample()`, `log_prob()` 等方法。