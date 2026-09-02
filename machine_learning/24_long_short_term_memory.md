# 长短期记忆网络 (Long Short-Term Memory, LSTM)

## 1. 算法原理

传统的循环神经网络（RNN）在处理长序列时容易遇到梯度消失和梯度爆炸问题。长短期记忆网络（LSTM）通过引入**门控机制（Gating Mechanism）** 和 **细胞状态（Cell State）**，极大地缓解了长距离依赖问题。

LSTM 包含三个关键门结构：

1. **遗忘门 (Forget Gate)**：决定从细胞状态中丢弃多少旧信息。
2. **输入门 (Input Gate)**：决定将多少新信息存入细胞状态。
3. **输出门 (Output Gate)**：决定隐藏状态输出多少细胞状态中的信息。

---

## 2. 数学公式与推导

对于时间步 $t$，当前输入为 $x_t$，上一时刻隐状态为 $h_{t-1}$，上一时刻细胞状态为 $C_{t-1}$：

### 2.1 门控机制计算

1. **遗忘门**：

$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

* $f_t$: 遗忘门的激活值（取值范围在 $0$ 到 $1$ 之间）
* $\sigma$(西格玛): Sigmoid 激活函数
* $W_f$: 遗忘门的权重矩阵
* $h_{t-1}$: 上一时刻（时间步 $t-1$）的隐藏状态向量
* $x_t$: 当前时刻（时间步 $t$）的输入向量
* $[h_{t-1}, x_t]$: 隐藏状态向量与输入向量的拼接（Concatenation）
* $b_f$: 遗忘门的偏置项

2. **输入门与候选状态**：

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$

* $i_t$: 输入门的激活值（控制更新信息的比例）
* $\sigma$(西格玛): Sigmoid 激活函数
* $W_i$: 输入门的权重矩阵
* $h_{t-1}$: 上一时刻的隐藏状态向量
* $x_t$: 当前时刻的输入向量
* $[h_{t-1}, x_t]$: 隐藏状态向量与输入向量的拼接
* $b_i$: 输入门的偏置项

$$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$

* $\tilde{C}_t$: 当前时刻的候选细胞状态（即拟加入的候选新信息）
* $\tanh$: 双曲正切激活函数（将数值压缩至 $[-1, 1]$ 范围）
* $W_c$: 候选细胞状态更新的权重矩阵
* $h_{t-1}$: 上一时刻的隐藏状态向量
* $x_t$: 当前时刻的输入向量
* $[h_{t-1}, x_t]$: 隐藏状态向量与输入向量的拼接
* $b_c$: 候选细胞状态更新的偏置项

3. **更新细胞状态**：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

* $C_t$: 当前时刻（时间步 $t$）更新后的细胞状态向量
* $f_t$: 遗忘门输出向量
* $\odot$: Hadamard 积（按元素相乘，Element-wise Product）
* $C_{t-1}$: 上一时刻（时间步 $t-1$）的细胞状态向量
* $i_t$: 输入门输出向量
* $\tilde{C}_t$: 当前时刻的候选细胞状态向量

4. **输出门与更新隐藏状态**：

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$

* $o_t$: 输出门的激活值（控制输出多少细胞状态信息）
* $\sigma$(西格玛): Sigmoid 激活函数
* $W_o$: 输出门的权重矩阵
* $h_{t-1}$: 上一时刻的隐藏状态向量
* $x_t$: 当前时刻的输入向量
* $[h_{t-1}, x_t]$: 隐藏状态向量与输入向量的拼接
* $b_o$: 输出门的偏置项

$$h_t = o_t \odot \tanh(C_t)$$

* $h_t$: 当前时刻（时间步 $t$）更新后的隐藏状态向量（最终的输出状态）
* $o_t$: 输出门输出向量
* $\odot$: Hadamard 积（按元素相乘）
* $\tanh$: 双曲正切激活函数
* $C_t$: 当前时刻更新后的细胞状态向量

其中 $\sigma(\cdot)$ 为 Sigmoid 函数，$\odot$ 表示 Hadamard 积。

---

## 3. ASCII 结构图

```
                  C_{t-1} ------------[x]----------------(+)----------> C_t
                                       |                  |
                                       | f_t              | i_t * C_tilde_t
                                +------+------+    +------+------+
                                |  Forget Gate|    |  Input Gate |
                                +------+------+    +------+------+
                                       |                  |
   h_{t-1} ----+-----------------------+--------+---------+------> [tanh] ---[x] ---> h_t
               |                                |                             |
               +---[ W_f ]                      +---[ W_i, W_c ]              | o_t
               |                                |                             |
   x_t --------+--------------------------------+--------------------[ W_o ]--+


```

---

## 4. Python 代码实现 (基于 NumPy)

```python
import numpy as np

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class LSTMCell:
    def __init__(self, input_dim, hidden_dim):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        concat_dim = input_dim + hidden_dim
        self.Wf = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.bf = np.zeros((hidden_dim, 1))
        
        self.Wi = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.bi = np.zeros((hidden_dim, 1))
        
        self.Wc = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.bc = np.zeros((hidden_dim, 1))
        
        self.Wo = np.random.randn(hidden_dim, concat_dim) * 0.01
        self.bo = np.zeros((hidden_dim, 1))

    def forward(self, x_t, h_prev, C_prev):
        concat = np.vstack((h_prev, x_t))
        
        f_t = sigmoid(np.dot(self.Wf, concat) + self.bf)
        i_t = sigmoid(np.dot(self.Wi, concat) + self.bi)
        C_tilde = np.tanh(np.dot(self.Wc, concat) + self.bc)
        
        C_t = f_t * C_prev + i_t * C_tilde
        
        o_t = sigmoid(np.dot(self.Wo, concat) + self.bo)
        h_t = o_t * np.tanh(C_t)
        
        return h_t, C_t

if __name__ == "__main__":
    cell = LSTMCell(input_dim=10, hidden_dim=20)
    x = np.random.randn(10, 1)
    h_prev = np.zeros((20, 1))
    C_prev = np.zeros((20, 1))
    
    h_next, C_next = cell.forward(x, h_prev, C_prev)
    print("新的隐状态 h_t 维度:", h_next.shape)
    print("新的细胞状态 C_t 维度:", C_next.shape)


```