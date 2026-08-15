
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


2. **输入门与候选状态**：

$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$


$$\tilde{C}_t = \tanh(W_c \cdot [h_{t-1}, x_t] + b_c)$$


3. **更新细胞状态**：

$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$


4. **输出门与更新隐藏状态**：

$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$


$$h_t = o_t \odot \tanh(C_t)$$



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

