
# 循环神经网络 (Recurrent Neural Network, RNN)

## 1. 算法原理

循环神经网络（RNN）专门用于处理**序列数据（Sequence Data）**，如自然语言、时间序列等。传统神经网络假设输入之间相互独立，而 RNN 引入了**隐藏状态（Hidden State）** 的概念，隐藏状态充当记忆功能，将先前的输入信息传递到后续步骤。

在时间步 $t$，RNN 结合当前输入 $x_t$ 和上一步的隐藏状态 $h_{t-1}$，计算当前的隐藏状态 $h_t$ 和输出 $y_t$。由于参数跨时间步共享（Shared Parameters），模型能够处理任意长度的序列。

---

## 2. 数学公式与推导

### 2.1 前向传播 (Forward Pass)

对于时间步 $t$：

1. **隐藏状态更新**：

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$



其中：
* $x_t \in \mathbb{R}^{d}$ 为 $t$ 时刻的输入
* $h_t \in \mathbb{R}^{m}$ 为 $t$ 时刻的隐藏状态
* $W_{hh} \in \mathbb{R}^{m \times m}$ 为隐状态到隐状态的转移权重矩阵
* $W_{xh} \in \mathbb{R}^{m \times d}$ 为输入到隐状态的权重矩阵
* $b_h \in \mathbb{R}^{m}$ 为偏置向量


2. **输出层计算**：

$$\hat{y}_t = \text{softmax}(W_{hy} h_t + b_y)$$



### 2.2 沿时间反向传播 (BPTT - Backpropagation Through Time)

总损失 $L$ 是所有时间步损失的累加：


$$L = \sum_{t=1}^T L_t$$

以损失 $L$ 对隐藏到隐藏权重 $W_{hh}$ 的导数为例：


$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^T \frac{\partial L_t}{\partial W_{hh}}$$

对单个时间步 $t$，因为 $h_t$ 依赖于 $h_{t-1}$，链式法则展开为：


$$\frac{\partial L_t}{\partial W_{hh}} = \sum_{k=1}^t \frac{\partial L_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}$$

---

## 3. ASCII 结构图

```
展开视角 (Unrolled Network):

       y_1               y_2               y_T
        ^                 ^                 ^
        | W_hy            | W_hy            | W_hy
     +-----+   W_hh    +-----+   W_hh    +-----+
h0 ->| h_1 | --------->| h_2 | --------->| h_T |
     +-----+           +-----+           +-----+
        ^                 ^                 ^
        | W_xh            | W_xh            | W_xh
       x_1               x_2               x_T

```

---

## 4. Python 代码实现 (基于 NumPy)

```python
import numpy as np

class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.hidden_size = hidden_size
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
        self.Why = np.random.randn(output_size, hidden_size) * 0.01
        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, inputs):
        h_states = {-1: np.zeros((self.hidden_size, 1))}
        outputs = {}
        
        for t, x in enumerate(inputs):
            h_states[t] = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h_states[t-1]) + self.bh)
            outputs[t] = np.dot(self.Why, h_states[t]) + self.by
            
        return outputs, h_states

if __name__ == "__main__":
    seq_len = 5
    input_dim = 3
    hidden_dim = 4
    output_dim = 2

    rnn = VanillaRNN(input_dim, hidden_dim, output_dim)
    inputs = [np.random.randn(input_dim, 1) for _ in range(seq_len)]
    
    outputs, h_states = rnn.forward(inputs)
    print("时间步 0 的输出维度:", outputs[0].shape)
    print("最后一个时间步的隐状态:", h_states[seq_len - 1].ravel())

```

