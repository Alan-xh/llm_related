
# Transformer 模型

## 1. 算法原理

Transformer 抛弃了传统 RNN/CNN 的循环与卷积结构，完全依赖**自注意力机制（Self-Attention Mechanism）** 来捕获序列中任意位置之间的全局依赖关系。

核心特性包括：

1. **缩放点积注意力 (Scaled Dot-Product Attention)**：高效计算序列内各 Token 的关联度。
2. **多头注意力 (Multi-Head Attention)**：允许模型在不同子空间联合关注来自不同位置的信息。
3. **位置编码 (Positional Encoding)**：由于缺乏循环结构，需要引入位置信息表示 Token 在序列中的位置。
4. **残差连接与层归一化 (Residual Connection & LayerNorm)**：保证深层网络稳定的梯度传播。

---

## 2. 数学公式与推导

### 2.1 缩放点积注意力 (Scaled Dot-Product Attention)

给定输入特征矩阵 $X \in \mathbb{R}^{N \times d}$，通过线性变换得到 Query ($Q$), Key ($K$), Value ($V$):


$$Q = X W_Q, \quad K = X W_K, \quad V = X W_V$$

注意力权重与最终输出计算公式为：


$$\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{Q K^T}{\sqrt{d_k}} \right) V$$

### 2.2 多头注意力 (Multi-Head Attention)

将 $Q, K, V$ 投影到 $h$ 个不同的子空间：


$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O$$

---

## 3. ASCII 结构图

```
                  [ 输出 Vector ]
                        ^
                        |
                 [ LayerNorm ]
                        |
            +-----------+-----------+
            | (残差连接)              |
            |                       |
            |          [ 前馈网络 (FFN) ]
            |                       |
            +-----------+-----------+
                        |
                 [ LayerNorm ]
                        |
            +-----------+-----------+
            | (残差连接)              |
            |                       |
            |   [ 多头注意力机制 ]   |
            |   ( Multi-Head Attn ) |
            |                       |
            +-----------+-----------+
                        ^
                        |
              [ 输入嵌入 + 位置编码 ]

```

---

## 4. Python 代码实现 (基于 NumPy)

```python
import numpy as np

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
        
    attn_weights = softmax(scores)
    output = np.matmul(attn_weights, V)
    
    return output, attn_weights

if __name__ == "__main__":
    seq_len = 4
    d_model = 8
    
    np.random.seed(42)
    X = np.random.randn(seq_len, d_model)
    
    WQ = np.random.randn(d_model, d_model)
    WK = np.random.randn(d_model, d_model)
    WV = np.random.randn(d_model, d_model)
    
    Q = np.dot(X, WQ)
    K = np.dot(X, WK)
    V = np.dot(X, WV)
    
    out, weights = scaled_dot_product_attention(Q, K, V)
    print("注意力输出矩阵形状:", out.shape)
    print("注意力权重矩阵 [0]:\n", weights.round(3))

```

