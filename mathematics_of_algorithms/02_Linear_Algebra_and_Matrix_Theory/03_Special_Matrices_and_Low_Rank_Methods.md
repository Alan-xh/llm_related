# 第三章：特殊矩阵结构、低秩分解与高维张量基础

## 1. 核心概念与特殊矩阵

### 1.1 正定与半正定矩阵 (Positive Definite Matrices)

对称矩阵 $A \in \mathbb{R}^{n \times n}$ 称为：

* **正定 ($A \succ 0$)**：若对任意非零向量 $x \in \mathbb{R}^n$，$x^T A x > 0$。 equivalent to 所有特征值 $\lambda_i > 0$。
* **半正定 ($A \succeq 0$)**：若 $x^T A x \ge 0$。 equivalent to 所有特征值 $\lambda_i \ge 0$。
* **Cholesky 分解**：若 $A \succ 0$，则存在唯一的下三角矩阵 $L$（对角线元素全正），使得 $A = L L^T$。

### 1.2 投影矩阵 (Projection Matrices) 与 伪逆 (Pseudoinverse)

* **投影矩阵 $P$**：满足自幂性 $P^2 = P$。若正交投影，则 $P = P^T$。
* 投影到子空间 $\mathcal{C}(A)$ 的矩阵：$P = A (A^T A)^{-1} A^T$。


* **Moore-Penrose 广义逆 (Pseudoinverse $A^+$)**：
对任意 $A \in \mathbb{R}^{m \times n}$，若其 SVD 为 $A = U \Sigma V^T$，则伪逆定义为：

$$A^+ = V \Sigma^+ U^T$$



其中 $\Sigma^+$ 是将 $\Sigma$ 的非零奇异值取倒数并转置得到的矩阵。
* **最小二乘通解**：对于超定方程组 $A x = b$，极小范数最小二乘解为 $\hat{x} = A^+ b$。



---

## 2. 低秩适应 (LoRA) 理论与公式证明

在大型语言模型 (LLM) 的微调过程中，假设预训练权重为 $W_0 \in \mathbb{R}^{d \times k}$。LoRA 假设权重的更新量 $\Delta W$ 具有很低的**本征秩 (Intrinsic Rank)** $r \ll \min(d, k)$。

### 2.1 参数分解形式

将 $\Delta W$ 因子分解为两个低秩矩阵的乘积：


$$W = W_0 + \Delta W = W_0 + \frac{\alpha}{r} B A$$


其中：

* $A \in \mathbb{R}^{r \times k}$，通常使用高斯随机初始化 $\mathcal{N}(0, \sigma^2)$。
* $B \in \mathbb{R}^{d \times r}$，初始化为全零矩阵 $\mathbf{0}$，确保在训练开始时 $\Delta W = 0$，模型输出与原始预训练模型完全一致。
* $\alpha$ 为常数缩放系数。

### 2.2 显存与计算开销对比分析

设输入维度为 $k=4096$，输出维度 $d=4096$，秩 $r=8$：

* 原参数量：$4096 \times 4096 \approx 1.67 \times 10^7$ (16.7M)
* LoRA 参数量：$r \times (d + k) = 8 \times (4096 + 4096) = 65,536$ (0.065M)
* **参数量削减幅度**：> 99.6%！

---

## 3. AI/ML 经典应用案例：从零手写 PyTorch 风格的 LoRA 线性层

```python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, lora_alpha=16, lora_dropout=0.05):
        super(LoRALinear, self).__init__()
        
        # 1. 冻结原始预训练权重
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.scaling = lora_alpha / rank
        
        # 原始权重 W0
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        self.weight.requires_grad = False # 冻结
        self.bias.requires_grad = False   # 冻结
        
        # 2. 可训练的低秩矩阵 A 和 B
        if rank > 0:
            self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
            self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
            self.dropout = nn.Dropout(p=lora_dropout)
            self.reset_lora_parameters()
            
    def reset_lora_parameters(self):
        # A 采用 Kaiming 均匀分布初始化，B 采用全零初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        # 主路计算: x * W0^T + b
        result = nn.functional.linear(x, self.weight, self.bias)
        
        # 旁路低秩计算: (x * A^T) * B^T * scaling
        if self.rank > 0:
            lora_out = nn.functional.linear(self.dropout(x), self.lora_A) # (batch, rank)
            lora_out = nn.functional.linear(lora_out, self.lora_B)        # (batch, out_features)
            result += lora_out * self.scaling
            
        return result

# 验证 LoRA 模块
if __name__ == "__main__":
    x = torch.randn(4, 128) # Batch=4, dim=128
    lora_layer = LoRALinear(in_features=128, out_features=256, rank=8)
    
    output = lora_layer(x)
    
    trainable_params = sum(p.numel() for p in lora_layer.parameters() if p.requires_grad)
    frozen_params = sum(p.numel() for p in lora_layer.parameters() if not p.requires_grad)
    
    print(f"输出维度: {output.shape}")
    print(f"可训练参数量 (LoRA B & A): {trainable_params}")
    print(f"冻结参数量 (W0 & bias): {frozen_params}")

```

