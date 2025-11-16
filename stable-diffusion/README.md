# stable diffusion

stable diffusion 由 comvis(德国慕尼黑大学和海德堡大学 IWR) 发布 2022 CVPR

github: https://github.com/CompVis/stable-diffusion


## 2D卷积输入输出形状计算公式

| 符号 | 描述 |
| :--- | :--- |
| **I** | **输入特征图的边长** (Input size, 假设为正方形 $I \times I$) |
| **K** | **卷积核（Kernel/Filter）的边长** (假设为正方形 $K \times K$) |
| **P** | **填充（Padding）的边长** (在每一边添加的像素数) |
| **S** | **步长（Stride）** (卷积核移动的步数) |
| **O** | **输出特征图的边长** (Output size, 假设为正方形 $O \times O$) |

$$O = \left\lfloor \frac{I - K + 2P}{S} \right\rfloor + 1$$

## 