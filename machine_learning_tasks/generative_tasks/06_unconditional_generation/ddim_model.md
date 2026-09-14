# 去噪扩散隐式模型（DDIM）技术架构与训练接口文档

## 1. DDIM 解决什么问题

DDIM（Denoising Diffusion Implicit Models）与 DDPM 使用相同的训练目标：

1. 从数据分布采样干净图像 `x_0`。
2. 随机采样时间步 `t` 和高斯噪声 `epsilon`。
3. 根据前向过程直接构造 `x_t`。
4. 训练 `epsilon_theta(x_t, t)` 预测加入的噪声。

两者的主要区别在于反向采样：

- DDPM 通常按照 `T -> T-1 -> ... -> 0` 的完整链条采样，并在每一步注入随机噪声。
- DDIM 在训练不变的前提下，从完整时间轴中抽取较短的子序列，例如 `999, 979, ..., 19, 0`，因此可以用较少的网络调用完成采样。
- `eta=0` 时，给定同一个初始高斯噪声，采样轨迹是确定性的。
- `eta>0` 时，每一步会加入额外随机噪声；`eta` 越大，采样随机性越强。

本目录中的 [train_ddim.py](./train_ddim.py) 是一个教学版、可独立运行的实现。它复用 DDPM 的噪声预测训练目标，但额外实现了可跳步的 DDIM 采样器。

---

## 2. 前向过程：训练阶段与 DDPM 相同

定义：

$$
\alpha_t = 1 - \beta_t,\qquad
\bar{\alpha}_t = \prod_{s=1}^{t}\alpha_s
$$

给定干净样本 `x_0`，任意时间步的带噪样本可以直接计算：

$$
q(x_t\mid x_0)
=\mathcal{N}\left(
x_t;\sqrt{\bar{\alpha}_t}x_0,
(1-\bar{\alpha}_t)\mathbf{I}
\right)
$$

重参数化形式为：

$$
x_t
=\sqrt{\bar{\alpha}_t}x_0
+\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon\sim\mathcal{N}(0,\mathbf{I})
$$

训练损失为：

$$
\mathcal{L}
=\mathbb{E}_{x_0,t,\epsilon}
\left[
\left\|
\epsilon-\epsilon_\theta(x_t,t)
\right\|_2^2
\right]
$$

因此，一个已经训练好的 DDPM 噪声预测器通常可以直接配合 DDIM 采样器使用，不需要为 DDIM 修改训练目标。

---

## 3. DDIM 反向采样公式

设当前采样时间步为 `t`，下一个较小的时间步为 `s`，满足 `s < t`。网络先预测噪声：

$$
\hat{\epsilon}_t=\epsilon_\theta(x_t,t)
$$

根据噪声预测得到对干净样本的估计：

$$
\hat{x}_0
=\frac{x_t-\sqrt{1-\bar{\alpha}_t}\hat{\epsilon}_t}
{\sqrt{\bar{\alpha}_t}}
$$

DDIM 的广义更新公式为：

$$
x_s
=\sqrt{\bar{\alpha}_s}\hat{x}_0
+\sqrt{1-\bar{\alpha}_s-\sigma_t^2}\hat{\epsilon}_t
+\sigma_t z
$$

其中：

$$
z\sim\mathcal{N}(0,\mathbf{I})
$$

$$
\sigma_t
=\eta
\sqrt{\frac{1-\bar{\alpha}_s}{1-\bar{\alpha}_t}}
\sqrt{1-\frac{\bar{\alpha}_t}{\bar{\alpha}_s}}
$$

特殊情况：

| 设置 | 采样特性 |
| --- | --- |
| `eta=0` | 确定性 DDIM，不在中间步骤注入随机噪声 |
| `eta>0` | 随机 DDIM，保留可控的随机性 |
| `sampling_steps=T` | 不跳步，使用完整时间轴 |
| `sampling_steps << T` | 大幅减少 U-Net 调用次数，通常更快 |

最后一步使用 `s=-1` 的概念性状态，并令 `bar_alpha_s=1`，此时更新结果就是 `x_0` 的估计。

---

## 4. 代码架构

```text
                    训练阶段
 [x0] -- q_sample(t, epsilon) --> [xt]
                                  |
                                  v
                         TinyUNet(xt, t)
                                  |
                                  v
                       predicted_noise
                                  |
              MSE(predicted_noise, epsilon)

                    推理阶段
 [Gaussian noise xT]
          |
          v
 [DDIMSampler: t -> s]
          |  预测 epsilon_theta
          |  重建 x0_hat
          |  按 eta 加入 sigma * z
          v
       [sample x0]
```

### 4.1 张量 Shape

以 MNIST 为例，`B=64`：

| 模块 | 输入 Shape | 输出 Shape | 说明 |
| --- | --- | --- | --- |
| 数据样本 `x0` | `[64, 1, 28, 28]` | `[64, 1, 28, 28]` | 归一化到 `[-1, 1]` |
| 时间步 `t` | `[64]` | `[64]` | 每个样本随机选择一个时间步 |
| `q_sample` | `x0`, `t`, `epsilon` | `[64, 1, 28, 28]` | 直接构造 `x_t` |
| `TinyUNet` | `x_t`, `t` | `[64, 1, 28, 28]` | 预测 `epsilon_theta` |
| `predicted_x0` | `x_t`, `predicted_noise` | `[64, 1, 28, 28]` | 估计干净图像 |
| DDIM step | `x_t`, `x0_hat`, `epsilon_hat` | `[64, 1, 28, 28]` | 得到较小时间步 `x_s` |

### 4.2 数学公式与代码映射

| 数学概念 | 代码位置 |
| --- | --- |
| `beta_t` 与 `bar_alpha_t` | `linear_beta_schedule`、`DiffusionSchedule.__init__` |
| 前向采样 `x_t` | `DiffusionSchedule.q_sample` |
| 噪声预测器 | `TinyUNet.forward` |
| 采样时间子序列 | `DDIMSampler.sample` 中的 `time_indices` |
| 干净样本估计 `x0_hat` | `DDIMSampler.sample` 中的 `predicted_x0` |
| `sigma_t` 与 `eta` | `DDIMSampler.sample` 中的 `sigma` |
| 反向更新 | `DDIMSampler.sample` 最后的 `x = ...` |
| 训练入口 | `main` |

---

## 5. 运行方式

在仓库根目录执行：

```bash
# 无需下载数据的快速示例
python machine_learning_tasks/generative_tasks/06_unconditional_generation/train_ddim.py \
  --dataset synthetic \
  --epochs 1 \
  --synthetic-samples 256 \
  --sampling-steps 20 \
  --output-dir ./ddim_outputs

# 使用 MNIST 训练
python machine_learning_tasks/generative_tasks/06_unconditional_generation/train_ddim.py \
  --dataset mnist \
  --epochs 10 \
  --sampling-steps 50 \
  --eta 0.0 \
  --output-dir ./ddim_mnist_outputs
```

脚本会输出：

- `ddim_model.pt`：模型权重及图像尺寸、时间步等元信息。
- `ddim_samples.png`：训练结束后的采样网格。

常用参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--timesteps` | `1000` | 训练时的完整扩散时间步数 |
| `--sampling-steps` | `50` | DDIM 推理时实际使用的步数 |
| `--eta` | `0.0` | 采样随机性，`0` 为确定性 DDIM |
| `--dataset` | `synthetic` | `synthetic`、`mnist` 或 `fashion-mnist` |
| `--device` | 自动选择 | `cuda` 或 `cpu` |

### 与 DDPM 的关系

DDIM 训练阶段仍然使用：

```python
xt = schedule.q_sample(x0, t, noise)
predicted_noise = model(xt, t)
loss = F.mse_loss(predicted_noise, noise)
```

变化只发生在采样阶段：DDPM 使用逐步的随机后验采样，而 DDIM 使用 `predicted_x0`、预测噪声和 `sigma_t` 组成的广义更新。

---

## 6. 参考文献

Song, J., Meng, C., & Ermon, S. (2021). *Denoising Diffusion Implicit Models*. ICLR 2021. arXiv:2010.02502.
