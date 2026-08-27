本报告介绍 **Stable Diffusion** 所依据的 **Latent Diffusion Models（LDM）**。其核心思想是先用自编码器把高分辨率图像压缩到低维 latent space，再在 latent 上执行扩散和去噪，最后由解码器还原图像。相较于直接在像素空间扩散，LDM 显著降低了计算和显存成本；通过 cross-attention，还可以把文本、类别、语义分割和其他条件注入生成过程。Stable Diffusion 于 2022 年公开传播，成为这一类 latent diffusion 技术的代表性应用。

论文：*High-Resolution Image Synthesis with Latent Diffusion Models*

作者：Robin Rombach、Andreas Blattmann、Dominik Lorenz 等
发表时间：2021 年（arXiv:2112.10752）；Stable Diffusion 公开模型于 2022 年广泛使用

---

### 2. 核心创新点提取 (Novelty & Key Contributions)

* **研究问题与背景 (Research Gap)**：
* 像素空间扩散需要在高分辨率张量上反复运行 U-Net，计算成本随分辨率快速上升。
* 直接压缩到过低维度会丢失纹理、边缘和局部结构。
* 条件生成需要一个灵活接口，支持文本等不同模态，而不是为每种条件设计独立网络。

* **核心技术贡献 (Core Technical Innovation)**：
1. **潜空间扩散**：在预训练 autoencoder 的 latent 表示上执行扩散，降低空间尺寸和计算量。
2. **感知压缩**：保留对人类视觉重要的结构，同时去除像素级冗余。
3. **cross-attention 条件控制**：把文本 token 或其他条件作为上下文注入 U-Net。
4. **高分辨率合成**：在较低计算成本下生成高分辨率图像，并支持 inpainting、super-resolution 等任务。
5. **模块化系统**：autoencoder、扩散 U-Net、文本编码器和采样器可以分别优化或替换。

* **本质区别 (Vs. Pixel-space DDPM)**：
* DDPM 直接在图像像素上扩散，LDM 在压缩 latent 上扩散。
* LDM 的速度和显存效率更好，但生成质量受 autoencoder 压缩损失和文本对齐能力限制。

---

### 3. 方法论/技术细节精炼 (Methodology Highlights)

* **图像压缩与解码**：

$$z=\mathcal{E}(x),\quad
\hat{x}=\mathcal{D}(z)$$

其中 $\mathcal{E}$ 是 encoder，$\mathcal{D}$ 是 decoder。扩散模型学习 latent $z$ 的生成分布。

* **Latent diffusion**：

先对干净 latent $z_0$ 加噪：

$$z_t=\sqrt{\bar{\alpha}_t}z_0+
\sqrt{1-\bar{\alpha}_t}\epsilon$$

再训练 U-Net 预测噪声：

$$\mathcal{L}_{\mathrm{LDM}}=
\mathbb{E}_{z_0,t,\epsilon}
\left[\|\epsilon-
\epsilon_\theta(z_t,t,\tau_\theta(y))\|_2^2\right]$$

其中 $y$ 是条件信息，$\tau_\theta$ 是文本编码器。

* **Cross-attention**：

$$\operatorname{Attention}(Q,K,V)=
\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V$$

U-Net 特征作为 query，条件 token 作为 key/value，使每个空间位置能够读取文本语义。

* **Stable Diffusion 系统组成**：
* VAE 将 $512\times512$ 图像压缩到更小的 latent feature map。
* 文本编码器将 prompt 转换为 token embeddings。
* U-Net 在 latent 空间执行逐步去噪。
* sampler 从高斯噪声开始，经过若干步反向更新生成 latent，再由 VAE decoder 解码。
* 实际系统常结合 classifier-free guidance 调节提示词遵循程度与样本多样性。

* **关键假设与边界条件**：
* autoencoder 的 latent 保留足够的语义和空间信息。
* 训练图文对能建立可靠的文本-图像对应。
* 采样器、步数、guidance scale 和 prompt 共同决定最终结果。

---

### 4. 实验设计与严谨性评估 (Experiments & Rigor Evaluation)

* **评测任务**：
* unconditional 和 class-conditional 图像生成。
* 文本到图像生成。
* 超分辨率、图像修复和语义条件生成。
* 与像素空间 diffusion、GAN 和其他高分辨率生成方法比较。

* **主要观察**：

| 方面 | 观察 |
| --- | --- |
| 计算效率 | latent 空间显著减少 U-Net 的空间计算 |
| 视觉质量 | 在保持较高质量的同时支持高分辨率 |
| 条件控制 | cross-attention 能统一接入文本和空间条件 |
| 生成速度 | 相比像素扩散更快，但仍需多步采样 |

* **消融分析 (Ablation Analysis)**：
* latent 压缩比例在速度、细节和重建误差之间形成权衡。
* 仅使用像素级重构会产生模糊或纹理损失，感知损失和对抗损失有助于保持视觉质量。
* cross-attention 比简单拼接条件更灵活，适合变长文本和多种条件。
* guidance scale 增大通常提高 prompt 遵循，但可能降低多样性并产生过饱和伪影。

* **审稿人视角的保留意见**：
* Stable Diffusion 不是与 LDM 论文完全同名的独立方法，公开模型还包含具体数据、文本编码器和工程配方。
* 网络图文数据的质量、版权、重复和过滤策略会显著影响结果。
* FID 等指标不能充分衡量文本遵循、构图准确性和细节真实性。
* 模型可能生成训练集近似副本或放大数据中的社会偏见。

---

### 5. 结论与局限性 (Conclusions & Limitations)

* **主要结论**：
* 在压缩 latent 上扩散可以大幅降低高分辨率图像生成成本。
* cross-attention 提供了统一的条件生成接口。
* LDM 在效率、质量和任务灵活性之间取得了实用平衡。

* **局限性**：
* VAE 压缩会损失细粒度纹理、精确文字和像素级信息。
* 多步采样仍带来明显延迟，实时生成需要额外加速。
* 文本编码器可能误解复杂组合关系、空间关系和计数。
* 生成质量依赖训练数据分布，模型对长尾概念和少数群体可能表现不均衡。

* **未充分讨论的风险与盲区**：
* 高保真文生图降低了误导性图片、冒充和非自愿内容的制作门槛。
* 训练数据的版权和创作者归因问题需要在模型之外建立治理机制。

---

### 6. 启发与技术迁移 (Actionable Takeaways)

1. **先压缩再生成**：对高维信号先学习语义保真的 latent 表示，再执行生成建模，通常更节省资源。
2. **条件注入应模块化**：cross-attention 可以把文本、布局、深度、边缘和分割统一成上下文接口。
3. **压缩比例不是越大越好**：需要根据目标任务在速度、重建质量和细节保真之间选择。
4. **采样器决定产品体验**：步数、调度器、guidance 和蒸馏应一起优化，而不是只换模型权重。
5. **生成系统要加入 provenance**：水印、内容溯源、隐私测试、版权政策和滥用检测应与模型能力同步建设。
