# 稳定扩散 v1 模型卡
本模型卡重点介绍与稳定扩散模型相关的模型，可在[此处](https://github.com/CompVis/stable-diffusion)获取。

## 模型详情
- **开发者：** 罗宾·隆巴赫（Robin Rombach），帕特里克·埃塞尔（Patrick Esser）
- **模型类型：** 基于扩散的文本到图像生成模型
- **语言：** 英语
- **许可证：** [专有许可证](LICENSE)
- **模型描述：** 该模型可用于根据文本提示生成和修改图像。它是一个[潜在扩散模型](https://arxiv.org/abs/2112.10752)，使用固定的预训练文本编码器（[CLIP ViT-L/14](https://arxiv.org/abs/2103.00020)），如[Imagen 论文](https://arxiv.org/abs/2205.11487)中建议。
- **更多信息资源：** [GitHub 仓库](https://github.com/CompVis/stable-diffusion)，[论文](https://arxiv.org/abs/2112.10752)。
- **引用格式：**

      @InProceedings{Rombach_2022_CVPR,
          author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
          title     = {高分辨率图像合成与潜在扩散模型},
          booktitle = {IEEE/CVF 计算机视觉与模式识别会议论文集 (CVPR)},
          month     = {六月},
          year      = {2022},
          pages     = {10684-10695}
      }

# 使用

## 直接使用
该模型仅用于研究目的。可能的研究领域和任务包括：

- 安全部署可能生成有害内容的模型。
- 探索和理解生成模型的局限性和偏见。
- 生成艺术品并用于设计和其他艺术过程。
- 在教育或创意工具中的应用。
- 生成模型的研究。

以下描述了禁止的使用场景。

### 误用、恶意使用和超出范围的使用
_注：本节内容摘自 [DALLE-MINI 模型卡](https://huggingface.co/dalle-mini/dalle-mini)，但同样适用于稳定扩散 v1。_

该模型不得用于故意创建或传播可能对人造成敌对或疏远环境的图像。这包括生成人们可预见会感到不安、痛苦或冒犯的图像；或传播历史或当前刻板印象的内容。

#### 超出范围使用
该模型未被训练为对人或事件的真实或准确表示，因此使用该模型生成此类内容超出了其能力范围。

#### 误用和恶意使用
使用该模型生成对个人残忍的内容属于误用。这包括但不限于：

- 生成对人或其环境、文化、宗教等的贬低、去人性化或其他有害表示。
- 故意推广或传播歧视性内容或有害刻板印象。
- 未经同意冒充个人。
- 未得到可能看到内容的个人同意的性内容。
- 虚假信息和误导信息。
- 极端暴力和血腥的表示。
- 违反使用条款分享受版权或许可保护的材料。
- 违反使用条款分享对受版权或许可保护材料的更改内容。

## 局限性和偏见

### 局限性

- 该模型无法实现完美的照片真实感。
- 该模型无法渲染清晰可读的文本。
- 该模型在涉及组合性的较困难任务上表现不佳，例如渲染“红色立方体在蓝色球体之上”的图像。
- 人脸和人物的生成可能不准确。
- 该模型主要使用英语描述训练，在其他语言中的表现较差。
- 模型的自动编码部分是有损的。
- 该模型在[LAION-5B](https://laion.ai/blog/laion-5b/)大规模数据集上训练，包含成人内容，未经额外安全机制和考虑不适合产品使用。
- 未采取额外措施对数据集进行去重。因此，我们观察到训练数据中重复图像存在一定程度的记忆化。
  可在[https://rom1504.github.io/clip-retrieval/](https://rom1504.github.io/clip-retrieval/)搜索训练数据，以协助检测记忆化的图像。

### 偏见
虽然图像生成模型的能力令人印象深刻，但它们也可能强化或加剧社会偏见。
稳定扩散 v1 主要在 [LAION-2B(en)](https://laion.ai/blog/laion-5b/) 的子集上训练，该数据集仅包含英语描述的图像。
使用其他语言的社区和文化的文本和图像可能未被充分考虑。
这影响了模型的整体输出，因为白色和西方文化常被设定为默认值。此外，模型生成非英语提示内容的能力明显低于英语提示。
稳定扩散 v1 反映并加剧了偏见，无论输入或意图如何，建议用户谨慎使用。

## 训练

**训练数据**
模型开发者使用了以下数据集进行模型训练：

- LAION-5B 及其子集（见下一节）

**训练流程**
稳定扩散 v1 是一个潜在扩散模型，结合了自动编码器和在自动编码器潜在空间中训练的扩散模型。在训练过程中：

- 图像通过编码器编码，将图像转换为潜在表示。自动编码器使用相对下采样因子 8，将形状为 H x W x 3 的图像映射到形状为 H/f x W/f x 4 的潜在表示。
- 文本提示通过 ViT-L/14 文本编码器编码。
- 文本编码器的非池化输出通过跨注意力机制输入到潜在扩散模型的 UNet 主干中。
- 损失函数是潜在表示中添加的噪声与 UNet 预测之间的重构目标。

我们目前提供以下检查点：

- `sd-v1-1.ckpt`：在 [laion2B-en](https://huggingface.co/datasets/laion/laion2B-en) 上以 `256x256` 分辨率训练 237k 步。
  在 [laion-high-resolution](https://huggingface.co/datasets/laion/laion-high-resolution) 上以 `512x512` 分辨率训练 194k 步（来自 LAION-5B 的 1.7 亿个分辨率 `>= 1024x1024` 的样本）。
- `sd-v1-2.ckpt`：从 `sd-v1-1.ckpt` 恢复。
  在 [laion-aesthetics v2 5+](https://laion.ai/blog/laion-aesthetics/) 上以 `512x512` 分辨率训练 515k 步（laion2B-en 的子集，估计美学分数 `> 5.0`，并额外过滤原始尺寸 `>= 512x512` 且估计水印概率 `< 0.5` 的图像。水印估计来自 [LAION-5B](https://laion.ai/blog/laion-5b/) 元数据，美学分数使用 [LAION-Aesthetics Predictor V2](https://github.com/christophschuhmann/improved-aesthetic-predictor) 估计）。
- `sd-v1-3.ckpt`：从 `sd-v1-2.ckpt` 恢复。在 "laion-aesthetics v2 5+" 上以 `512x512` 分辨率训练 195k 步，并将文本条件丢弃 10% 以改进[无分类器引导采样](https://arxiv.org/abs/2207.12598)。
- `sd-v1-4.ckpt`：从 `sd-v1-2.ckpt` 恢复。在 "laion-aesthetics v2 5+" 上以 `512x512` 分辨率训练 225k 步，并将文本条件丢弃 10% 以改进[无分类器引导采样](https://arxiv.org/abs/2207.12598)。

- **硬件：** 32 x 8 x A100 GPU
- **优化器：** AdamW
- **梯度累积：** 2
- **批次：** 32 x 8 x 2 x 4 = 2048
- **学习率：** 在 10,000 步内预热至 0.0001，然后保持不变

## 评估结果
使用不同无分类器引导尺度（1.5、2.0、3.0、4.0、5.0、6.0、7.0、8.0）和 50 个 PLMS 采样步骤的评估显示了检查点的相对改进：

![pareto](assets/v1-variants-scores.jpg)

使用 50 个 PLMS 步骤和 COCO2017 验证集中的 10,000 个随机提示进行评估，分辨率为 512x512。未针对 FID 分数优化。

## 环境影响

**稳定扩散 v1 估计排放**
根据该信息，我们使用 [Lacoste 等人 (2019)](https://arxiv.org/abs/1910.09700) 提出的 [机器学习影响计算器](https://mlco2.github.io/impact#compute) 估计以下 CO2 排放。硬件、运行时间、云提供商和计算区域用于估计碳影响。

- **硬件类型：** A100 PCIe 40GB
- **使用小时数：** 150,000
- **云提供商：** AWS
- **计算区域：** 美国东部
- **排放碳量（功耗 x 时间 x 根据电网位置产生的碳）：** 11,250 千克 CO2 等量

## 引用
    @InProceedings{Rombach_2022_CVPR,
        author    = {Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bj\"orn},
        title     = {高分辨率图像合成与潜在扩散模型},
        booktitle = {IEEE/CVF 计算机视觉与模式识别会议论文集 (CVPR)},
        month     = {六月},
        year      = {2022},
        pages     = {10684-10695}
    }

*此模型卡由罗宾·隆巴赫（Robin Rombach）和帕特里克·埃塞尔（Patrick Esser）撰写，基于 [DALL-E Mini 模型卡](https://huggingface.co/dalle-mini/dalle-mini)。*