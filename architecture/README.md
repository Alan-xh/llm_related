# 常见模型架构教学实现

本目录按模型家族拆分为相互独立的 PyTorch 教学实现。每个模型家族继续按主要版本或代表性架构分目录，版本目录后续统一放置：

- `model.py`：网络结构、损失函数和关键数据变换。
- `train.py`：最小可运行训练循环。
- `inference.py`：模型推理或生成入口。
- `README.md`：论文来源、开源代码实现库、数据流、Tensor Shape、核心公式和运行命令。

当前版本目录先完成结构化占位，后续实现可以直接放入对应目录，不需要再调整导航。

## 目录

| 目录 | 主要版本/架构 | 任务 |
| --- | --- | --- |
| [`yolo/`](./yolo/) | YOLOv1、v3、v4、v5、v8、v10、YOLO26 | 多尺度目标检测 |
| [`detr/`](./detr/) | DETR、Deformable-DETR、DAB-DETR、DN-DETR、DINO、RT-DETR、Grounding-DINO | Transformer 目标检测 |
| [`SAM/`](./SAM/) | SAM、MobileSAM、FastSAM、SAM-HQ、SAM2、SAM2.1、SAM3、SAM3.1 | Promptable 图像/视频分割 |
| [`qwen/`](./qwen/) | Qwen1、Qwen1.5、Qwen2、Qwen2.5、Qwen3 | Decoder-only 语言模型 |
| [`glm/`](./glm/) | GLM-130B、ChatGLM、ChatGLM2、ChatGLM3、GLM-4、GLM-4.5、GLM-4.6、GLM-4.7 | 生成式语言模型 |
| [`sd/`](./sd/) | Stable Diffusion v1、v2、SDXL、SD3、SD3.5 | 文生图潜空间扩散 |
| [`wan/`](./wan/) | Wan2.1、Wan2.2 | 文生视频/图生视频扩散 Transformer |

这些实现用于理解模型架构和训练/推理接口，参数规模、数据管道、分布式策略和 checkpoint 兼容性均不是官方模型的完整复刻。

## 各版本创新技术

### DETR

| 版本 | 核心创新 |
| --- | --- |
| `DETR` | 用 object queries 和 Hungarian matching 直接做集合预测，去除 anchor 与 NMS。 |
| `Deformable-DETR` | 在多尺度特征上只采样少量参考点附近的特征，降低全局注意力成本并加快收敛。 |
| `DAB-DETR` | 将动态 anchor box 作为 query，使 query 自带位置先验并逐层迭代边界框。 |
| `DN-DETR` | 在训练时加入带噪标签和框的 denoising query，缓解 DETR 收敛慢的问题。 |
| `DINO` | 改进 denoising anchor box、query selection 和对比式去噪，提升端到端检测精度。 |
| `RT-DETR` | 使用高效混合编码器和 IoU-aware query selection，面向实时检测保持 NMS-free 推理。 |
| `Grounding-DINO` | 融合文本和图像特征，以语言引导 query，实现开放词汇和短语定位检测。 |

### SAM

| 版本 | 核心创新 |
| --- | --- |
| `SAM` | 用 image encoder、prompt encoder 和轻量 mask decoder 组成可提示的通用图像分割模型。 |
| `MobileSAM` | 用轻量视觉编码器蒸馏 SAM 的图像特征，降低移动端和边缘设备的推理成本。 |
| `FastSAM` | 先生成全图实例候选，再根据 prompt 选择目标，以实时实例分割模型替代大 image encoder。 |
| `SAM-HQ` | 注入更细粒度的高质量特征，改善物体边界、细小结构和复杂细节。 |
| `SAM2` | 引入 streaming memory，将 promptable segmentation 扩展到视频并支持跨帧交互式纠正。 |
| `SAM2.1` | 发布改进 checkpoint、训练/微调代码和 demo，延续 SAM 2 的图像/视频推理接口。 |
| `SAM3` | 使用文本、示例图和视觉 prompt 做开放词汇概念分割，并统一图像检测与视频跟踪。 |
| `SAM3.1` | 通过 shared-memory Object Multiplex 联合处理多个目标，提升多目标视频推理吞吐。 |

### Qwen

| 版本 | 核心创新 |
| --- | --- |
| `Qwen1` | 以大规模中英文及多语言数据预训练，并通过 ChatML 统一基础模型、对话和工具调用格式。 |
| `Qwen1.5` | 统一 dense 与 MoE 模型系列，强化指令对齐、代码/数学能力和多尺寸部署覆盖。 |
| `Qwen2` | 在多数尺寸中引入 Grouped-Query Attention，并增强长上下文、多语言和代码建模能力。 |
| `Qwen2.5` | 扩大高质量预训练与后训练数据，重点提升长上下文、代码、数学和结构化输出能力。 |
| `Qwen3` | 将 thinking 与 non-thinking 模式合并到同一模型，并同时提供 dense/MoE、工具调用和多语言能力。 |

### GLM

| 版本 | 核心创新 |
| --- | --- |
| `GLM-130B` | 使用自回归空白填充目标和二维位置编码，将双向理解与自回归生成结合起来。 |
| `ChatGLM` | 在 GLM 目标上加入中英双语对话对齐，形成面向消费级硬件的对话模型。 |
| `ChatGLM2` | 改进注意力和上下文建模，支持更长上下文，并优化推理效率与对话质量。 |
| `ChatGLM3` | 强化工具调用、代码解释器和多轮对话模板，使模型从聊天扩展到任务执行。 |
| `GLM-4` | 扩展长上下文、多语言、工具调用和视觉理解，形成统一的多模态模型系列。 |
| `GLM-4.5` | 引入面向 agentic reasoning 和 coding 的混合推理能力，强化复杂任务分解与工具使用。 |
| `GLM-4.6` | 继续优化长上下文、代码生成和智能体工作流，提升多步任务的稳定性。 |
| `GLM-4.7` | 强化 thinking-before-acting、终端操作和多语言智能体编程能力。 |

### Stable Diffusion

| 版本 | 核心创新 |
| --- | --- |
| `SD-v1` | 首次将扩散过程放到 VAE 潜空间中，并用 CLIP 文本特征通过 U-Net cross-attention 控制生成。 |
| `SD-v2` | 切换到 OpenCLIP 文本编码器，支持更高分辨率、v-prediction 和深度条件等扩展。 |
| `SDXL` | 使用双文本编码器、更大的 U-Net 以及 base/refiner 两阶段流水线，改善高分辨率生成。 |
| `SD3` | 使用 MMDiT 统一处理文本和图像 token，并采用 flow matching 改善文本-图像对齐。 |
| `SD3.5` | 在 MMDiT 和训练数据上继续改进，同时提供不同规模与 Turbo 变体以平衡质量和速度。 |

### Wan

| 版本 | 核心创新 |
| --- | --- |
| `Wan2.1` | 使用面向视频的 3D causal VAE、Flow Matching DiT 和 T5 文本条件，统一支持 T2V/I2V。 |
| `Wan2.2` | 将 MoE 引入视频扩散，在高噪声和低噪声阶段使用不同专家，并结合高压缩 VAE 支持高分辨率 TI2V。 |

上表只列出最具代表性的架构或训练创新；具体版本目录中的实现会进一步补充张量 Shape、损失函数和推理流程。

## 目录层级

```text
architecture/
├── detr/
│   ├── DETR/
│   ├── Deformable-DETR/
│   ├── DAB-DETR/
│   ├── DN-DETR/
│   ├── DINO/
│   ├── RT-DETR/
│   └── Grounding-DINO/
├── SAM/
│   ├── SAM/
│   ├── MobileSAM/
│   ├── FastSAM/
│   ├── SAM-HQ/
│   ├── SAM2/
│   ├── SAM2.1/
│   ├── SAM3/
│   └── SAM3.1/
├── glm/
│   ├── GLM-130B/
│   ├── ChatGLM/
│   ├── ChatGLM2/
│   ├── ChatGLM3/
│   ├── GLM-4/
│   ├── GLM-4.5/
│   ├── GLM-4.6/
│   └── GLM-4.7/
├── qwen/
│   ├── Qwen1/
│   ├── Qwen1.5/
│   ├── Qwen2/
│   ├── Qwen2.5/
│   └── Qwen3/
├── sd/
│   ├── SD-v1/
│   ├── SD-v2/
│   ├── SDXL/
│   ├── SD3/
│   └── SD3.5/
├── wan/
│   ├── Wan2.1/
│   └── Wan2.2/
└── yolo/
    ├── YOLOv1/
    ├── YOLOv3/
    ├── YOLOv4/
    ├── YOLOv5/
    ├── YOLOv8/
    ├── YOLOv10/
    └── YOLO26/
```

从仓库根目录运行具体版本的实现，例如：

```bash
python architecture/yolo/YOLOv1/model.py
python architecture/qwen/Qwen3/inference.py
```

也可以把目录作为 Python package 使用：

```bash
python -m architecture.qwen.Qwen3.inference
```
