# AI/ML 工程师后训练项目

这是一个围绕大语言模型、生成式 AI 和模型工程实践整理的学习与实验仓库。内容从数学基础、机器学习和 NLP，延伸到 LLM 训练与对齐、推理优化、视觉与语音、多模态以及可交付项目设计。

仓库包含两类内容：

- **学习资料**：结构化 Markdown 笔记、论文精读、面试题和自测题。
- **动手实践**：可独立阅读的 PyTorch/Transformers 实验脚本、Notebook、模型组件实现和项目设计文档。

根目录不是一个可以一键启动的单体应用，而是一组按主题组织的学习项目。不同子目录可能有独立的依赖、数据集、模型权重和运行方式。

## 内容导航

### 学习资料

| 目录 | 内容 |
| --- | --- |
| [`llm_interview_note/`](./llm_interview_note/) | LLM 基础、Transformer、MoE、分布式训练、微调、强化学习、RAG、Agent、评估与应用 |
| [`cv_interview_note/`](./cv_interview_note/) | 计算机视觉基础、CNN、ViT、检测、分割、生成模型、多模态和部署 |
| [`speech_interview_note/`](./speech_interview_note/) | 音频基础、ASR、TTS、音频生成、声纹、语音大模型、训练和流式部署 |
| [`machine_learning/`](./machine_learning/) | 线性/逻辑回归、树模型、聚类、降维、强化学习和经典神经网络 |
| [`machine_learning_tasks/`](./machine_learning_tasks/) | 分类、回归、检测、分割、生成、自监督、强化学习和模仿学习代码及笔记 |
| [`mathematics_of_algorithms/`](./mathematics_of_algorithms/) | 优化理论、线性代数、矩阵分解、概率分布和随机过程 |
| [`nlp/`](./nlp/) | HMM、CRF、N-gram、最短路径、分词和命名实体识别 |
| [`paper/`](./paper/) | 从 AlexNet、Transformer、BERT、GPT 到 DPO、FlashAttention、DeepSeek-R1 的论文精读 |
| [`自测/`](./自测/) | 与各主题对应的闭卷自测题，覆盖基础知识、训练、推理、对齐和工程实践 |

### 实验与模型实现

| 目录 | 内容 | 主要入口 |
| --- | --- | --- |
| [`handwrite_network/`](./handwrite_network/) | 手写 GPT-2、MLA、RMSNorm 和 Rotary Embedding 等基础组件 | `gpt2.py`、`MLA.py` |
| [`train_llm_from_scratch/`](./train_llm_from_scratch/) | 从零实现小型 Causal LM，并串联预训练、SFT、DPO | `train.py`、`sft_train.py`、`dpo_train.py` |
| [`train_moe_from_scratch/`](./train_moe_from_scratch/) | MoE 语言模型、路由器、专家网络、负载均衡和 SFT | `moe_train.py`、`moe_sft_train.py` |
| [`train_multimodal_from_scratch/`](./train_multimodal_from_scratch/) | SigLIP/视觉编码器与 Qwen Causal LM 对接的多模态模型 | `train.py`、`sft_train.py`、`test.py` |
| [`train_siglip_from_scratch/`](./train_siglip_from_scratch/) | SigLIP 图文表示学习模型和训练数据管道 | `model.py`、`train.py` |
| [`deepseek_learn/`](./deepseek_learn/) | DeepSeek 相关机制的复现与实验 | MLA、MTP、GRPO |
| [`ppo_from_scratch/`](./ppo_from_scratch/) | Actor、Reference、Reward、Critic 组成的 PPO/RLHF 训练流程 | `ppo_train.py` |
| [`s1_from_scratch/`](./s1_from_scratch/) | S1 推理时扩展、数据蒸馏和预算控制的学习记录 | `s1_train.py`、`generate.py` |
| [`knowledge_distillation_llm/`](./knowledge_distillation_llm/) | 教师模型与学生模型的 KL 散度知识蒸馏 | `train.py`、`utils.py` |
| [`inference_engines/`](./inference_engines/) | vLLM 与 SGLang 推理引擎核心技术的教学版实现 | `vllm/`、`sglang/` |
| [`stable-diffusion/`](./stable-diffusion/) | Stable Diffusion/LDM、Chinese-CLIP、ONNX/TensorRT 部署与相关实验 | `stable-diffusion/`、`clip-vit/` |
| [`voxcpm/`](./voxcpm/) | VoxCPM 连续空间 TTS、语音克隆和 Gradio/CLI 入口 | `app.py`、`main.py` |

### 项目设计

[`work_projects/`](./work_projects/) 用于记录面向完整交付物的项目方案，当前包括：

1. [`mini 推理引擎`](./work_projects/01_mini_inference_engine/DESIGN.md)：PagedAttention、Continuous Batching 和 INT8 量化。
2. [`MoE 双语模型全流程对齐`](./work_projects/02_moe_bilingual_llm/DESIGN.md)：预训练、SFT、DPO 和评估。
3. [`多模态 Agent 系统`](./work_projects/03_multimodal_agent/DESIGN.md)：多模态 RAG、长上下文压缩、工具调用和评测。

## 推荐学习路线

### 1. 打基础

```text
mathematics_of_algorithms/
        ↓
machine_learning/
        ↓
nlp/ + machine_learning_tasks/
        ↓
llm_interview_note/01.大语言模型基础
```

### 2. 理解 LLM

建议按以下顺序阅读：

1. [`paper/`](./paper/) 中的 Word2Vec、Transformer、BERT、GPT、LLaMA 和 Scaling Laws。
2. [`llm_interview_note/01.大语言模型基础/`](./llm_interview_note/01.大语言模型基础/)。
3. [`llm_interview_note/02.大语言模型架构/`](./llm_interview_note/02.大语言模型架构/)。
4. [`llm_interview_note/03.训练数据集/`](./llm_interview_note/03.训练数据集/) 和 [`04.分布式训练/`](./llm_interview_note/04.分布式训练/)。
5. [`llm_interview_note/05.有监督微调/`](./llm_interview_note/05.有监督微调/)、[`07.强化学习/`](./llm_interview_note/07.强化学习/) 和 [`08.检索增强rag/`](./llm_interview_note/08.检索增强rag/)。
6. [`llm_interview_note/06.推理/`](./llm_interview_note/06.推理/)、[`09.大语言模型评估/`](./llm_interview_note/09.大语言模型评估/) 和 [`10.大语言模型应用/`](./llm_interview_note/10.大语言模型应用/)。

### 3. 从组件到完整训练流程

```text
handwrite_network/
        ↓
train_llm_from_scratch/
        ↓
train_moe_from_scratch/
        ↓
train_multimodal_from_scratch/ + train_siglip_from_scratch/
        ↓
inference_engines/ + work_projects/
```

每个主题学习后，可以回到 [`自测/`](./自测/) 中进行闭卷回忆。建议先只看问题，完成回答后再回到笔记核对。

## 环境准备

### 基础环境

- Python `3.11`，版本记录在 [`.python-version`](./.python-version)。
- 训练和大多数模型推理建议使用带 CUDA 的 NVIDIA GPU。
- CPU 可用于阅读、运行部分 mock/demo、数据处理和不依赖 GPU kernel 的代码。
- Node.js 仅在使用 Docsify 在线阅读笔记时需要。

根目录使用 [`pyproject.toml`](./pyproject.toml) 管理基础依赖，同时保留了 [`uv.lock`](./uv.lock) 和现有的 [`requiremens.txt`](./requiremens.txt) 依赖快照。文件名 `requiremens.txt` 按仓库现状保留。

使用 uv 安装根目录基础环境：

```bash
uv sync
```

也可以使用 Python 虚拟环境后安装依赖：

```bash
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
.venv\Scripts\Activate.ps1

pip install -r requiremens.txt
```

不同实验还可能需要额外安装 `datasets`、`peft`、`trl`、`vllm`、`deepspeed` 等未统一纳入根目录依赖的包。请以对应子目录的代码和 README 为准，不建议为了运行单个实验安装所有依赖。

## 快速运行示例

以下命令均从仓库根目录执行。训练前请先准备对应的数据集和模型，并检查脚本中的路径配置。

### 从零训练小型 LLM

[`train_llm_from_scratch/README.md`](./train_llm_from_scratch/README.md) 使用了 minimind 数据格式，支持预训练、SFT 和 DPO：

```bash
cd train_llm_from_scratch

# 预训练
python train.py

# 监督微调
python sft_train.py

# 偏好优化
python dpo_train.py
```

多 GPU 训练可以使用：

```bash
torchrun --nproc_per_node=2 train.py
torchrun --nproc_per_node=2 sft_train.py
```

### MoE 训练

```bash
cd train_moe_from_scratch
python moe_train.py
python moe_sft_train.py
python moe_test.py
```

详细的数据准备和 `torchrun`/DeepSpeed 用法见 [`train_moe_from_scratch/README.md`](./train_moe_from_scratch/README.md)。

### 推理引擎教学实现

`inference_engines/` 中的 vLLM/SGLang 示例按文件拆分，每个文件包含独立 demo：

```bash
python inference_engines/vllm/01_PagedAttention.py
python inference_engines/vllm/05_SpeculativeDecoding.py
python inference_engines/sglang/01_RadixAttention.py
```

标记为 Triton 的实现需要 CUDA GPU；不依赖真实模型权重的 mock/demo 可以直接用于理解数据结构、调度和算法流程。完整目录、技术对比和阅读顺序见 [`inference_engines/README.md`](./inference_engines/README.md)。

### Stable Diffusion

Stable Diffusion 子目录基本沿用上游 LDM 工程，拥有独立的 [`environment.yaml`](./stable-diffusion/stable-diffusion/environment.yaml) 和模型权重要求：

```bash
cd stable-diffusion/stable-diffusion
conda env create -f environment.yaml
conda activate ldm
pip install -e .
python scripts/txt2img.py --prompt "一张宇航员骑马的照片" --plms
```

运行前需要按照 [`stable-diffusion/stable-diffusion/README.md`](./stable-diffusion/stable-diffusion/README.md) 准备 checkpoint，并将其放到脚本默认路径或通过 `--ckpt` 指定。

### VoxCPM

VoxCPM 提供 Gradio 页面、Python 推理示例和统一 CLI。模型可从 Hugging Face 加载，也可以通过参数指定本地模型：

```bash
# 直接文本转语音
python -m voxcpm.voxcpm.cli --text "你好，欢迎使用 VoxCPM。" --output output.wav

# 参考音频语音克隆
python -m voxcpm.voxcpm.cli --text "这是一段语音克隆示例。" --prompt-audio voice.wav --prompt-text "参考音频对应的文本。" --output cloned.wav

# 启动 Gradio demo
python voxcpm/app.py
```

模型、ASR 和 ZipEnhancer 权重可能需要联网下载；显存、模型缓存和音频文件要求见 [`voxcpm/README.md`](./voxcpm/README.md) 及对应源码。

## 在线阅读笔记

LLM 和 CV 笔记使用 Docsify 目录结构，可以分别启动：

```bash
npm install -g docsify-cli

docsify serve llm_interview_note
docsify serve cv_interview_note
```

默认浏览器地址为 `http://localhost:3000`。如果端口冲突，可以通过 Docsify 参数指定其他端口，例如：

```bash
docsify serve llm_interview_note --port 3001
```

语音笔记、机器学习、数学和论文目录以普通 Markdown 文件为主，可直接在编辑器中阅读。

## 数据、模型与路径说明

- 仓库中的训练脚本大多是研究和学习用途，数据集与模型权重不随仓库提供。
- 部分脚本仍保留作者本地路径，例如 `/home/user/...`。在运行前需要改成当前机器上的模型、数据和输出目录。
- `train_llm_from_scratch/` 与 `train_moe_from_scratch/` 的 README 参考了 [`minimind`](https://github.com/jingyaogong/minimind) 数据和训练组织方式。
- 多模态实验需要 Qwen、SigLIP 以及图文数据集；具体版本和数据来源见 [`train_multimodal_from_scratch/README.md`](./train_multimodal_from_scratch/README.md)。
- Stable Diffusion 和 Chinese-CLIP 的权重、许可证及安全限制请阅读其各自的模型卡和子项目 README。
- VoxCPM 的音频输出默认按 16 kHz 保存，参考音频和模型下载可能产生较大的磁盘占用。

## 代码与文档约定

- Markdown 笔记以概念推导、公式、论文摘要和面试题为主。
- Python 脚本优先保证单文件可读性，很多实现是教学版或实验性代码，不等同于生产级框架。
- Notebook 用于训练记录、数据处理、验证和可视化；运行前请确认工作目录、模型路径和 GPU 设置。
- `work_projects/` 中的 `DESIGN.md` 是项目设计与里程碑文档，尚不表示对应工程已经全部实现。

## 许可证与致谢

根目录许可证见 [`LICENSE`](./LICENSE)。Stable Diffusion、Chinese-CLIP、VoxCPM 等子项目包含来自上游项目的代码或模型说明，使用时请同时遵守对应子目录中的许可证、模型卡和原项目条款。

感谢所有论文作者、开源项目和数据集维护者。
