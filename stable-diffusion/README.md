# Stable Diffusion 学习与实验

本目录整理了 Stable Diffusion/LDM 的代码、Chinese-CLIP 文本图像编码器实现，以及模型部署和训练相关实验。内容主要用于源码阅读、模型推理和计算机视觉生成模型学习，不是一个开箱即用的 Web UI。

上游项目：

- [CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
- [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion)
- [OFA-Sys/Chinese-CLIP](https://github.com/OFA-Sys/Chinese-CLIP)

## 目录结构

```text
stable-diffusion/
├── stable-diffusion/       # Stable Diffusion v1 与 LDM 主代码
│   ├── configs/            # 模型配置文件
│   ├── ldm/                # 自动编码器、UNet、扩散采样器等
│   ├── models/             # 配置文件及模型权重存放位置
│   ├── scripts/            # txt2img、img2img、inpaint 等入口
│   ├── assets/             # 示例图片和评估结果
│   ├── environment.yaml    # 上游 Conda 环境
│   └── README.md           # Stable Diffusion v1 详细说明
├── clip-vit/               # Chinese-CLIP 模型与 tokenizer
│   ├── model.py            # 视觉编码器和文本编码器
│   ├── utils.py            # 加载、预处理、tokenize 等工具
│   ├── training/           # 训练与蒸馏脚本
│   ├── deploy/             # ONNX、TensorRT、Core ML 部署脚本
│   └── model_configs/      # ViT/ResNet 模型配置
└── README.md
```

## Stable Diffusion

Stable Diffusion v1 是一种以 CLIP ViT-L/14 文本编码器为条件的潜在扩散模型。图像先由下采样因子为 8 的自动编码器压缩到潜在空间，再由扩散模型在潜在空间中完成去噪，因此相比直接在像素空间扩散更加节省计算资源。

主代码支持以下实验：

- 文本生成图像：`scripts/txt2img.py`
- 文本引导的图像修改：`scripts/img2img.py`
- 图像修复：`scripts/inpaint.py`
- DDIM、PLMS 和 DPM-Solver 采样
- LDM 自动编码器、条件扩散和检索增强扩散实验

更完整的模型说明、检查点列表和论文引用见 [`stable-diffusion/README.md`](./stable-diffusion/README.md)。模型卡中的使用限制和偏差说明见 [`Stable_Diffusion_v1_Model_Card.md`](./stable-diffusion/Stable_Diffusion_v1_Model_Card.md)。

### 环境安装

Stable Diffusion 子项目使用独立的旧版依赖，建议不要直接与仓库根目录的 Python 环境混用：

```bash
cd stable-diffusion

conda env create -f environment.yaml
conda activate ldm
pip install -e .
```

`environment.yaml` 固定了 Python 3.8.5、PyTorch 1.11.0、CUDA Toolkit 11.3 等版本。实际运行时还需要根据本机驱动、CUDA 版本和显卡情况调整环境。

### 模型权重

模型权重不包含在本仓库中。运行 Stable Diffusion v1 推理前，请从 [CompVis Hugging Face 组织](https://huggingface.co/CompVis) 获取相应检查点，并按许可证和模型卡要求使用。

脚本默认读取：

```text
stable-diffusion/models/ldm/stable-diffusion-v1/model.ckpt
```

也可以直接通过 `--ckpt` 指定权重路径：

```powershell
python scripts/txt2img.py `
  --ckpt D:/models/sd-v1-4.ckpt `
  --prompt "a photograph of an astronaut riding a horse on Mars" `
  --plms
```

Linux/macOS 使用反斜杠换行：

```bash
python scripts/txt2img.py \
  --ckpt /path/to/sd-v1-4.ckpt \
  --prompt "a photograph of an astronaut riding a horse on Mars" \
  --plms
```

首次运行还会加载 Stable Diffusion safety checker 和不可见水印相关模型，通常需要联网访问 Hugging Face。

### 文本生成图像

以下命令从 `stable-diffusion/stable-diffusion` 目录执行：

```bash
python scripts/txt2img.py \
  --prompt "a watercolor painting of a mountain lake at sunrise" \
  --plms \
  --n_samples 2 \
  --ddim_steps 50 \
  --seed 42 \
  --outdir outputs/txt2img
```

常用参数：

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--prompt` | 文本提示词 | 内置示例 |
| `--ckpt` | 检查点路径 | `models/ldm/stable-diffusion-v1/model.ckpt` |
| `--config` | 模型配置 | `configs/stable-diffusion/v1-inference.yaml` |
| `--plms` | 使用 PLMS 采样 | 关闭 |
| `--dpm_solver` | 使用 DPM-Solver 采样 | 关闭 |
| `--ddim_steps` | 采样步数 | `50` |
| `--scale` | 无分类器引导尺度 | `7.5` |
| `--H`、`--W` | 输出高度和宽度 | `512` |
| `--seed` | 随机种子 | `42` |
| `--precision` | `full` 或 `autocast` | `autocast` |

运行 `python scripts/txt2img.py --help` 可查看完整参数。

### 图像到图像

`--strength` 控制输入图像被破坏的程度。数值越大，输出越容易偏离输入图像：

```bash
python scripts/img2img.py \
  --init-img assets/stable-samples/img2img/sketch-mountains-input.jpg \
  --prompt "a detailed fantasy landscape" \
  --strength 0.75 \
  --ddim_steps 50 \
  --outdir outputs/img2img
```

### 图像修复

`scripts/inpaint.py` 会读取输入目录中的成对文件：

```text
inputs/inpaint/example.png
inputs/inpaint/example_mask.png
```

其中 mask 中的白色区域会被修复，黑色区域保留原图。运行：

```bash
python scripts/inpaint.py \
  --indir inputs/inpaint \
  --outdir outputs/inpaint \
  --steps 50
```

该脚本默认读取：

```text
models/ldm/inpainting_big/config.yaml
models/ldm/inpainting_big/last.ckpt
```

### LDM 旧实验

以下脚本用于下载和解压原始 LDM 实验权重，属于 Stable Diffusion v1 之外的历史实验：

```bash
# 在 stable-diffusion/stable-diffusion 目录执行
bash scripts/download_first_stages.sh
bash scripts/download_models.sh
```

脚本依赖 `wget` 和 `unzip`，并会将文件写入 `models/`。如果只运行 Stable Diffusion v1 的 txt2img/img2img，一般不需要执行这两个脚本。

## Chinese-CLIP

`clip-vit/` 包含中文文本编码器、图像编码器、tokenizer 和模型加载工具，支持以下模型名称：

```text
ViT-B-16
ViT-L-14
ViT-L-14-336
ViT-H-14
RN50
```

模型权重可以使用 ModelScope 或 Hugging Face 下载。具体下载逻辑位于 [`clip-vit/utils.py`](./clip-vit/utils.py)。

由于目录名 `clip-vit` 含有连字符，不能直接作为普通 Python 包名导入。下面的示例使用 Python 的包加载器加载本地代码：

```python
import importlib.util
import pathlib
import sys

root = pathlib.Path("clip-vit").resolve()
spec = importlib.util.spec_from_file_location(
    "clip_vit",
    root / "__init__.py",
    submodule_search_locations=[str(root)],
)
clip_vit = importlib.util.module_from_spec(spec)
sys.modules["clip_vit"] = clip_vit
spec.loader.exec_module(clip_vit)

print(clip_vit.available_models())
```

加载模型并计算图文相似度：

```python
import torch
from PIL import Image

model, preprocess = clip_vit.load_from_name(
    "ViT-B-16",
    device="cuda" if torch.cuda.is_available() else "cpu",
    download_root="./checkpoints",
    use_modelscope=True,
)
model.eval()

image = preprocess(Image.open("path/to/image.jpg")).unsqueeze(0)
text = clip_vit.tokenize(["一只猫", "一只狗"])

device = next(model.parameters()).device
image = image.to(device)
text = text.to(device)

with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(text)
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    similarity = image_features @ text_features.T

print(similarity)
```

训练、蒸馏和部署相关入口分别位于：

- [`clip-vit/training/`](./clip-vit/training/)
- [`clip-vit/deploy/`](./clip-vit/deploy/)
- [`clip-vit/flash_atten.md`](./clip-vit/flash_atten.md)

这些实验可能需要额外安装 `modelscope`、`huggingface_hub`、ONNX、TensorRT、Core ML 或 FlashAttention，请以对应脚本的导入和 README 为准。

## 2D 卷积输入输出形状

符号定义：

| 符号 | 描述 |
| :--- | :--- |
| **I** | 输入特征图的边长，输入形状为 `I × I` |
| **K** | 卷积核的边长，卷积核形状为 `K × K` |
| **P** | 每一侧添加的 padding 像素数 |
| **S** | stride，卷积核移动的步长 |
| **O** | 输出特征图的边长，输出形状为 `O × O` |

输出边长计算公式：

```text
O = floor((I - K + 2P) / S) + 1
```

当输入不是正方形时，分别对高度和宽度使用同一公式计算。

## 许可证与安全说明

- Stable Diffusion 代码和权重的许可证并不等同，请同时阅读 [`stable-diffusion/LICENSE`](./stable-diffusion/LICENSE)、模型卡和权重发布页面。
- Stable Diffusion v1 是研究模型，可能生成有害、偏见或不准确的内容，不应在没有额外安全机制和人工审核的情况下直接用于面向用户的产品。
- 生成涉及人物、版权材料、虚假信息或敏感内容时，应确认拥有必要的授权，并遵守适用法律和平台规则。
- Chinese-CLIP 代码和模型权重也可能有各自的许可条件，使用前请核对上游项目说明。

## 参考资料

- [High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)
- [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502)
- [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)