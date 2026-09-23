# 开源多模态模型工程实践

本页面向实际 checkpoint 的推理、微调和部署。模型能力、显存占用和 API 由模型权重、
processor、Transformers/vLLM 版本及媒体预处理配置共同决定；上线前应锁定模型 revision
与依赖版本，并用真实业务样本回归。

## 框架选择

| 工作 | 优先评估 | 适用边界 |
| --- | --- | --- |
| 单机 Python 推理、快速验证 | Hugging Face Transformers | 最接近模型官方示例，通常覆盖模型完整能力；多模态预处理按模型仓库安装 |
| 高吞吐在线/离线生成 | vLLM | 连续批处理、张量并行和 OpenAI-compatible serving；各模型/模态支持范围不同，先核对模型矩阵 |
| 另一种高吞吐推理引擎 | SGLang | 支持部分 VL 模型与结构化多模态输入；部署前核对具体架构、视频和音频功能 |
| LoRA/QLoRA SFT、评测、量化、部署 | ms-swift | 对 Qwen、InternVL、LLaVA、MiniCPM、GLM-V 等模型族有较广覆盖；具体训练目标仍有模型限制 |
| 可视化微调与配方化训练 | LLaMA-Factory | 支持常见 VLM SFT/LoRA 工作流；特殊 Omni 输入输出与新架构需核实支持状态 |

上游入口：

- [Transformers 多模态模型文档](https://huggingface.co/docs/transformers/index)
- [vLLM supported models](https://docs.vllm.ai/en/latest/models/supported_models.html)；[SGLang 文档](https://docs.sglang.ai/)
- [ms-swift 文档](https://swift.readthedocs.io/en/latest/)；[LLaMA-Factory 文档](https://llamafactory.readthedocs.io/en/latest/)
- [Qwen 官方代码与 cookbooks](https://github.com/QwenLM)

选择规则：

- 先用 Transformers 确认 checkpoint、预处理和任务定义正确，再考虑推理引擎优化。
- 推理服务先验证 OpenAI-compatible API 的图像/视频/音频 payload 约定；文本 API 能通不代表媒体输入链路可用。
- 微调优先从 LoRA/QLoRA 和冻结视觉塔开始。确认数据质量、验证集与线上收益后，再扩展可训练模块。
- Omni 的音频生成、Thinker/Talker、多流对齐与训练支持不是一般 VLM SFT 的自然延伸；逐项确认框架是否支持目标输入/输出模态。

## 建议的项目结构

```text
multimodal_app/
  configs/
    model.yaml             # model id/revision, dtype, device map, pixel/frame limits
  data/
    train.jsonl
    val.jsonl
  src/
    load_model.py          # runtime-specific loader
    preprocess.py          # media validation, resize/sample, duration limits
    inference.py           # task interface and output normalization
    serving.py             # API/server adapter
    evaluate.py            # task metrics and regression set
  outputs/
```

将业务接口与引擎适配分开：统一输入可以包含 `messages`、媒体引用和 generation config；
适配层负责转换为 Transformers processor 或推理引擎 payload。输出规范化为文本、时间戳/
结构化结果和可选音频文件，避免业务代码依赖模型专用 token。

## 数据与预处理

- JSONL 每行一个样本，保留多轮 `messages`；媒体引用单独存路径或 URI，避免把大文件编码进 JSON。
- 记录来源、许可、去重键、语言、任务标签以及损坏/缺失媒体状态；拆分训练/验证集时避免同源泄漏。
- 图片设置像素上限并记录 resize/crop 策略。动态分辨率会改变视觉 token 数和显存/延迟。
- 视频设置采样 FPS、最大帧数、最大时长和总像素预算；对齐时间戳，避免默认解码完整长视频。
- 音频明确采样率、声道、时长上限和是否使用视频中的音轨；检查静音、削波和转码错误。
- 缓存预处理结果时将 processor/tokenizer revision 和预处理参数纳入 cache key。
- 混合模态 SFT 样本要检查模板渲染后的媒体占位与文件数一一对应；只检查 JSON 可解析不足以证明样本有效。

通用对话样本示例（媒体路径字段和数据集加载方式按所选框架适配）：

```json
{"messages":[{"role":"user","content":"请概括视频中的操作步骤。"},{"role":"assistant","content":"先打开包装，再取出设备并连接电源。"}],"videos":["/data/clip_001.mp4"]}
```

## 微调起点

ms-swift 提供面向多个多模态模型的统一 SFT 接口。以下是命令形态示例；
先根据目标模型的官方最佳实践调整图像 token、视频帧数、序列长度、目标模块和分布式配置：

```bash
swift sft \
  --model Qwen/Qwen3-VL-8B-Instruct \
  --dataset /data/train.jsonl \
  --val_dataset /data/val.jsonl \
  --tuner_type lora \
  --torch_dtype bfloat16 \
  --max_length 8192 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8 \
  --learning_rate 1e-4 \
  --output_dir output/qwen3-vl-lora
```

此处参数不是通用最优配方。至少需要核验：媒体字段格式、chat template、LoRA target modules、
视觉塔是否训练、分辨率/视频帧预算、有效 batch size、梯度检查点、评测解码配置。
Qwen2.5-Omni 的 ms-swift 支持范围包含 Thinker 训练但不包括 Talker 训练；不要据此推断
端到端语音生成微调已被覆盖。Qwen3-Omni 的支持状态应以所用 ms-swift 版本文档为准。

## 服务与运行检查

对 vLLM 执行 `vllm serve <model-id>` 前，查阅当前版本模型支持列表及模型官方部署说明。
部署配置通常需要明确 `--dtype`、张量并行数、最大上下文、媒体大小/帧数、允许的本地媒体目录、
API 端口和并发限制。Omni 的多模态输出与流式接口可能与标准 chat completion 有差异。

上线前至少跑完：

- 代表性图片、长宽比极端图片、损坏文件和超大媒体的输入验证。
- 视频抽帧数量、采样时间戳和输入 token/显存上限检查。
- 音频采样率、端到端延迟、首 token/首音频延迟及分块连续性检查。
- 多轮对话、停止条件、超时、取消请求和并发压力测试。
- 任务指标、格式正确率、拒答/幻觉、延迟、峰值显存与成本的版本回归。

## 可扩展的开源模型族

| 模型族 | 典型任务 | 实际接入提示 |
| --- | --- | --- |
| Qwen2.5-VL / Qwen3-VL | OCR、文档理解、视觉问答、grounding、视频理解 | 使用官方 processor 和视觉工具库；按像素与帧预算控成本 |
| Qwen2.5-Omni / Qwen3-Omni | 音频/视频/图像综合感知，文本或语音回答 | 需要 Omni 专用预处理；确认实际运行后端是否支持所需输出模态 |
| InternVL3.5 | 通用视觉问答、文档和高分辨率图像 | 官方提供 GitHub 与 HF 权重格式；格式和推理代码需配套使用 |
| MiniCPM-o | 端侧/较低资源多模态交互 | 核实对应版本支持的视觉、音频、视频输入输出以及设备量化组合 |
| GLM-4.5V、LLaVA、DeepSeek-VL2、Ovis | 视觉问答、图文理解、特定视觉任务 | 用 ms-swift 等框架的模型支持矩阵筛选，再检查上游模型原生推理说明 |

模型仓库示例：[InternVL](https://github.com/OpenGVLab/InternVL)、
[MiniCPM-o](https://github.com/OpenBMB/MiniCPM-o)、
[GLM-V](https://github.com/zai-org/GLM-V)、
[LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)。

更多模型接入时，优先新增模型配置、处理器适配、任务样例和回归测试；只有模型本身引入新的
数据流/结构机制时，才在架构层新增自定义 PyTorch 网络。
