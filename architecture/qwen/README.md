# Qwen 系列架构

本目录覆盖 Qwen 语言模型、视觉语言模型（VL）和全模态模型（Omni），并将架构机制与实际模型使用分开维护。

Qwen1 到 Qwen3 的 `model.py`/`train.py` 是机制验证用的 tiny 实现，使用字节级 tokenizer，不兼容官方 checkpoint，也不是官方训练配方复刻。VL/Omni 目录则以官方开源 checkpoint 为目标，整理 Transformers 推理、服务部署、微调和数据处理等实际工作流；多模态模型依赖官方 processor、视觉/音频预处理和对应运行时，不用 tiny 模型伪装成可部署实现。

| 目录 | 主要方向 | 主要创新技术 |
| --- | --- | --- |
| [`Qwen1/`](./Qwen1/) | 初代基础语言模型 | 大规模多语言预训练，使用 ChatML 统一对话和工具调用格式 |
| [`Qwen1.5/`](./Qwen1.5/) | 统一语言模型系列 | dense/MoE 多尺寸覆盖，强化指令对齐、代码和数学能力 |
| [`Qwen2/`](./Qwen2/) | 基础模型与指令模型 | 多数尺寸引入 GQA，增强长上下文、多语言和代码建模 |
| [`Qwen2.5/`](./Qwen2.5/) | 长上下文与代码扩展 | 扩大高质量预训练/后训练数据，提升代码、数学和结构化输出 |
| [`Qwen3/`](./Qwen3/) | 思考/非思考模式与 MoE | 同一模型切换 thinking/non-thinking，并强化 agent 与工具调用 |
| [`Qwen2.5-VL/`](./Qwen2.5-VL/) | 图像与视频理解 | 动态分辨率、视觉 grounding、视频时序建模；官方 processor 推理与部署 |
| [`Qwen3-VL/`](./Qwen3-VL/) | 新一代视觉语言模型 | Interleaved-MRoPE、DeepStack、多尺度图像/视频理解；Transformers/vLLM |
| [`Qwen2.5-Omni/`](./Qwen2.5-Omni/) | 端到端多模态交互 | Thinker-Talker、音视频联合感知、文本与语音生成 |
| [`Qwen3-Omni/`](./Qwen3-Omni/) | 新一代全模态交互 | MoE Thinker-Talker、音视频输入、流式语音/文本输出 |

## 实际模型工作流

多模态线上/离线推理应从模型仓库自带的 `Processor`、chat template 和媒体预处理器开始；
不要手工拼 `<image>` 等 token，也不要把媒体数据直接当普通文本 tokenizer 的输入。
[`production/README.md`](./production/README.md) 对比 Transformers、vLLM、SGLang、
ms-swift、LLaMA-Factory 等开源框架，并给出数据、微调、量化、服务和评测的落地检查表。
各模型目录记录其特有的处理流程、运行命令和限制。

VL/Omni 的 `model.py` 是实际 checkpoint 的 Transformers 加载/推理封装，
`inference.py` 可从本地图片、视频、音频路径运行；它们依赖官方模型权重和多模态运行环境，
不会在无权重时退化为随机初始化的小模型。

## 统一数据流

```text
text
  -> ByteTokenizer                         [B, T] token ids
  -> token embedding                       [B, T, H]
  -> N x (RMSNorm -> RoPE attention -> residual
          -> RMSNorm -> SwiGLU/MoE -> residual)
  -> final RMSNorm                         [B, T, H]
  -> lm_head                               [B, T, V]
  -> shifted cross entropy                 scalar loss
```

训练时令 `x = tokens[:, :-1]`、`y = tokens[:, 1:]`，目标函数为：

```text
L_lm = - sum_t log p(y_t | x_<t>) / T
L_total = L_lm + 0.01 * L_aux       # 仅 MoE 版本启用 L_aux
```

注意力的基本 Shape 为 `Q: [B, n_heads, Tq, d]`、`K/V: [B, n_kv_heads, Tk, d]`、输出
`[B, Tq, H]`。当 `n_kv_heads < n_heads` 时，KV head 会按
`n_heads / n_kv_heads` 复制为 GQA。推理首轮缓存每层的 `K/V`，后续只输入最后一个 token，
将复杂度从重复计算整段前缀改为追加一个 query。

## 快速运行

项目依赖中需要 `torch`。在仓库根目录执行：

```bash
python architecture/qwen/Qwen1/model.py
python architecture/qwen/Qwen2/inference.py --prompt "Qwen2 is" --max-new-tokens 16
python architecture/qwen/Qwen3/inference.py --thinking --prompt "Why use GQA?"
python architecture/qwen/Qwen3/train.py --steps 5 --checkpoint qwen3_tiny.pt
python architecture/qwen/Qwen3/inference.py --checkpoint qwen3_tiny.pt
```

语言模型版本的 `train.py` 都默认使用很小的合成语料；替换 `--text` 即可观察 causal LM 的训练接口。
`Qwen1.5` 额外支持 `--moe`，`Qwen3` 默认启用 top-k MoE。

## 官方参考

- [Qwen Technical Report](https://arxiv.org/abs/2309.16609) 与 [Qwen 官方仓库](https://github.com/QwenLM/Qwen)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) 与 [Qwen2 官方仓库](https://github.com/QwenLM/Qwen2)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) 与 [Qwen2.5 官方仓库](https://github.com/QwenLM/Qwen2.5)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) 与 [Qwen3 官方仓库](https://github.com/QwenLM/Qwen3)
- [Qwen2.5-VL 官方仓库](https://github.com/QwenLM/Qwen2.5-VL)、[Qwen3-VL 官方仓库](https://github.com/QwenLM/Qwen3-VL)
- [Qwen2.5-Omni 官方仓库](https://github.com/QwenLM/Qwen2.5-Omni)、[Qwen3-Omni 官方仓库](https://github.com/QwenLM/Qwen3-Omni)

官方实现使用各自的 tokenizer、配置、训练语料和优化策略。tiny 目录聚焦可读的机制实现；
VL/Omni 与 production 指南聚焦官方模型及开源运行框架的工程接入，命令和 API 可能随上游版本变化。
