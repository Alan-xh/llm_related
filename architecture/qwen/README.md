# Qwen 系列架构

本目录按 Qwen 主要代际拆分，重点记录 decoder-only Transformer、旋转位置编码、长上下文、MoE 和对话推理接口。

这里的代码是**可运行的教学实现**，使用极小配置和无依赖的字节级 tokenizer，目的是把数据流、Tensor Shape、KV cache 和 MoE 路由展示出来；它不兼容官方 checkpoint，也不是官方训练配方的复刻。所有版本共享 [`common.py`](./common.py) 中的基础组件，每个版本目录仍提供独立的 `model.py`、`train.py` 和 `inference.py`。

| 目录 | 主要方向 | 主要创新技术 |
| --- | --- | --- |
| [`Qwen1/`](./Qwen1/) | 初代基础语言模型 | 大规模多语言预训练，使用 ChatML 统一对话和工具调用格式 |
| [`Qwen1.5/`](./Qwen1.5/) | 统一语言模型系列 | dense/MoE 多尺寸覆盖，强化指令对齐、代码和数学能力 |
| [`Qwen2/`](./Qwen2/) | 基础模型与指令模型 | 多数尺寸引入 GQA，增强长上下文、多语言和代码建模 |
| [`Qwen2.5/`](./Qwen2.5/) | 长上下文与代码扩展 | 扩大高质量预训练/后训练数据，提升代码、数学和结构化输出 |
| [`Qwen3/`](./Qwen3/) | 思考/非思考模式与 MoE | 同一模型切换 thinking/non-thinking，并强化 agent 与工具调用 |

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

五个版本的 `train.py` 都默认使用很小的合成语料；替换 `--text` 即可观察 causal LM 的训练接口。
`Qwen1.5` 额外支持 `--moe`，`Qwen3` 默认启用 top-k MoE。

## 官方参考

- [Qwen Technical Report](https://arxiv.org/abs/2309.16609) 与 [Qwen 官方仓库](https://github.com/QwenLM/Qwen)
- [Qwen2 Technical Report](https://arxiv.org/abs/2407.10671) 与 [Qwen2 官方仓库](https://github.com/QwenLM/Qwen2)
- [Qwen2.5 Technical Report](https://arxiv.org/abs/2412.15115) 与 [Qwen2.5 官方仓库](https://github.com/QwenLM/Qwen2.5)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388) 与 [Qwen3 官方仓库](https://github.com/QwenLM/Qwen3)

官方实现使用各自的 tokenizer、配置、训练语料和优化策略；本目录仅抽取最适合教学的结构差异。
