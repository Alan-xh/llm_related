# LLM 推理引擎核心技术实现

vLLM 与 SGLang 两大推理引擎所有核心技术的教学版实现,每个文件独立可读,带详细中文注释。

## 目录结构

```
inference_engines/
├── README.md                  ← 本文件
├── vllm/                      ← vLLM 13 项核心技术
│   ├── 01_PagedAttention.py        [Triton] 分页注意力 + block table 寻址
│   ├── 02_ContinuousBatching.py    连续批处理 / iteration-level scheduling
│   ├── 03_ChunkedPrefill.py        长 prompt 切块与 decode 混合调度
│   ├── 04_PrefixCaching.py         基于 block hash 的自动前缀缓存
│   ├── 05_SpeculativeDecoding.py   Draft model + 拒绝采样验证
│   ├── 06_TensorParallelism.py     Megatron 风格 column/row parallel
│   ├── 07_Quantization.py          GPTQ / AWQ / FP8 / SmoothQuant
│   ├── 08_LoRA.py                  Multi-LoRA + BGMN-style 批量计算
│   ├── 09_StructuredOutput.py      Choice / Regex / JSON Schema 约束
│   ├── 10_CUDAGraph.py             CUDA Graph 捕获 + 多 bucket
│   ├── 11_AsyncOutputProcessor.py  GPU forward 与 CPU 后处理重叠
│   ├── 12_MultiModal.py            LLaVA 风格 Vision Tower + Projector
│   └── 13_MultiStepScheduler.py    K 步一次调度,减少 CPU 同步
│
└── sglang/                    ← SGLang 14 项核心技术
    ├── 01_RadixAttention.py        [Triton] 基数树 KV cache + 显式前缀匹配
    ├── 02_CacheAwareScheduler.py   按 prefix hit 长度调度 + 饥饿防护
    ├── 03_OverlapScheduler.py      CPU 调度与 GPU forward 双缓冲重叠
    ├── 04_FrontendDSL.py           sgl.function / gen / select / fork
    ├── 05_ConstrainedDecoding.py   xgrammar 风格 Regex/JSON/CFG + 组合
    ├── 06_EAGLE.py                 EAGLE-2 投机解码 + draft tree + tree attention
    ├── 07_MLA.py                   [Triton] DeepSeek MLA + weight absorption
    ├── 08_MTP.py                   Multi-Token Prediction 作为 specDec proposer
    ├── 09_FP8Kernel.py             [Triton] FP8 E4M3 GEMM + per-tensor/channel/block
    ├── 10_TritonAttention.py       [Triton] FlashAttention v2 + paged decode + GQA
    ├── 11_FlashInferIntegration.py FlashInfer batch prefill/decode 后端
    ├── 12_MultiLoRA.py             S-LoRA grouped GEMM
    ├── 13_DataParallelRouter.py    DP 缓存感知路由器
    └── 14_MooncakeKVTransfer.py    RDMA KV 传输 + P/D 分离
```

标记 `[Triton]` 的文件包含 Triton GPU kernel 实现。

## 核心技术对比

| 维度 | vLLM | SGLang |
|------|------|--------|
| **KV 管理** | PagedAttention(block hash) | RadixAttention(基数树) |
| **调度** | Continuous batching + chunked prefill | Cache-aware + overlap scheduler |
| **CPU-GPU 重叠** | Multi-step + async output | 默认 overlap scheduler |
| **结构化输出** | outlines / xgrammar / llguidance | xgrammar + composable CFG |
| **Speculative** | Draft / Medusa / Lookahead | EAGLE / MTP |
| **DeepSeek 优化** | 通用 MLA 支持 | weight absorption + FP8 + MTP |
| **并行** | TP / PP / DP / EP | TP / DP(cache-aware router) |
| **量化** | GPTQ / AWQ / FP8 / bnb | FP8(W8A8)为主 |
| **编程模型** | OpenAI-style API + LLM class | DSL(@sgl.function) |

## 运行方式

每个文件都是独立的教学程序,带 `demo()` 函数,直接运行即可:

```bash
# 单个文件
python inference_engines/vllm/01_PagedAttention.py

# 批量运行(以 vllm 为例)
for f in inference_engines/vllm/*.py; do
    echo "=== $f ==="
    python "$f"
done
```

## 依赖

```
torch >= 2.1
triton >= 2.1
numpy
```

- 含 `[Triton]` 的文件需要 CUDA GPU 才能运行 kernel 部分,但 Python 层数据结构可在 CPU 上跑。
- 模拟场景(async、benchmark)用 mock,无真实 LLM 依赖。

## 阅读建议

**入门路径**(按推荐顺序):

1. `vllm/01_PagedAttention.py` — 理解 KV cache 与 block 抽象
2. `sglang/01_RadixAttention.py` — 对比 PagedAttention 的树结构方案
3. `vllm/02_ContinuousBatching.py` — 理解 iteration-level 调度
4. `vllm/03_ChunkedPrefill.py` — 长 prompt 处理
5. `sglang/03_OverlapScheduler.py` — CPU/GPU 重叠

**进阶路径**:

6. `vllm/05_SpeculativeDecoding.py` — 经典 specDec 算法
7. `sglang/06_EAGLE.py` — EAGLE tree attention
8. `sglang/07_MLA.py` — DeepSeek MLA + weight absorption(关键)
9. `vllm/06_TensorParallelism.py` — Megatron 风格 TP
10. `sglang/13_DataParallelRouter.py` — DP + 缓存感知路由

**专项路径**:

- 量化:`vllm/07_Quantization.py` + `sglang/09_FP8Kernel.py`
- LoRA:`vllm/08_LoRA.py` + `sglang/12_MultiLoRA.py`
- 结构化输出:`vllm/09_StructuredOutput.py` + `sglang/05_ConstrainedDecoding.py`
- 多模态:`vllm/12_MultiModal.py`
- 性能优化:`vllm/10_CUDAGraph.py` + `vllm/11_AsyncOutputProcessor.py` + `vllm/13_MultiStepScheduler.py`

## 实现说明

- **教学精简版**:每个文件 100-300 行,独立可读,带详细中文注释
- **可运行**:用 mock 模型/数据演示算法逻辑,不依赖真实 LLM 权重
- **忠实于原实现**:算法步骤与 vLLM/SGLang 源码一致,简化的是工程细节
- **关键 kernel 用 Triton**:PagedAttention、RadixAttention、MLA、FP8 GEMM、FlashAttention 等核心创新点都提供了 Triton 实现

## 选型建议

| 场景 | 推荐 |
|------|------|
| 通用 OpenAI 替代品,生态广 | vLLM |
| 大规模生产,需要 PP / 多硬件后端 | vLLM |
| DeepSeek-V3 / R1 推理 | SGLang |
| 大量多轮对话 / agent(强前缀复用) | SGLang |
| 严格 JSON / 结构化输出 | SGLang |
| 投机解码 + 高 acceptance | SGLang(EAGLE) |
| 简单部署 + LoRA 多 adapter | vLLM |
| 学习/研究 KV 管理 | 两者都看 |
