# 项目 01: mini 推理引擎 (mini-infer)

## 一句话目标

从零实现一个生产可用的 LLM 推理引擎，支持 PagedAttention + Continuous Batching + INT8 量化，在 Llama/Qwen 系列模型上吞吐达到 vLLM 的 50%+，P99 延迟不超过 vLLM 的 1.5 倍。

## 为什么这个项目难

主流教程讲 PagedAttention 都停在"分块管理 KV cache"概念层，真要落地需要解决一堆工程问题：

- KV cache 的物理块映射、引用计数、copy-on-write（beam search 共享前缀）
- 调度器要在 token 级别做抢占恢复（高优请求来了把低优请求 swap 出去）
- Triton kernel 写错了不会报错，只是结果悄悄偏差 0.3%，定位极困难
- Continuous Batching 下 prefill 和 decode 阶段的 batch shape 完全不同，混合调度很容易写出维度对不上的 bug

vLLM 源码有 5w+ 行，能看懂但完全重写一个 mini 版仍有大量坑。

## 核心难点分解

| 编号 | 难点 | 难度 | 备注 |
| --- | --- | --- | --- |
| D1 | PagedAttention block-level KV 管理 | ★★★★ | 物理/逻辑 block 映射 + 引用计数 + COW |
| D2 | Continuous Batching 调度器 | ★★★★ | 请求优先级、抢占、slot 分配、prefill/decode 混合 |
| D3 | Triton attention kernel | ★★★★ | 替代 CUDA，但要懂 GPU 编程模型和 shared memory |
| D4 | INT8 weight-only 量化 + 反量化 kernel | ★★★ | 精度/速度权衡，需要校准 |
| D5 | 投机解码（draft + target + verify） | ★★★★ | 选做，难度高但加分 |
| D6 | Benchmark 体系搭建 | ★★ | 不能只看 throughput，要分 prefill/decode/混合 |

## 可行性论证

- **Triton 远比 CUDA 友好**：OpenAI Triton 用 Python 写 GPU kernel，3 周可入门，能写出 attention kernel
- **不需要训练**：纯推理工程，4-bit/8-bit 量化校准用现成模型
- **可参考但不抄**：vLLM、SGLang、TGI 都开源，可对照实现思路
- **硬件门槛低**：单卡 A10/A100 即可开发，80GB 显存能跑 7B 模型 INT8
- **已有基础**：`handwrite_network/gpt2.py` 和 `inference_engines/` 笔记铺垫了必要知识

## 里程碑路线图

### M0: Naive 推理 (无 KV cache)
- 用 PyTorch 实现 Llama 前向，每步重算 KV
- **DoD**: 能正确生成 100 token，与 HuggingFace 输出一致 (logits 最大误差 < 1e-5)

### M1: KV cache + naive batching
- 实现 per-request KV cache，static batching
- **DoD**: 同 batch 4 个请求，吞吐比 M0 提升 3x+

### M2: PagedAttention (Triton)
- 实现 block-level KV cache 管理（block_size=16）
- 用 Triton 写 paged attention kernel
- **DoD**: KV cache 显存利用率 > 90%，长序列吞吐比 M1 提升 2x+

### M3: Continuous Batching 调度器
- 实现 prefill/decode 混合调度 + 抢占恢复
- 支持 max_num_seqs、max_num_batched_tokens 两个核心参数
- **DoD**: 在 32 个并发请求下吞吐比 M2 提升 2x+，无 OOM

### M4: INT8 weight-only 量化
- 实现 GPTQ/AWQ 风格的 weight 量化加载
- 写反量化 Triton kernel (INT8 -> bf16)
- **DoD**: 7B 模型显存 < 8GB，吞吐损失 < 15%

### M5: 投机解码 (选做)
- 集成小模型作为 draft
- 实现 tree attention verify
- **DoD**: 接受率 > 60% 时端到端吞吐提升 1.5x+

### M6: Benchmark + 文档
- 对比 vLLM 在 Llama-3-8B / Qwen2.5-7B 上的吞吐
- 写完整文档 + 性能分析报告
- **DoD**: 吞吐达到 vLLM 的 50%+，文档可对外发布

## 评估指标

| 指标 | 目标 | 测量方式 |
| --- | --- | --- |
| Throughput (decode, tokens/s) | >= vLLM × 50% | ShareGPT 数据集，并发 32 |
| Throughput (prefill, tokens/s) | >= vLLM × 40% | 长 prompt 单请求 |
| Latency P99 | <= vLLM × 1.5 | 同上 |
| 显存利用率 | > 90% | 实际 KV / 总 KV 显存 |
| 支持模型 | Llama/Qwen/Mistral 系列 | 至少 3 个 |

## 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| Triton kernel 调试困难 | 先用 PyTorch 实现 reference，再迁移 Triton，逐步对照 |
| 调度器状态管理复杂 | 用有限状态机建模请求生命周期，先纸面设计再编码 |
| 性能不达标 | M2 起每个里程碑都 benchmark，不达标不进入下一阶段 |
| 时间超期 | M5 投机解码设为选做，M0-M4 是核心交付 |

## 技术栈

- Python 3.10+ / PyTorch 2.x
- OpenAI Triton (kernel)
- Numba (CPU 端调度逻辑可选)
- Pytest (正确性测试)
- locust / asyncio (并发压测)

## 参考资料

- vLLM: Efficient Memory Management for LLM Serving (SOSP 2023)
- Triton: an intermediate language and compiler for tiled neural network computations
- Orca: A Distributed Serving System for Transformer-Based Generative Models (OSDI 2022)
- Continuous Batching (AnyScale blog)
- GPTQ: Accurate Post-Training Quantization (ICLR 2023)
- Speculative Decoding (Leviathan et al., ICML 2023)

## 目录结构（建议）

```
01_mini_inference_engine/
├── DESIGN.md
├── journal.md
├── mini_infer/
│   ├── core/            # 调度器、KV manager、tokenizer
│   ├── models/          # 模型实现（Llama、Qwen）
│   ├── kernels/         # Triton kernels
│   ├── quantization/    # INT8/FP8
│   └── engine.py        # 入口
├── benchmarks/
├── tests/
└── pyproject.toml
```
