# 项目级实战目录

> 与 `*_from_scratch/` 学习型练习不同，本目录下的项目面向**完整可交付产物**：要求难度高于一般面试项目，但单人 2-3 月内可实现。

## 项目清单

| 编号 | 项目 | 一句话目标 | 核心难度 | 对应已学知识 |
| --- | --- | --- | --- | --- |
| 01 | [mini 推理引擎](./01_mini_inference_engine/DESIGN.md) | 从零实现支持 PagedAttention + Continuous Batching + INT8 量化的 LLM 推理引擎，吞吐达到 vLLM 的 50%+ | GPU 内存管理 + Triton kernel + 调度器 | `inference_engines/`、`handwrite_network/`、`06.推理` |
| 02 | [MoE 双语模型全流程对齐](./02_moe_bilingual_llm/DESIGN.md) | 从零训练 500M 总参/100M 激活的 MoE 中英双语模型，跑通 PT→SFT→DPO 全流程，C-Eval 接近 Qwen2.5-0.5B | MoE 路由稳定性 + 训练稳定 + 评估 | `train_llm_from_scratch/`、`train_moe_from_scratch/`、`04.分布式训练` |
| 03 | [多模态 Agent 系统](./03_multimodal_agent/DESIGN.md) | 端到端多模态 Agent：图文混合 RAG + 长上下文压缩 + 多步工具调用 + 自我反思 | 系统集成 + 评测体系 + 工具可靠性 | `train_siglip_from_scratch/`、`train_multimodal_from_scratch/`、`08.RAG与Agent` |

## 选题原则

1. **可实现**：单人在已有知识基础上 2-3 月内做出可演示的 MVP，不依赖未学领域
2. **高门槛**：每个项目都有 3-5 个非平凡技术难点，不是教程拼接
3. **可评估**：有量化指标（throughput / accuracy / 任务完成率），不是"做个 demo"
4. **可延伸**：MVP 完成后仍有清晰的扩展空间，能作为简历主线项目持续打磨

## 共同纪律

- 每个 `DESIGN.md` 必须包含：目标、难点分解、可行性论证、里程碑路线图、评估指标、风险与缓解、参考资料
- 实现阶段以 `M0 → M6` 里程碑推进，每个里程碑有明确的"完成定义"（Definition of Done）
- 每个项目根目录放 `journal.md` 记录关键决策与踩坑，作为简历素材
