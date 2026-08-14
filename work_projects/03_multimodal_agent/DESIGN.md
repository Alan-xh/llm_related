# 项目 03: 多模态 Agent 系统 (mm-agent)

## 一句话目标

构建一个端到端多模态 Agent 系统，支持图文混合 RAG、长上下文压缩、多步工具调用与自我反思，在自建的多步推理任务集上任务完成率 >= 70%，工具调用准确率 >= 85%。

## 为什么这个项目难

市面上"Agent"项目大多是 LangChain 套壳，真正难的点都被掩盖了：

- **图文混合检索**：图片和文本的 embedding 在不同空间，融合检索需要解决相似度归一化、跨模态对齐
- **长上下文压缩**：LLMLingua 风格压缩对纯文本有效，多模态 token（图像 patch）压缩尚未有成熟方案
- **多步规划可靠性**：ReAct 在 5 步以上推理时错误率指数级累积，需要自我反思 + 回溯
- **工具调用鲁棒性**：function calling 生成的 JSON 经常 schema 错误，需要 schema 校验 + 重试 + fallback
- **端到端评估**：多步 Agent 没有标准 benchmark，需要自建评测集并定义"完成"标准

## 核心难点分解

| 编号 | 难点 | 难度 | 备注 |
| --- | --- | --- | --- |
| D1 | 图文混合 embedding 检索 | ★★★★ | SigLIP 视觉 + 文本 encoder 融合 + 跨模态对齐 |
| D2 | 长上下文压缩（含多模态 token） | ★★★★ | LLMLingua 扩展 + 图像 patch 压缩策略 |
| D3 | Agent 多步规划 + 自我反思 | ★★★★ | ReAct + Tree of Thoughts + 错误回溯 |
| D4 | 工具调用 schema 生成与校验 | ★★★ | JSON schema 约束生成 + 重试机制 |
| D5 | 多模态记忆系统 | ★★★ | 短期上下文 + 长期向量库 + 图文关联 |
| D6 | 端到端评测集搭建 | ★★★★ | 自建多步任务集 + 完成度评分标准 |

## 可行性论证

- **已有基础**：`train_siglip_from_scratch/` + `train_multimodal_from_scratch/` 已跑通 VLM 训练
- **后端可选**：Qwen2.5-VL / InternVL2 / 自训 VLM 都可作 backbone
- **数据可得**：图文对（LAION 子集）、工具 API（公开 function calling 数据集）、RAG 评测（MMLU、MMMU 子集）
- **工程量集中**：核心难点在系统集成与评测，不在底层训练
- **单人 2 月可出 MVP**：M0-M3 是核心，M4-M5 是扩展

## 里程碑路线图

### M0: 多模态检索模块
- 实现 SigLIP 图像 encoder + BGE 文本 encoder
- 融合检索（晚期融合 + 早期融合两种策略对比）
- 支持图文混合文档入库
- **DoD**: 在自建 10K 图文对数据集上 recall@10 >= 80%

### M1: 长上下文压缩模块
- 实现 LLMLingua 风格文本压缩
- 扩展到图像 patch token 压缩（基于 attention score 丢弃低重要 patch）
- 压缩比 5x 时下游任务准确率损失 < 5%
- **DoD**: 在长文档 QA 任务上压缩比 5x，F1 下降 < 5%

### M2: Agent 规划器
- 实现 ReAct（Thought-Action-Observation 循环）
- 实现 Tree of Thoughts（多路径探索 + 评分剪枝）
- 自我反思：失败后总结原因并重试
- **DoD**: 在 5 步推理任务上完成率 >= 60%

### M3: 工具调用框架
- 工具 schema 定义（Python 函数签名 -> JSON schema）
- constrained decoding 保证 JSON 合法（参考 outlines / guidance）
- schema 校验 + 自动重试 + fallback
- 内置工具集：搜索、计算器、代码执行、图像生成、数据库查询
- **DoD**: 工具调用 schema 合法率 99%+，语义准确率 85%+

### M4: 多模态记忆系统
- 短期：当前对话上下文（带压缩）
- 长期：向量库 + 图文关联图
- 检索策略：时间衰减 + 重要性评分
- **DoD**: 长对话（50+ 轮）下事实一致性 >= 80%

### M5: 端到端评测集搭建
- 自建多步任务集（100+ 任务，覆盖数学/代码/检索/视觉）
- 任务完成度评分（0/0.5/1 三档）
- 自动化评测 + 人工抽检
- **DoD**: 任务完成率 >= 70%，工具调用准确率 >= 85%

## 评估指标

| 指标 | 目标 | 测量方式 |
| --- | --- | --- |
| 检索 recall@10 | >= 80% | 自建 10K 图文对 |
| 压缩 5x 后 QA F1 | >= baseline × 95% | 长文档 QA |
| 5 步任务完成率 | >= 60% (M2) / 70% (M5) | 自建任务集 |
| 工具调用准确率 | >= 85% | schema 合法 + 语义正确 |
| 长对话事实一致性 | >= 80% | 50+ 轮对话抽检 |
| 端到端 latency | <= 15s/step | 单步推理 |

## 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| VLM backbone 能力不足 | 优先用 Qwen2.5-VL-7B，自训 VLM 作为对照 |
| 工具调用 JSON 不稳定 | 用 outlines / guidance 做 constrained decoding，避免纯 prompt |
| 评测集主观偏差 | 评分规则公开 + 多人交叉标注 + Cohen kappa 一致性检查 |
| 多步推理错误累积 | Tree of Thoughts 多路径 + 自我反思 + 最大步数限制 |
| 系统复杂度高难调试 | 每个模块独立 unit test + 集成 test + trace 日志 |

## 技术栈

- Python 3.10+ / PyTorch 2.x
- transformers / vllm (推理后端)
- SigLIP / BGE (embedding)
- FAISS / Milvus (向量数据库)
- outlines / guidance (constrained decoding)
- langgraph (Agent 状态机，可选)
- pytest / 评测脚本

## 参考资料

- LLMLingua: Compressing Prompts for Accelerated Inference (EMNLP 2024)
- ReAct: Synergizing Reasoning and Acting (ICLR 2023)
- Tree of Thoughts (NeurIPS 2023)
- Reflexion: Language Agents with Verbal Reinforcement Learning (NeurIPS 2023)
- ToolFormer / Gorilla / API-Bank
- MMMU / MMBench (多模态评测)
- SigLIP (ICCV 2023)

## 目录结构（建议）

```
03_multimodal_agent/
├── DESIGN.md
├── journal.md
├── mm_agent/
│   ├── retrieval/        # 多模态检索
│   ├── compression/      # 长上下文压缩
│   ├── planner/          # Agent 规划器
│   ├── tools/            # 工具调用框架
│   ├── memory/           # 记忆系统
│   └── engine.py         # 入口
├── eval/
│   ├── tasks/            # 自建评测任务集
│   └── runner.py
├── tests/
└── pyproject.toml
```
