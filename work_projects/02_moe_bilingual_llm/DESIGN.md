# 项目 02: MoE 双语模型全流程对齐 (mini-moe-zh)

## 一句话目标

从零训练一个 500M 总参 / ~100M 激活参数的稀疏 MoE 中英双语模型，跑通 Pretrain -> SFT -> DPO 全流程，C-Eval 达到 Qwen2.5-0.5B 的 70%+，MMLU 达到 60%+，并提供完整训练日志与评估体系。

## 为什么这个项目难

`train_moe_from_scratch/` 已有 MoE 训练雏形，但只是 toy scale。真要做出能上 benchmark 的模型，会遇到：

- MoE 路由在训练初期极易塌缩（所有 token 都路由到 1-2 个 expert）
- 100B token 预训练的 loss spike 难预测，bf16 数值精度问题在 MoE 上更严重
- 中英双语数据配比不当会让模型偏向某一语言，跨语言迁移差
- Tokenizer 训练不好（中文字粒度太细）会导致序列变长、训练效率低
- 评估阶段要严格防 benchmark contamination，否则数字虚高
- DPO 偏好数据质量直接决定对齐效果，公开数据集质量参差

## 核心难点分解

| 编号 | 难点 | 难度 | 备注 |
| --- | --- | --- | --- |
| D1 | Tokenizer 训练（中英 BPE，~30K vocab） | ★★★ | 字 vs 词粒度权衡 |
| D2 | 数据流水线（去重、过滤、配比、课程学习） | ★★★★ | minhash 去重 + 质量过滤 + 配比调优 |
| D3 | MoE 路由稳定性 + 负载均衡 | ★★★★ | auxiliary loss + router z-loss + 专家容量 |
| D4 | 大规模预训练稳定性 | ★★★★ | loss spike 处理、grad clip、bf16 + dynamic loss scale |
| D5 | 评估体系（防 contamination） | ★★★ | C-Eval/MMLU/MT-Bench + 自建 holdout |
| D6 | DPO 偏好数据构造与训练 | ★★★ | 偏好数据质量 + DPO loss 调参 |

## 可行性论证

- **算力可行**：500M MoE 100B token 预训练，4×A100 80GB 约 5-7 天
- **数据全部公开**：SkyPile-150M / Wudao / RedPajama / OpenHermes / UltraChat
- **已有基础**：`train_llm_from_scratch/` + `train_moe_from_scratch/` 已跑通小规模 MoE
- **架构成熟**：参考 Qwen-MoE / DeepSeek-MoE 的公开报告，关键设计点都明确
- **评估可行**：C-Eval/MMLU 都是公开 benchmark，本地能跑

## 里程碑路线图

### M0: Tokenizer + 数据流水线
- 训练 32K vocab BPE tokenizer（中英混合语料）
- 数据：SkyPile-150M + RedPajama-en，去重 + 质量过滤
- 输出 100B token 的预训练数据集
- **DoD**: tokenizer 压缩率（bytes/token）接近 Qwen2.5；数据集抽样质量人审通过

### M1: MoE 模型实现
- 12 层 / d_model=1024 / 8 expert / top-2 routing
- auxiliary load balance loss + router z-loss
- 共享专家（参考 DeepSeek-MoE）
- **DoD**: 单 batch 前向 + 反向跑通，专家利用率方差 < 0.3

### M2: 预训练（100B token）
- 4×A100 + DeepSpeed ZeRO-2 + bf16
- cosine schedule, warmup 2B, peak lr 3e-4
- **DoD**: loss 收敛到 < 2.0，无 spike 后无法恢复；专家利用率稳定

### M3: SFT
- 数据：OpenHermes + Bell-OpenChat 中英混合，~5M 条
- packing 训练，max_len=4096
- **DoD**: SFT 后模型能稳定遵循指令，MT-Bench 初步评估 > 4.0

### M4: DPO
- 偏好数据：UltraFeedback + 自建中英偏好对
- DPO 训练，beta=0.1
- **DoD**: DPO 后 MT-Bench 提升 >= 0.3，reward margin 显著

### M5: 评估
- C-Eval / MMLU / GSM8K / MT-Bench
- 严格防 contamination（训练数据 hash 比对）
- **DoD**: C-Eval >= Qwen2.5-0.5B × 70%，MMLU >= 60%

### M6: 发布 + 复盘
- 模型权重 + 训练日志 + 评估报告
- 写技术博客总结踩坑
- **DoD**: 模型可在 HuggingFace 发布，博客可对外

## 评估指标

| 指标 | 目标 | 测量方式 |
| --- | --- | --- |
| C-Eval (5-shot) | >= 35 | 官方评测脚本 |
| MMLU (5-shot) | >= 40 | lm-eval-harness |
| GSM8K (5-shot) | >= 15 | lm-eval-harness |
| MT-Bench | >= 5.0 | GPT-4 评判 |
| 专家利用率方差 | < 0.2 | 训练日志统计 |
| 训练稳定性 | spike 数 < 3 | 训练日志 |

## 风险与缓解

| 风险 | 缓解 |
| --- | --- |
| 算力不足 | 可降到 200M 总参，100B token -> 30B token；优先保架构完整 |
| MoE 路由塌缩 | 启动时强制噪声 + 高 auxiliary loss，前 1B token 严格监控 |
| 评估虚高 | 训练前用 n-gram hash 标记 benchmark 题目，从训练数据剔除 |
| 数据质量参差 | 多源数据 + 质量打分（perplexity + 规则） + 人工抽样 |
| DPO 训练发散 | 严格 SFT -> DPO 数据同分布，beta 从 0.05 起调 |

## 技术栈

- Python 3.10+ / PyTorch 2.x
- DeepSpeed (ZeRO-2/3) 或 FSDP
- transformers + tokenizers
- datasets / dataloader 自定义
- lm-eval-harness (评估)
- tensorboard / wandb (监控)

## 参考资料

- DeepSeek-MoE: Towards Ultimate Expert Specialization (arxiv 2401.06066)
- Qwen-MoE / Switch Transformer
- GShard: Scaling Giant Models with Conditional Computation
- Mixtral of Experts
- DPO: Direct Preference Optimization (NeurIPS 2023)
- Skywork-MoE 训练实践报告

## 目录结构（建议）

```
02_moe_bilingual_llm/
├── DESIGN.md
├── journal.md
├── tokenizer/
│   └── train_tokenizer.py
├── data/
│   ├── preprocess.py
│   ├── dedup.py
│   └── mix.py
├── model/
│   ├── moe.py
│   ├── router.py
│   └── config.py
├── train/
│   ├── pretrain.py
│   ├── sft.py
│   └── dpo.py
├── eval/
│   └── run_eval.py
└── configs/
```
