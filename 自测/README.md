# 自测问答系统

> **核心原理**：读笔记是"再认"（看到答案觉得懂），自测是"回忆"（不看答案自己掏出来）。后者费力但才真正固化记忆、暴露漏洞。面试考的就是闭卷回忆能力。

## 用法

1. **闭卷作答**：只看问题，口答或默写，**不要先翻材料**。
2. **核对标记**：答完再翻对应材料核对，在答错的题前标 `✗`。
3. **间隔重答**：错题隔 **1 天 / 3 天 / 7 天** 重答，连续两次答对算"通过"。
4. **代码题**：合上源码默写，或对着空气讲一遍 forward 里每步 shape 变化。讲不出的地方就是"以为懂了其实没懂"。

## 领域索引

| 领域 | 对应材料 | 自测文件 |
| --- | --- | --- |
| 01 论文精读 | `paper/` | `01_论文精读/自测.md` |
| 02 LLM 基础 | `llm_interview_note/01.*`、`nlp/` | `02_LLM基础/自测.md` |
| 03 LLM 架构 | `llm_interview_note/02.*` | `03_LLM架构/自测.md` |
| 04 分布式训练 | `llm_interview_note/04.*` | `04_分布式训练/自测.md` |
| 05 训练与微调 | `llm_interview_note/03.*`、`05.*` | `05_训练与微调/自测.md` |
| 06 推理优化 | `llm_interview_note/06.*` | `06_推理优化/自测.md` |
| 07 强化学习与对齐 | `llm_interview_note/07.*`、`ppo_from_scratch/`、`s1_from_scratch/` | `07_强化学习与对齐/自测.md` |
| 08 RAG 与 Agent | `llm_interview_note/08.*` | `08_RAG与Agent/自测.md` |
| 09 评估与应用 | `llm_interview_note/09.*`、`10.*` | `09_评估与应用/自测.md` |
| 10 从零训练 LLM | `train_llm_from_scratch/` | `10_从零训练LLM/自测.md` |
| 11 MoE | `train_moe_from_scratch/`、MoE 笔记 | `11_MoE/自测.md` |
| 12 多模态 | `train_multimodal_from_scratch/`、`train_siglip_from_scratch/` | `12_多模态/自测.md` |
| 13 DeepSeek | `deepseek_learn/` | `13_DeepSeek/自测.md` |
| 14 手写网络组件 | `handwrite_network/` | `14_手写网络组件/自测.md` |
| 15 NLP 基础算法 | `nlp/` | `15_NLP基础算法/自测.md` |
| 16 机器学习任务 | `machine_learning_tasks/` | `16_机器学习任务/自测.md` |
| 17 知识蒸馏 | `knowledge_distillation_llm/` | `17_知识蒸馏/自测.md` |

## 纪律

- **只问不答**：文件里不放答案，逼自己回忆。答案在你已有的材料里。
- **诚实标记**：含糊其辞、看了提示才想起来的，都算 `✗`。
- **错题优先**：每次先刷上次标 `✗` 的，而不是从头读一遍已经会的。
