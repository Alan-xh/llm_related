# 常规的大模型训练方案
pretrain -> sft -> rl

# DeepSeek-R1-Zero
pretrain -> rl

缺陷：中英文混合、格式混乱

# DeepSeek-R1
pretrain -> sft一阶段 -> rl一阶段 -> sft二阶段 -> rl二阶段

## sft一阶段（冷启动）

目的：引入数千条高质量长推理链数据对基础模型微调，强制规范输出格式（如\<think>推理过程\</think>），提升可读性。\
数据来源：收集DeepSeek-R1-Zero的输出结果，以可读的格式呈现，最后通过人工标注者进行后处理以优化结果

## rl一阶段（推理导向的rl）

rl方法：GRPO\
奖励模型：基于规则的奖励（答案准确性和语言一致性），针对代码、数学、编程等有固定答案的任务设计奖励函数。

## sft二阶段

数据来源：推理数据和非推理数据合并

推理数据：rl一阶段checkpoint输出数据（60万）。rl一阶段，仅纳入了可以基于规则的奖励进行评估的数据。在sft二阶段，通过引入额外的数据来扩展数据集，其中一些数据通过将真实答案和模型预测输入DeepSeek-V3进行判断，使用生成式奖励模型。此外，由于模型输出有时会显得混乱且难以阅读，过滤掉了包含混合语言、长段落和代码块的推理链。对于每个提示，采样多个回答，仅保留正确的回答。收集了大约60万个与推理相关的训练样本。

非推理数据：如写作、事实问答、自我认知和翻译等，重用DeepSeek-V3监督微调数据集的部分内容。收集了大约20万个与推理无关的训练样本。

## rl二阶段(通用对齐的rl)

通用对齐RL（RLHF）：融入人类偏好奖励模型（Helpfulness & Harmlessness），确保模型在开放域任务中的安全性与实用性。


# 基于 GRPO 的大模型强化学习对齐 Pipeline 技术架构与接口文档

## 1. 架构总览

本流水线实现了基于 **GRPO (Group Relative Policy Optimization)** 的大语言模型数学推理对齐方案。整个架构省去了传统 PPO 训练中复杂的 Critic（价值）模型，通过让模型针对单个 Prompt 批量生成多条候选回复（`num_generations=16`），并在组内通过多维奖励函数（包含格式、标签标记、数值类型、结果正确性）进行相对优势归一化，直接对 Policy 模型进行策略梯度更新。

```
[Prompt 输入] ---> [Actor 模型 (Qwen2.5-0.5B)] ---> [生成多条候选回复 Group (N=16)]
                                                           |
          +------------------------------------------------+
          |
          v
[多维奖励计算组件] ---> [Mark 奖励] / [格式奖励] / [数值奖励] / [正确性奖励]
          |
          v
[组内标准化优势计算 (Advantage)] ---> [GRPO 策略截断更新损失] ---> [权重迭代]

```

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| Prompt 编码 | `[B, Seq_Len_Prompt]` | `[B, Seq_Len_Prompt]` | 输入 Token 化后的 Prompt 序列 |
| 模型生成 (Generations) | `[B, Seq_Len_Prompt]` | `[B * Num_Gen, Seq_Len_Completion]` | 针对每个 Prompt 批量采样生成多条回复 (`Num_Gen=16`) |
| 奖励计算器 | `[B * Num_Gen, Seq_Len]` | `[B * Num_Gen]` | 对每条生成文本评估并输出对应的标量奖励值 |
| 优势归一化 (Advantage) | `[B * Num_Gen]` | `[B * Num_Gen]` | 在组内（Group 维度）减去均值并除以标准差进行标准化 |

## 3. 核心公式与代码映射

* **群组相对优势估计公式**:
$A_i = \frac{R_i - \text{mean}(R_{\text{group}})}{\text{std}(R_{\text{group}})}$
* **代码对应**: 由 `GRPOTrainer` 内部自动完成组内奖励的收集、均值/标准差计算及优势向量生成。


* **多维奖励联合加权**:
$R_{\text{total}} = R_{\text{correctness}} + R_{\text{digit}} + R_{\text{format}} + R_{\text{mark}}$
* **代码对应**: `reward_funcs` 列表中传入的 `[mark_reward, soft_format_reward, hard_format_reward, digit_reward, correctness_reward]` 依次计算各项分值并由 Trainer 聚合。