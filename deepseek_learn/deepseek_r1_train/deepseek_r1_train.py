"""
任务定义: 基于 GRPO (Group Relative Policy Optimization, 组相对策略优化) 的大语言模型强化学习对齐
领域分类: 自然语言处理 / 大模型对齐 / 强化学习 (RLHF)
代表架构/算法: DeepSeekMath / TRLLM GRPOTrainer (Qwen2.5-0.5B-Instruct + LoRA)
核心思想与机制: 
    - 采用群组相对策略优化（GRPO），无需单独训练 Critic 模型，而是通过对同一 Prompt 生成一组回复（Num Generations），
      计算组内奖励的均值与标准差来进行优势估计（Advantage Estimation）。
    - 配合多重奖励函数（正确性、格式、数值类型、标记细粒度奖励）缓解强化学习中的稀疏奖励与冷启动难题。
数学公式/目标函数:
    - 优势计算: A_i = (R_i - mean(R)) / std(R)
    - 目标函数: J_GRPO = E [ min(r_t * A_i, clip(r_t, 1-eps, 1+eps) * A_i) ]
数据输入规范:
    - 输入文本 (Prompt): 包含 System Prompt 与用户问题，维度形式为 List[Dict[str, str]]
    - 输出文本 (Completion): 模型生成的推理步骤与最终答案，形式为 `<think>...</think><answer>...</answer>`
"""

import re
import math
import torch
import torch.nn as nn
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import trl
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig, get_peft_model, TaskType


# ==========================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ==========================================
MODEL_NAME = "/home/user/Downloads/Qwen2.5-0.5B-Instruct"
DATASET_PATH = "/home/user/wyf/deepseek_learn/gsm8k_chinese"
OUTPUT_DIR = "output"

SYSTEM_PROMPT = """
按照如下格式生成：
<think>
...
</think>
<answer>
...
</answer>
"""


# ==========================================
# 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils)
# ==========================================
def process_data(data: Dataset) -> Dataset:
    """
    处理原始数据集，构建符合 TRL GRPOTrainer 规范的 prompt 与 answer 字段。

    Args:
        data (Dataset): 原始 Hugging Face 数据集对象。

    Returns:
        Dataset: 包含 'prompt' 和 'answer' 字段的处理后数据集。
    """
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question_zh-cn']}
        ],
        'answer': x['answer_only']
    }) 
    return data


def extract_answer(text: str) -> str:
    """
    从模型生成的完整文本中提取 <answer> 标签内部的文本结果。

    Args:
        text (str): 模型生成的完整文本字符串。

    Returns:
        str: 提取并剥离空白字符后的纯答案字符串。
    """
    try:
        answer = text.split("<answer>")[-1]
        answer = answer.split("</answer>")[0]
        return answer.strip()
    except Exception:
        return ""


# ==========================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ==========================================
def mark_num(text: str) -> float:
    """
    标记奖励函数：对输出中的标签完整性进行分步细粒度打分，缓解稀疏奖励。
    """
    reward = 0.0
    if text.count("<think>\n") == 1:
        reward += 0.125
    if text.count("</think>\n") == 1:
        reward += 0.125
    if text.count("<answer>\n") == 1:
        reward += 0.125
    if text.count("</answer>\n") == 1:
        reward += 0.125
    return reward


def correctness_reward(prompts, completions, answer, **kwargs) -> list:
    """
    结果正确性奖励：判断模型提取出的答案是否与标准答案一致。
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    print(f"问题:\n{prompts[0][-1]['content']}", f"\n答案:\n{answer[0]}", f"\n模型输出:\n{responses[0]}", f"\n提取后的答案:\n{extracted_responses[0]}")
    return [2.0 if response == str(ans) else 0.0 for response, ans in zip(extracted_responses, answer)]


def digit_reward(completions, **kwargs) -> list:
    """
    数值类型奖励：当答案为纯数字时给予基础奖励，防止奖励完全为零导致模型停滞。
    """
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_answer(r) for r in responses]
    return [0.5 if response.isdigit() else 0.0 for response in extracted_responses]


def hard_format_reward(completions, **kwargs) -> list:
    """
    硬格式奖励：使用严格的正则表达式匹配输出格式。
    """
    pattern = r"^<think>\n.*?n</think>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


def soft_format_reward(completions, **kwargs) -> list:
    """
    软格式奖励：允许标签间存在部分松散空白字符的格式匹配。
    """
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, response) for response in responses]
    return [0.5 if match else 0.0 for match in matches]


def mark_reward(completions, **kwargs) -> list:
    """
    标记奖励组合：调用 mark_num 返回细粒度标签奖励。
    """
    responses = [completion[0]["content"] for completion in completions]
    return [mark_num(response) for response in responses]


# ==========================================
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ==========================================
def main():
    """
    主执行函数：加载模型与分词器、配置 LoRA、构造 Dataset、配置 GRPO 参数并启动 Trainer 训练。
    """
    # 初始化分词器与基础模型
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.bfloat16
    )

    # 如需使用 LoRA 方法训练，可开启以下注释
    # lora_config = LoraConfig(
    #     r=8,  
    #     lora_alpha=256,  
    #     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    #     lora_dropout=0.1, 
    #     task_type=TaskType.CAUSAL_LM
    # )
    # model = get_peft_model(model, lora_config)
    
    model.cuda()
    
    # 加载并预处理数据集
    ds = load_dataset(DATASET_PATH)
    train_data = process_data(ds['train'])
    
    # 配置 GRPO 训练超参数
    training_args = GRPOConfig(
        output_dir=OUTPUT_DIR,
        learning_rate=5e-6,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.1,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        logging_steps=1,
        bf16=True,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        num_generations=16,
        max_prompt_length=256,
        max_completion_length=200,
        num_train_epochs=1,
        save_steps=100,
        max_grad_norm=0.1,
        log_on_each_node=False,
        use_vllm=False,
        report_to="tensorboard"
    )
    
    # 实例化 GRPOTrainer
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[
            mark_reward,
            soft_format_reward,
            hard_format_reward,
            digit_reward,
            correctness_reward
        ],
        args=training_args,
        train_dataset=train_data,
    )
    
    # 执行训练与模型保存
    trainer.train()
    trainer.save_model(OUTPUT_DIR)


if __name__ == '__main__':
    main()