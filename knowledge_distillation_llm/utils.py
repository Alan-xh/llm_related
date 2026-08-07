import torch
from typing import Optional


def compute_fkl(
    logits: torch.Tensor, 
    teacher_logits: torch.Tensor, 
    target: torch.Tensor, 
    padding_id: int,
    reduction: str = "sum",
    temp: float = 1.0, 
) -> torch.Tensor:
    """
    计算前向 KL 散度 (Forward KL Divergence)，常用于知识蒸馏。
    公式: KL(P_teacher || P_student)
    
    Args:
        logits (torch.Tensor): 学生模型的输出 logits，形状通常为 (batch_size, seq_len, vocab_size).
        teacher_logits (torch.Tensor): 教师模型的输出 logits，形状通常为 (batch_size, seq_len, vocab_size).
        target (torch.Tensor): 标签/目标张量，用于识别 padding 位置，形状通常为 (batch_size, seq_len).
        padding_id (int): 填充标记（padding）的 ID，该位置的损失将被忽略。
        reduction (str, optional): 归约方式，可选 "sum"（求和）或无。默认为 "sum".
        temp (float, optional): 温度系数 (Temperature)，用于平滑概率分布。默认为 1.0.

    Returns:
        torch.Tensor: 计算得到的 KL 散度损失值。
    """
    logits = logits / temp
    teacher_logits = teacher_logits / temp

    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    
    kl = (teacher_probs * (teacher_log_probs - log_probs)) 
    kl = kl.sum(-1)
    
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        kl = kl.sum()

    return kl


def compute_rkl(
    logits: torch.Tensor, 
    teacher_logits: torch.Tensor, 
    target: torch.Tensor, 
    padding_id: int,
    reduction: str = "sum", 
    temp: float = 1.0
) -> torch.Tensor:
    """
    计算反向 KL 散度 (Reverse KL Divergence)，常用于模式覆盖。
    公式: KL(P_student || P_teacher)
    
    Args:
        logits (torch.Tensor): 学生模型的输出 logits，形状通常为 (batch_size, seq_len, vocab_size).
        teacher_logits (torch.Tensor): 教师模型的输出 logits，形状通常为 (batch_size, seq_len, vocab_size).
        target (torch.Tensor): 标签/目标张量，用于识别 padding 位置，形状通常为 (batch_size, seq_len).
        padding_id (int): 填充标记（padding）的 ID，该位置的损失将被忽略。
        reduction (str, optional): 归约方式，可选 "sum"（求和）或无。默认为 "sum".
        temp (float, optional): 温度系数 (Temperature)，用于平滑概率分布。默认为 1.0.

    Returns:
        torch.Tensor: 计算得到的反向 KL 散度损失值。
    """
    logits = logits / temp
    teacher_logits = teacher_logits / temp

    probs = torch.softmax(logits, -1, dtype=torch.float32)
    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    
    kl = (probs * (log_probs - teacher_log_probs))
    kl = kl.sum(-1)
    
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        kl = kl.sum()
        
    return kl


def compute_skewed_fkl(
    logits: torch.Tensor, 
    teacher_logits: torch.Tensor, 
    target: torch.Tensor, 
    padding_id: int, 
    reduction: str = "sum", 
    temp: float = 1.0,
    skew_lambda: float = 0.1
) -> torch.Tensor:
    """
    计算偏向型前向 KL 散度 (Skewed Forward KL Divergence)，通过混合分布缓解零概率惩罚问题。
    
    Args:
        logits (torch.Tensor): 学生模型的输出 logits，形状通常为 (batch_size, seq_len, vocab_size).
        teacher_logits (torch.Tensor): 教师模型的输出 logits，形状通常为 (batch_size, seq_len, vocab_size).
        target (torch.Tensor): 标签/目标张量，用于识别 padding 位置，形状通常为 (batch_size, seq_len).
        padding_id (int): 填充标记（padding）的 ID，该位置的损失将被忽略。
        reduction (str, optional): 归约方式，可选 "sum"（求和）或无。默认为 "sum".
        temp (float, optional): 温度系数 (Temperature)，用于平滑概率分布。默认为 1.0.
        skew_lambda (float, optional): 偏向混合系数 (Skew Lambda). 默认为 0.1.

    Returns:
        torch.Tensor: 计算得到的偏向型前向 KL 散度损失值。
    """
    logits = logits / temp
    teacher_logits = teacher_logits / temp

    probs = torch.softmax(logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    mixed_probs = skew_lambda * teacher_probs + (1 - skew_lambda) * probs
    mixed_log_probs = torch.log(mixed_probs)
    teacher_log_probs = torch.log_softmax(teacher_logits, -1, dtype=torch.float32)
    
    kl = (teacher_probs * (teacher_log_probs - mixed_log_probs))
    kl = kl.sum(-1)
    
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        kl = kl.sum()
         
    return kl


def compute_skewed_rkl(
    logits: torch.Tensor, 
    teacher_logits: torch.Tensor, 
    target: torch.Tensor,
    padding_id: int,
    reduction: str = "sum", 
    temp: float = 1.0,
    skew_lambda: float = 0.1
) -> torch.Tensor:
    """
    计算偏向型反向 KL 散度 (Skewed Reverse KL Divergence)。
    
    Args:
        logits (torch.Tensor): 学生模型的输出 logits，形状通常为 (batch_size, seq_len, vocab_size).
        teacher_logits (torch.Tensor): 教师模型的输出 logits，形状通常为 (batch_size, seq_len, vocab_size).
        target (torch.Tensor): 标签/目标张量，用于识别 padding 位置，形状通常为 (batch_size, seq_len).
        padding_id (int): 填充标记（padding）的 ID，该位置的损失将被忽略。
        reduction (str, optional): 归约方式，可选 "sum"（求和）或无。默认为 "sum".
        temp (float, optional): 温度系数 (Temperature)，用于平滑概率分布。默认为 1.0.
        skew_lambda (float, optional): 偏向混合系数 (Skew Lambda). 默认为 0.1.

    Returns:
        torch.Tensor: 计算得到的偏向型反向 KL 散度损失值。
    """
    logits = logits / temp
    teacher_logits = teacher_logits / temp
    
    probs = torch.softmax(logits, -1, dtype=torch.float32)
    teacher_probs = torch.softmax(teacher_logits, -1, dtype=torch.float32)
    mixed_probs = (1 - skew_lambda) * teacher_probs + skew_lambda * probs
    mixed_log_probs = torch.log(mixed_probs)
    log_probs = torch.log_softmax(logits, -1, dtype=torch.float32)
    
    kl = (probs * (log_probs - mixed_log_probs))
    kl = kl.sum(-1)
    
    if reduction == "sum":
        pad_mask = target.eq(padding_id)
        kl = kl.masked_fill_(pad_mask, 0.0)
        kl = kl.sum()

    return kl