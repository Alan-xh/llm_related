from transformers import AutoModelForCausalLM, AutoTokenizer, DefaultDataCollator
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Trainer, TrainingArguments
from dataset import SFTDataset
from utils import compute_fkl, compute_rkl, compute_skewed_fkl, compute_skewed_rkl
from typing import Optional, Dict, Any, Union, Tuple, Callable, List
from torch.utils.data import Dataset
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction


class KGTrainer(Trainer):
    """
    自定义的知识蒸馏训练器 (KGTrainer)，继承自 Hugging Face 的 Trainer。
    主要用于实现学生模型 (Student Model) 与教师模型 (Teacher Model) 之间的知识蒸馏训练。
    """
    
    def __init__(
        self,
        model: Optional[Union[nn.Module, str]] = None,
        teacher_model: Optional[nn.Module] = None,
        if_use_entropy: bool = False,
        args: Optional[TrainingArguments] = None,
        data_collator: Optional[Any] = None, 
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[Any] = None,
        model_init: Optional[Callable[[], nn.Module]] = None, 
        compute_metrics: Optional[Callable[[EvalPrediction], Dict[str, float]]] = None, 
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None), 
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
    ) -> None:
        """
        初始化 KGTrainer 实例，完成基础训练配置以及教师模型与蒸馏超参数的绑定。

        Args:
            model (Optional[Union[nn.Module, str]], optional): 待训练的学生模型实例或模型路径. 默认为 None.
            teacher_model (Optional[nn.Module], optional): 用于提供软标签指导的教师模型实例. 默认为 None.
            if_use_entropy (bool, optional): 是否结合学生模型原生交叉熵损失与蒸馏损失共同训练. 默认为 False.
            args (Optional[TrainingArguments], optional): Hugging Face 的训练参数配置对象. 默认为 None.
            data_collator (Optional[Any], optional): 数据整理器，用于将样本批次化. 默认为 None.
            train_dataset (Optional[Dataset], optional): 训练数据集对象. 默认为 None.
            eval_dataset (Optional[Union[Dataset, Dict[str, Dataset]]], optional): 验证数据集对象或字典. 默认为 None.
            tokenizer (Optional[Any], optional): 分词器对象. 默认为 None.
            model_init (Optional[Callable[[], nn.Module]], optional): 模型初始化函数. 默认为 None.
            compute_metrics (Optional[Callable[[EvalPrediction], Dict[str, float]]], optional): 评估指标计算函数. 默认为 None.
            callbacks (Optional[List[TrainerCallback]], optional): 训练过程中的回调函数列表. 默认为 None.
            optimizers (Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]], optional): 优化器和学习率调度器的元组. 默认为 (None, None).
            preprocess_logits_for_metrics (Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]], optional): 用于在计算指标前预处理 logits 的函数. 默认为 None.

        Returns:
            None
        """
        # 调用父类 Trainer 的初始化方法，完成基础训练配置
        super().__init__(
            model,
            args,
            data_collator,
            train_dataset,
            eval_dataset,
            tokenizer,
            model_init,
            compute_metrics,
            callbacks,
            optimizers,
            preprocess_logits_for_metrics,
        )
        self.teacher_model: Optional[nn.Module] = teacher_model      # 教师模型（通常参数量较大，提供软标签指导）
        self.if_use_entropy: bool = if_use_entropy                  # 是否结合交叉熵损失（SFT Loss）与蒸馏损失共同训练
        
    
    def compute_loss(self, model: nn.Module, inputs: Dict[str, torch.Tensor], return_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        重写 Trainer 的 compute_loss 方法，用于自定义前向传播及知识蒸馏损失计算逻辑。

        Args:
            model (nn.Module): 当前正在训练的学生模型实例.
            inputs (Dict[str, torch.Tensor]): 包含输入数据特征及标签的字典 (例如 'input_ids', 'attention_mask', 'labels').
            return_outputs (bool, optional): 是否在返回损失的同时返回模型的前向输出结果. 默认为 False.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, Any]]: 
                如果 return_outputs 为 False，则仅返回计算出的总损失张量；
                如果 return_outputs 为 True，则返回一个元组 `(loss_total, outputs)`。
        """
        # 1. 学生模型前向传播，获取输出（包含 logits 和原生的交叉熵 loss）
        outputs = model(**inputs)
        
        # 2. 教师模型前向传播，计算时不需要梯度更新，节省显存并加速
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
        
        loss: torch.Tensor = outputs.loss
        logits: torch.Tensor = outputs.logits
        teacher_outputs_logits: torch.Tensor = teacher_outputs.logits
        
        # 3. 处理由于不同模型词表大小或维度不一致导致的 logits 形状不匹配问题
        # 如果教师模型输出的词表维度大于学生模型，则对教师模型的 logits 进行截断以对齐
        if logits.shape[-1] != teacher_outputs_logits.shape[-1]:
            # gap = teacher_outputs_logits.shape[-1] - logits.shape[-1]
            # if gap > 0:
            #     pad_logits = torch.zeros((logits.shape[0], logits.shape[1], gap)).to(logits.device)
            #     logits = torch.cat([logits, pad_logits], dim=-1)
            
            teacher_outputs_logits = teacher_outputs_logits[:, :, :logits.shape[-1]]
        
        # 4. 获取标签数据，并利用前向 KL 散度（Forward KL）计算蒸馏损失
        labels: torch.Tensor = inputs['labels']
        kl: torch.Tensor = compute_fkl(logits, teacher_outputs_logits, labels, padding_id=-100, temp=2.0)
        
        # 5. 根据配置决定最终的总损失：
        # 如果开启 if_use_entropy，则将蒸馏损失与学生模型的原生交叉熵损失按 1:1 的权重加权混合；
        # 否则，仅使用 KL 散度损失作为总损失。
        if self.if_use_entropy:
            loss_total: torch.Tensor = 0.5 * kl + 0.5 * loss
        else:
            loss_total = kl
        
        # 根据 Trainer 的要求，返回最终损失（若 return_outputs 为 True，则同时返回模型输出）
        return (loss_total, outputs) if return_outputs else loss_total
        

if __name__ == '__main__':
    
    # 1. 加载学生模型（基础预训练语言模型，较小体量如 Qwen2.5-0.5B）
    model = AutoModelForCausalLM.from_pretrained("Qwen2.5-0.5B-Instruct")
    
    # 定义 LoRA 微调配置
    lora_config = LoraConfig(
        r=8,  
        lora_alpha=256,  
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.1, 
        task_type=TaskType.CAUSAL_LM
    )
    
    # 使用 LoRA 方法将学生模型包装为 PEFT 模型，并转移至 GPU
    model = get_peft_model(model, lora_config)
    model.cuda()
    # 打印可训练参数量及其占比
    print(model.print_trainable_parameters())
    
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained("Qwen2.5-0.5B-Instruct")
    
    # 2. 加载教师模型（体量较大的模型，如 Qwen2.5-7B-Instruct）
    teacher_model = AutoModelForCausalLM.from_pretrained("Qwen2.5-7B-Instruct")
    
    # 加载已在特定领域数据上通过 LoRA 微调过的教师模型权重
    lora_path = 'qwen2.5_7b/lora/sft'
    teacher_model = PeftModel.from_pretrained(teacher_model, lora_path)
    teacher_model.cuda()
    teacher_model.eval()  # 设置教师模型为评估模式（不启用 Dropout 等）
    
    # 3. 设置训练超参数及训练配置
    args = TrainingArguments(
        output_dir='./results',                    # 训练产物保存路径
        num_train_epochs=10,                       # 训练总轮数
        do_train=True,                             # 执行训练
        per_device_train_batch_size=2,             # 单张 GPU 的批处理大小
        gradient_accumulation_steps=16,            # 梯度累积步数
        logging_steps=10,                          # 日志打印步数间隔
        report_to='tensorboard',                   # 使用 Tensorboard 记录日志
        save_strategy='epoch',                     # 每个 epoch 保存一次模型
        save_total_limit=10,                       # 最多保存的 checkpoint 数量限制
        bf16=True,                                 # 开启 bfloat16 混合精度训练
        learning_rate=0.0005,                      # 学习率
        lr_scheduler_type='cosine',                # 学习率衰减策略（余弦退火）
        dataloader_num_workers=8,                  # 数据加载线程数
        dataloader_pin_memory=True                 # 锁页内存，加速数据传输至 GPU
    )
    
    # 4. 准备数据和数据整理器
    data_collator = DefaultDataCollator()
    dataset = SFTDataset('data.json', tokenizer=tokenizer, max_seq_len=512)
    
    # 5. 初始化自定义的知识蒸馏训练器 (KGTrainer)
    trainer = KGTrainer(
        model=model,
        teacher_model=teacher_model, 
        if_use_entropy=True,                       # 开启混合损失（蒸馏损失 + 交叉熵损失）
        args=args, 
        train_dataset=dataset, 
        tokenizer=tokenizer, 
        data_collator=data_collator
    )
    
    # 6. 开始训练（resume_from_checkpoint 设为 False 表示从头开始训练；若为 True 则从最近的 checkpoint 恢复）
    trainer.train(resume_from_checkpoint=False)
    
    # 7. 保存最终训练好的模型及状态
    trainer.save_model('./saves')
    trainer.save_state()