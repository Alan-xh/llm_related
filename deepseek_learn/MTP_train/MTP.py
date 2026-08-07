"""
任务定义: 多Token预测 (Multi-Token Prediction, MTP) 大语言模型加速与训练架构
代表架构/算法: DeepSeek-Style Multi-Token Prediction (基于 Qwen2.5 构建)
核心思想与机制:
    1. 在标准自回归语言模型（Causal LLM）基础上，扩展多个预测头（MTP Heads），
       在单个训练/前向步骤中并行预测未来多个连续的 Token。
    2. 通过拼接前序隐藏状态与当前 Token 嵌入，经由轻量级多层感知机（MLP）模块进行特征融合与推演。
数学公式/目标函数:
    - 隐藏层融合: h_mtp^(k) = MLP(cat(h_prev, Embed(x)))
    - 交叉熵损失: L_total = sum_{k=0}^{K} CrossEntropy(Head_k(h^(k)), y_{t+k})
数据输入规范:
    - 输入 (input_ids): [Batch_Size, Seq_Len] (LongTensor)
    - 掩码 (attention_mask): [Batch_Size, Seq_Len] (FloatTensor)
    - 标签 (labels): [Batch_Size, Seq_Len] (LongTensor)
"""

import os
import json
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, AutoModelForCausalLM


# ==================== 3. 超参数与全局配置 (Hyperparameters & Config) ====================
class Config:
    """
    模型与训练全局配置文件。
    """
    def __init__(self,
                 llm_model_path: str = '/home/user/Downloads/Qwen2.5-0.5B-Instruct',
                 predict_tokens_num: int = 5,
                 **kwargs):
        self.llm_model_path = llm_model_path
        self.predict_tokens_num = predict_tokens_num
        super().__init__(**kwargs)


# ==================== 4. 数据处理与 Dataset 管道 (Data Pipeline & Utils) ====================
class MyDataset(Dataset):
    """
    多轮对话 JSONL 数据集处理管道。
    """
    def __init__(self, data_path: str, tokenizer):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        
        with open(self.data_path, 'r', encoding='utf-8') as f:
            self.datas = f.readlines()

    def __len__(self) -> int:
        return len(self.datas)
    
    def __getitem__(self, index: int) -> Dict[str, List[int]]:
        sample = self.datas[index].strip()
        sample = json.loads(sample)
        conversations = sample['conversations']
        user = conversations[0]['content']
        assistant = conversations[1]['content']
        
        q = self.tokenizer.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)
        a = assistant + self.tokenizer.eos_token
        
        q_input_ids = self.tokenizer(q)['input_ids']
        a_input_ids = self.tokenizer(a)['input_ids']
        
        # 输入序列与标签拼接 (Prompt 部分 Mask 设为 -100)
        input_ids = q_input_ids + a_input_ids
        labels = [-100] * len(q_input_ids) + a_input_ids
        
        return {
            "input_ids": input_ids,
            "labels": labels,
        }


class MyDataCollator:
    """
    数据填充与批处理工具 (Padding Collator)。
    """
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(feature['input_ids']) for feature in features)
        input_ids = []
        labels = []
        for feature in features:
            input_ids.append(feature['input_ids'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['input_ids'])))
            labels.append(feature['labels'] + [self.tokenizer.pad_token_id] * (max_len - len(feature['labels'])))
            
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


# ==================== 5. 核心子模块 / Encoder / Decoder (Sub-components) ====================
class MTPModule(nn.Module):
    """
    多 Token 预测子模块 (MTP Sub-module)。
    将前一层的隐藏状态与当前 Token 的嵌入向量拼接后进行非线性变换。
    
    数学原理 / 变换逻辑:
        h_out = Linear2(ReLU(Linear1(cat(h_prev, e_token))))

    Args:
        hidden_size (int): 主干大模型的隐藏层维度。
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(2 * hidden_size, 4 * hidden_size)
        self.linear2 = nn.Linear(4 * hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (Tensor): 拼接后的张量，shape: [B, Seq_Len, 2 * Hidden_Size]
        Outputs:
            out (Tensor): 变换后的隐藏状态，shape: [B, Seq_Len, Hidden_Size]
        """
        x = self.linear1(x) # [B, Seq_Len, 4 * Hidden_Size]
        x = F.silu(x)       # 现代高效激活函数
        x = self.linear2(x) # [B, Seq_Len, Hidden_Size]
        return x


# ==================== 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model) ====================
class MTP(nn.Module):
    """
    多 Token 预测主模型架构 (Multi-Token Prediction Framework)。
    整合主干 Causal LLM 与多个 MTP 预测头，支持并行前向推理与投机采样生成。
    """
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.main_model = AutoModelForCausalLM.from_pretrained(self.config.llm_model_path).base_model
        
        # MTP 预测头模块列表
        self.mtp_modules = nn.ModuleList([
            MTPModule(self.main_model.config.hidden_size) 
            for _ in range(self.config.predict_tokens_num - 1)
        ])
        
        # 共享参数输出头
        self.output_head = nn.Linear(self.main_model.config.hidden_size, self.main_model.config.vocab_size)
         
    def forward_main(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs):
        """
        主干模型前向计算。
        Inputs:
            input_ids (Tensor): 输入 Token 索引，shape: [B, Seq_Len]
            attention_mask (Tensor, optional): 注意力掩码，shape: [B, Seq_Len]
        Outputs:
            main_hidden_output (Tensor): 主干隐藏状态，shape: [B, Seq_Len, Hidden_Size]
            main_head_output (Tensor): 主干分类 Logits，shape: [B, Seq_Len, Vocab_Size]
        """
        main_hidden_output = self.main_model(input_ids, attention_mask, **kwargs).last_hidden_state
        main_head_output = self.output_head(main_hidden_output)
        return main_hidden_output, main_head_output
    
    def forward_mtp(self, input_ids: torch.Tensor, previous_hidden_output: torch.Tensor, head_index: int):
        """
        单个 MTP 预测头前向计算。
        Inputs:
            input_ids (Tensor): 输入 Token 索引，shape: [B, Seq_Len]
            previous_hidden_output (Tensor): 前序隐藏状态，shape: [B, Seq_Len, Hidden_Size]
            head_index (int): 当前 MTP 头索引
        Outputs:
            mtp_hidden_output (Tensor): MTP 隐藏状态，shape: [B, Seq_Len, Hidden_Size]
            mtp_head_output (Tensor): MTP 分类 Logits，shape: [B, Seq_Len, Vocab_Size]
        """
        input_embed = self.main_model.get_input_embeddings()(input_ids) # [B, Seq_Len, Hidden_Size]
        mtp_input = torch.cat([previous_hidden_output, input_embed], dim=-1) # [B, Seq_Len, 2 * Hidden_Size]
        mtp_hidden_output = self.mtp_modules[head_index](mtp_input) # [B, Seq_Len, Hidden_Size]
        mtp_head_output = self.output_head(mtp_hidden_output) # [B, Seq_Len, Vocab_Size]
        
        return mtp_hidden_output, mtp_head_output
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None, **kwargs) -> Dict[str, torch.Tensor]:
        """
        总体前向推理。
        """
        outputs = {}
        main_hidden_output, main_head_output = self.forward_main(input_ids, attention_mask, **kwargs)
        previous_hidden_output = main_hidden_output
        outputs['head_main'] = main_head_output
        
        for head_index in range(0, self.config.predict_tokens_num - 1):
            previous_hidden_output, mtp_head_output = self.forward_mtp(input_ids, previous_hidden_output, head_index)
            outputs[f'mtp_head_{head_index}'] = mtp_head_output
             
        return outputs
    
    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_length: int, **kwargs) -> torch.Tensor:
        """
        基于多 Token 预测的投机采样加速生成流程。
        """
        self.eval()
        seq = input_ids.clone()
        
        while seq.size(1) < max_length:
            outputs = self.forward(seq)
            speculative_tokens = []
            
            # 1. 主模型头生成下一个 Token
            logits = outputs['head_main'][:, -1, :]
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.argmax(probs, dim=-1)
            speculative_tokens.append(next_token)
            
            # 2. 汇总 MTP 预测头生成的 Token
            for i in range(self.config.predict_tokens_num - 1):
                logits = outputs[f'mtp_head_{i}'][:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1)
                speculative_tokens.append(next_token)
             
            speculative_tokens = torch.cat(speculative_tokens, dim=-1).unsqueeze(0) # [1, Num_Tokens]
            all_tokens = torch.cat([seq, speculative_tokens], dim=-1)
            
            # 3. 验证阶段
            _, all_logits = self.forward_main(all_tokens)
            validation_logits = all_logits[:, -speculative_tokens.shape[1]:]
            
            accept_probs = []
            for i in range(speculative_tokens.shape[1]):
                logits = validation_logits[:, i]
                probs = torch.softmax(logits, dim=-1)
                token = speculative_tokens[:, i]
                token_prob = probs.gather(1, token.unsqueeze(0))
                accept_probs.append(token_prob)
           
            accept_probs = torch.cat(accept_probs, dim=-1)
            accept_mask = (accept_probs > 1e-6)
            
            if accept_mask.any():
                reject_token_index = (~accept_mask).nonzero(as_tuple=True)[1]
                if reject_token_index.shape[0] > 0:
                    accept_num = reject_token_index[0]
                else:
                    accept_num = speculative_tokens.shape[1]
            else:
                accept_num = 0      
             
            if accept_num > 0:
                accept_tokens = speculative_tokens[:, :accept_num]
                seq = torch.cat([seq, accept_tokens], dim=1)
            else:
                logits = outputs['head_main'][:, -1, :]
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.argmax(probs, dim=-1).unsqueeze(0)
                seq = torch.cat([seq, next_token], dim=-1)
                 
        return seq


# ==================== 7. 损失函数与评估指标 (Loss & Metrics) ====================
def train(config: Config, model: nn.Module, dataloader: DataLoader, optimizer: torch.optim.Optimizer, 
          writer: SummaryWriter, device: str, epochs: int, print_step: int, save_step: int, save_path: str):
    """
    联合多任务训练 Pipeline。
    """
    steps = 0
    model.train()
    for epoch in range(epochs):
        for step, batch in enumerate(dataloader):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            
            main_hidden_output, main_head_output = model.forward_main(input_ids)
            previous_hidden_output = main_hidden_output
            
            for index in range(0, config.predict_tokens_num - 1):
                previous_hidden_output, mtp_head_output = model.forward_mtp(input_ids, previous_hidden_output, index)
                
                # MTP 损失计算
                mtp_head_output = mtp_head_output[:, :-(1 + index + 1)]
                mtp_head_output = mtp_head_output.reshape(-1, model.main_model.config.vocab_size)
                
                target = labels[:, 1 + index + 1:]
                target = target.contiguous().view(-1)
                
                mtp_loss = F.cross_entropy(mtp_head_output, target, ignore_index=-100)
                mtp_loss.backward(retain_graph=True)
                 
            # 主模型损失计算
            main_loss = F.cross_entropy(
                main_head_output[:, :-1].reshape(-1, model.main_model.config.vocab_size), 
                labels[:, 1:].reshape(-1), 
                ignore_index=-100
            )
            
            main_loss.backward()
            optimizer.step()
             
            if (steps + 1) % print_step == 0:
                writer.add_scalar('main_loss', main_loss.item(), steps)
                writer.add_scalar('mtp_loss', mtp_loss.item(), steps)
                print(f"Epoch {epoch+1}, Step {steps+1}, main_loss: {main_loss.item():.4f}, mtp_loss: {mtp_loss.item():.4f}")
                 
            if (steps + 1) % save_step == 0:
                torch.save(model.state_dict(), f"{save_path}/model_{steps}.pth")
             
            steps += 1


# ==================== 8. 训练/推理逻辑与入口 (Training/Inference Execution) ====================
if __name__ == '__main__':
    writer = SummaryWriter('./runs')
    config = Config()
    model = MTP(config)
    model.cuda()
    
    print(f'模型可训练参数量为: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}')
    
    tokenizer = AutoTokenizer.from_pretrained(config.llm_model_path)
    dataset = MyDataset('/home/user/wyf/deepseek_learn/MTP_train/lora_medical.jsonl', tokenizer)
    dataloader = DataLoader(
        dataset=dataset, 
        batch_size=8, 
        shuffle=True, 
        num_workers=2, 
        collate_fn=MyDataCollator(tokenizer)
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    save_path = './mtp'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        
    train(
        config=config, 
        model=model, 
        dataloader=dataloader, 
        optimizer=optimizer, 
        writer=writer, 
        device='cuda', 
        epochs=10, 
        print_step=10, 
        save_step=500, 
        save_path=save_path
    )