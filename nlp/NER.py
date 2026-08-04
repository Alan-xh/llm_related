import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

'''
工业落地、快速上线	BIO	最简单，训练快，CRF解码成熟，调参容易。
学术发论文、追求SOTA BIOES 或 Span-based	BIOES提供最精细的边界信号；Span-based在嵌套数据集（如ACE2004）上表现惊艳。
实体严重嵌套（如医疗、法律） Span-based 或 生成式	序列标注（BIO等）天生不适合处理嵌套，必须换赛道。
数据极少（少于1000条） BIO + CRF	CRF 的转移矩阵能弥补数据不足，强行教会模型“I不能开头”。Span-based在少数据下容易过拟合。
实体极长（如书名、地址） BIOES	E 标签能明确告知模型“这里结束了”，防止长实体被过早截断或无限延伸。
'''

# ==================== 1. CRF 层实现 ====================
class CRF(nn.Module):
    """
    条件随机场（Conditional Random Field）层
    
    用于序列标注任务，学习标签之间的转移约束关系。
    实现前向算法计算损失，维特比算法进行解码。
    
    Attributes:
        num_tags (int): 标签类别数量
        batch_first (bool): 是否batch维度在第一维
        transitions (nn.Parameter): 标签转移矩阵 [num_tags, num_tags]
        start_transitions (nn.Parameter): 起始标签转移得分 [num_tags]
        end_transitions (nn.Parameter): 结束标签转移得分 [num_tags]
    """
    
    def __init__(self, num_tags, batch_first=True):
        """
        初始化CRF层
        
        Args:
            num_tags (int): 标签类别数量
            batch_first (bool, optional): 输入张量是否 batch_first 格式. Defaults to True.
        """
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # 转移矩阵: transitions[i][j] 表示从标签j转移到标签i的得分
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # 起始和结束标签的转移得分
        self.start_transitions = nn.Parameter(torch.randn(num_tags))
        self.end_transitions = nn.Parameter(torch.randn(num_tags))
        
    def forward(self, emissions, tags, mask):
        """
        计算负对数似然损失（Negative Log-Likelihood Loss）
        
        Args:
            emissions (torch.FloatTensor): 发射分数 [batch_size, seq_len, num_tags]
            tags (torch.LongTensor): 真实标签序列 [batch_size, seq_len]
            mask (torch.BoolTensor): padding掩码 [batch_size, seq_len]
            
        Returns:
            torch.FloatTensor: 平均负对数似然损失（标量）
        """
        if self.batch_first:
            batch_size, seq_len = tags.shape
            emissions = emissions.transpose(0, 1)  # [seq_len, batch_size, num_tags]
            tags = tags.transpose(0, 1)  # [seq_len, batch_size]
            mask = mask.transpose(0, 1)  # [seq_len, batch_size]
        
        # 计算真实路径得分
        real_score = self._compute_real_score(emissions, tags, mask)
        
        # 计算所有路径的总得分（配分函数）
        total_score = self._compute_total_score(emissions, mask)
        
        # 负对数似然损失
        nll = total_score - real_score
        return nll.mean()
    
    def _compute_real_score(self, emissions, tags, mask):
        """
        计算真实标签序列的得分,每个标签的得分 + 上一个标签转移至当前标签的得分
        
        Args:
            emissions (torch.FloatTensor): 发射分数 [seq_len, batch_size, num_tags]
            tags (torch.LongTensor): 真实标签序列 [seq_len, batch_size]
            mask (torch.BoolTensor): padding掩码 [seq_len, batch_size]
            
        Returns:
            torch.FloatTensor: 真实路径得分 [batch_size]
        """
        seq_len, batch_size = emissions.shape[:2]
        score = self.start_transitions[tags[0]]
        
        for i in range(seq_len - 1):
            current_score = emissions[i, :, tags[i]] + self.transitions[tags[i+1], tags[i]]
            score += current_score * mask[i+1]
        
        # 最后一个有效标签的得分 + 结束转移
        last_tag = tags.gather(0, mask.sum(0).long().unsqueeze(0) - 1).squeeze(0)
        score += self.end_transitions[last_tag]
        
        return score
    
    def _compute_total_score(self, emissions, mask):
        """
        使用前向算法计算所有路径的总得分（配分函数）
        
        Args:
            emissions (torch.FloatTensor): 发射分数 [seq_len, batch_size, num_tags]
            mask (torch.BoolTensor): padding掩码 [seq_len, batch_size]
            
        Returns:
            torch.FloatTensor: 所有路径的总得分 [batch_size]
        """
        seq_len, batch_size = emissions.shape[:2]
        
        # 初始化: 起始转移
        alpha = self.start_transitions + emissions[0]
        
        for i in range(1, seq_len):
            # 广播计算: [batch_size, num_tags, 1] + [num_tags, num_tags]
            alpha_expand = alpha.unsqueeze(2)  # [batch_size, num_tags, 1]
            trans_expand = self.transitions.unsqueeze(0)  # [1, num_tags, num_tags]
            
            # 计算所有可能的转移得分
            next_alpha = alpha_expand + trans_expand + emissions[i].unsqueeze(1)
            
            # LogSumExp (数值稳定的求和)
            next_alpha = torch.logsumexp(next_alpha, dim=1)
            
            # 根据mask决定是否更新
            alpha = torch.where(mask[i].unsqueeze(1), next_alpha, alpha)
        
        # 最后加上结束转移
        alpha = alpha + self.end_transitions
        total_score = torch.logsumexp(alpha, dim=1)
        
        return total_score.sum()
    
    def decode(self, emissions, mask=None):
        """
        维特比解码: 预测最优标签序列
        
        Args:
            emissions (torch.FloatTensor): 发射分数 [batch_size, seq_len, num_tags]
            mask (torch.BoolTensor, optional): padding掩码 [batch_size, seq_len]. 
                                               Defaults to None.
            
        Returns:
            torch.LongTensor: 预测的标签序列 [batch_size, seq_len]
        """
        if self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        seq_len, batch_size = emissions.shape[:2]
        
        if mask is None:
            mask = torch.ones(seq_len, batch_size, dtype=torch.bool, device=emissions.device)
        
        # 初始化
        score = self.start_transitions + emissions[0]
        history = []
        
        for i in range(1, seq_len):
            # 计算所有可能的转移
            broadcast_score = score.unsqueeze(2)
            broadcast_emissions = emissions[i].unsqueeze(1)
            next_score = broadcast_score + self.transitions + broadcast_emissions
            
            # 记录最优路径
            indices = next_score.argmax(1)
            history.append(indices)
            
            # 更新得分
            next_score = next_score.max(1)[0]
            score = torch.where(mask[i].unsqueeze(1), next_score, score)
        
        # 结束转移
        score = score + self.end_transitions
        best_tags = score.argmax(1)
        
        # 回溯
        best_paths = []
        for b in range(batch_size):
            best_tag = best_tags[b].item()
            best_path = [best_tag]
            for hist in reversed(history):
                best_tag = hist[b][best_tag].item()
                best_path.append(best_tag)
            best_path.reverse()
            best_paths.append(best_path)
        
        return torch.tensor(best_paths, device=emissions.device)


# ==================== 2. 模型定义 ====================
class BertBiLSTMCRF(nn.Module):
    """
    BERT + BiLSTM + CRF 序列标注模型
    
    结合了BERT的上下文表示能力、BiLSTM的序列建模能力和CRF的标签约束能力。
    适用于命名实体识别（NER）、词性标注（POS）等序列标注任务。
    
    Architecture:
        BERT -> Dropout -> BiLSTM -> Linear -> CRF
        
    Attributes:
        bert (BertModel): 预训练的BERT模型
        lstm (nn.LSTM): 双向LSTM层
        fc (nn.Linear): 全连接层，映射到标签数量
        dropout (nn.Dropout): Dropout层
        crf (CRF): 条件随机场层
    """
    
    def __init__(self, bert_path, num_tags, lstm_hidden_size=256, lstm_layers=2, dropout=0.1):
        """
        初始化BERT+BiLSTM+CRF模型
        
        Args:
            bert_path (str): BERT模型路径或名称（如 'bert-base-chinese'）
            num_tags (int): 标签类别数量
            lstm_hidden_size (int, optional): LSTM隐藏层维度. Defaults to 256.
            lstm_layers (int, optional): LSTM层数. Defaults to 2.
            dropout (float, optional): Dropout比率. Defaults to 0.1.
        """
        super(BertBiLSTMCRF, self).__init__()
        
        # BERT层
        self.bert = BertModel.from_pretrained(bert_path)
        self.bert_config = self.bert.config
        
        # BiLSTM层
        self.lstm = nn.LSTM(
            input_size=self.bert_config.hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        # 全连接层（映射到标签数量）
        self.fc = nn.Linear(lstm_hidden_size * 2, num_tags)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # CRF层
        self.crf = CRF(num_tags, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        模型前向传播
        
        Args:
            input_ids (torch.LongTensor): 输入token IDs [batch_size, seq_len]
            attention_mask (torch.LongTensor): 注意力掩码 [batch_size, seq_len]
            labels (torch.LongTensor, optional): 真实标签序列 [batch_size, seq_len].
                                                 如果提供，则计算损失. Defaults to None.
            
        Returns:
            tuple: 
                - 如果提供labels: (loss, emissions) 
                  loss: 负对数似然损失（标量）
                  emissions: 发射分数 [batch_size, seq_len, num_tags]
                - 如果不提供labels: (predictions, emissions)
                  predictions: 预测的标签序列 [batch_size, seq_len]
                  emissions: 发射分数 [batch_size, seq_len, num_tags]
        """
        # BERT输出
        bert_outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # BiLSTM
        lstm_output, _ = self.lstm(sequence_output)  # [batch_size, seq_len, hidden_size*2]
        
        # 全连接层
        emissions = self.fc(lstm_output)  # 置信度 [batch_size, seq_len, num_tags]
        
        # 如果提供了标签，计算损失
        if labels is not None:
            # 创建mask（排除padding位置）
            mask = attention_mask.bool()
            loss = self.crf(emissions, labels, mask)
            return loss, emissions
        else:
            # 解码预测
            mask = attention_mask.bool()
            predictions = self.crf.decode(emissions, mask)
            return predictions, emissions


# ==================== 3. 数据集类 ====================
class NERDataset(Dataset):
    """
    命名实体识别（NER）数据集类
    
    负责将原始数据转换为模型可接受的输入格式，处理BERT的tokenization
    和标签对齐问题。
    
    Attributes:
        data (list): 原始数据列表，每项为 (tokens, labels) 元组
        tokenizer (BertTokenizer): BERT分词器
        label2id (dict): 标签到ID的映射字典
        max_len (int): 最大序列长度
    """
    
    def __init__(self, data, tokenizer, label2id, max_len=128):
        """
        初始化NER数据集
        
        Args:
            data (list): 原始数据，格式为 [(tokens, labels), ...]
            tokenizer (BertTokenizer): BERT分词器实例
            label2id (dict): 标签到ID的映射，如 {'O': 0, 'B-PER': 1, ...}
            max_len (int, optional): 最大序列长度. Defaults to 128.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_len = max_len
        
    def __len__(self):
        """
        返回数据集大小
        
        Returns:
            int: 数据样本数量
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据样本
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含 'input_ids', 'attention_mask', 'labels' 的字典
                - input_ids: token IDs [max_len]
                - attention_mask: 注意力掩码 [max_len]
                - labels: 标签序列 [max_len]
        """
        # data格式: (tokens, labels)
        tokens, labels = self.data[idx]
        
        # 使用tokenizer编码
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze(0)
        attention_mask = encoding['attention_mask'].squeeze(0)
        
        # 对齐标签：将每个单词的标签映射到该单词的第一个token
        word_ids = encoding.word_ids()
        aligned_labels = []
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                # [CLS], [SEP], [PAD] 等特殊token，标记为忽略
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                # 每个单词的第一个token，使用该单词的标签
                aligned_labels.append(self.label2id[labels[word_idx]])
                previous_word_idx = word_idx
            else:
                # 同一单词的其他token（子词），标记为忽略
                aligned_labels.append(-100)
        
        # 截断或填充
        aligned_labels = aligned_labels[:self.max_len]
        if len(aligned_labels) < self.max_len:
            aligned_labels.extend([-100] * (self.max_len - len(aligned_labels)))
        
        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels_tensor
        }


# ==================== 4. 训练函数 ====================
def train_epoch(model, dataloader, optimizer, scheduler, device):
    """
    训练一个epoch
    
    遍历整个训练数据集，执行前向传播、反向传播和参数更新。
    
    Args:
        model (nn.Module): 待训练的模型
        dataloader (DataLoader): 训练数据加载器
        optimizer (torch.optim.Optimizer): 优化器
        scheduler (torch.optim.lr_scheduler): 学习率调度器
        device (torch.device): 设备（CPU或GPU）
        
    Returns:
        float: 当前epoch的平均训练损失
    """
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc='Training')
    
    for batch in progress_bar:
        # 将数据移到设备
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # 前向传播
        loss, _ = model(input_ids, attention_mask, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device, id2label):
    """
    评估模型性能
    
    在验证集或测试集上评估模型，返回预测标签和真实标签。
    
    Args:
        model (nn.Module): 待评估的模型
        dataloader (DataLoader): 数据加载器
        device (torch.device): 设备（CPU或GPU）
        id2label (dict): ID到标签的映射字典
        
    Returns:
        tuple: (predictions, true_labels)
            - predictions (list): 所有预测标签列表
            - true_labels (list): 所有真实标签列表
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # 预测
            predictions, _ = model(input_ids, attention_mask)
            
            # 收集非忽略的标签
            batch_size, seq_len = labels.shape
            for b in range(batch_size):
                pred = []
                true = []
                for t in range(seq_len):
                    if labels[b, t].item() != -100:
                        pred.append(id2label[predictions[b, t].item()])
                        true.append(id2label[labels[b, t].item()])
                all_preds.extend(pred)
                all_labels.extend(true)
    
    return all_preds, all_labels


# ==================== 5. 主程序 ====================
def main():
    """
    主训练函数
    
    配置模型参数，准备数据，执行训练循环，保存最佳模型。
    包含完整的训练流程：数据加载、模型初始化、训练、评估和保存。
    """
    # 配置
    BERT_PATH = 'bert-base-chinese'  # 中文BERT，可以替换为其他预训练模型
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 5
    LEARNING_RATE = 2e-5
    LSTM_HIDDEN_SIZE = 256
    LSTM_LAYERS = 2
    DROPOUT = 0.1
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 标签映射（以NER为例）
    labels = ['O', 'B-PER', 'I-PER', 'B-LOC', 'I-LOC', 'B-ORG', 'I-ORG']
    label2id = {label: i for i, label in enumerate(labels)}
    id2label = {i: label for i, label in enumerate(labels)}
    NUM_TAGS = len(labels)
    
    # 准备数据（这里使用示例数据，实际使用时需要加载自己的数据）
    # 格式: [(tokens, labels), ...]
    sample_data = [
        (['我', '在', '北', '京', '工', '作'], ['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O']),
        (['张', '三', '是', '中', '国', '人'], ['B-PER', 'I-PER', 'O', 'B-LOC', 'I-LOC', 'O']),
    ]
    # 在实际项目中，这里应该加载真实数据集，并划分训练集和验证集
    
    # 初始化tokenizer
    tokenizer = BertTokenizer.from_pretrained(BERT_PATH)
    
    # 创建数据集
    train_dataset = NERDataset(sample_data, tokenizer, label2id, MAX_LEN)
    # 实际应用中需要创建验证集
    # valid_dataset = NERDataset(valid_data, tokenizer, label2id, MAX_LEN)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)
    
    # 初始化模型
    model = BertBiLSTMCRF(
        bert_path=BERT_PATH,
        num_tags=NUM_TAGS,
        lstm_hidden_size=LSTM_HIDDEN_SIZE,
        lstm_layers=LSTM_LAYERS,
        dropout=DROPOUT
    ).to(DEVICE)
    
    # 优化器和调度器
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )
    
    # 训练循环
    print(f"Using device: {DEVICE}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    best_loss = float('inf')
    
    for epoch in range(EPOCHS):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch+1}/{EPOCHS}")
        print('='*50)
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, DEVICE)
        print(f"Train Loss: {train_loss:.4f}")
        
        # 评估（如果有验证集）
        # valid_preds, valid_labels = evaluate(model, valid_loader, DEVICE, id2label)
        # print(classification_report(valid_labels, valid_preds, digits=4))
        
        # 保存最佳模型
        if train_loss < best_loss:
            best_loss = train_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'best_model.pth')
            print("Saved best model!")
    
    print(f"\nTraining completed! Best loss: {best_loss:.4f}")


# ==================== 6. 推理代码 ====================
def predict(text, model, tokenizer, device, id2label):
    """
    对单条文本进行NER预测
    
    输入原始文本，输出每个词的预测标签。
    
    Args:
        text (str): 输入文本
        model (nn.Module): 训练好的模型
        tokenizer (BertTokenizer): BERT分词器
        device (torch.device): 设备（CPU或GPU）
        id2label (dict): ID到标签的映射字典
        
    Returns:
        list: 预测结果列表，每项为 {'word': 词, 'tag': 标签} 字典
              例如: [{'word': '北京', 'tag': 'B-LOC'}, ...]
    """
    model.eval()
    
    # 分词（中文按字符切分）
    tokens = list(text)
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=128,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    
    with torch.no_grad():
        predictions, _ = model(input_ids, attention_mask)
    
    # 解码
    pred_tags = predictions[0].cpu().numpy()
    word_ids = encoding.word_ids()
    
    results = []
    prev_word_idx = None
    for idx, word_idx in enumerate(word_ids):
        if word_idx is None or word_idx == prev_word_idx:
            continue
        results.append({
            'word': tokens[word_idx],
            'tag': id2label[pred_tags[idx]]
        })
        prev_word_idx = word_idx
    
    return results


if __name__ == '__main__':
    main()