"""
基于 NumPy 从零实现的中文 Word2Vec 训练脚本。

功能：
    1. 使用 jieba 对原始中文语料进行分词预处理；
    2. 构建词表、负采样表与子采样概率；
    3. 分别实现并训练 Skip-gram 与 CBOW 两种 Word2Vec 模型（带负采样）；
    4. 保存模型参数（.npz）与词向量文本格式（.vector）；
    5. 提供相似词查询与词向量类比评估功能。

输入：
    - raw_corpus.txt：原始中文语料（每行一句）；
    - corpus.txt：分词后的语料（若不存在则由 raw_corpus.txt 自动生成）。

输出：
    - word2vec_sg.npz / word2vec_cbow.npz：模型参数文件；
    - word2vec_sg.npz.vector / word2vec_cbow.npz.vector：词向量文本文件；
    - 控制台打印训练损失与评估结果。
"""

import jieba
import numpy as np
import os
from pathlib import Path
from typing import List, Dict
import random
from collections import Counter

# ================== 配置参数 ==================
file_path = Path(__file__).resolve()
file_dir = file_path.parent
CORPUS_PATH = file_dir / 'corpus.txt'          # 分词语料文件
RAW_CORPUS_PATH = file_dir / 'raw_corpus.txt'  # 原始语料文件
MODEL_PATH_SG = file_dir / 'word2vec_sg.npz'   # Skip-gram 模型保存路径
MODEL_PATH_CBOW = file_dir / 'word2vec_cbow.npz'  # CBOW 模型保存路径

VECTOR_SIZE = 100                   # 词向量维度
WINDOW = 5                          # 上下文窗口大小
MIN_COUNT = 5                       # 最小词频
EPOCHS = 10                         # 训练轮数
NEGATIVE = 5                        # 负采样数量
LEARNING_RATE = 0.025               # 初始学习率
SEED = 42
ARCH = 'skipgram'                   # 'skipgram' 或 'cbow'


# ---------------------- 1. 分词预处理 ----------------------
def preprocess_corpus(input_path, output_path):
    """
    使用 jieba 对中文语料分词，并将结果写入输出文件。

    输入：
        input_path (str | Path)：原始中文语料路径，每行一句；
        output_path (str | Path)：分词结果输出路径，词之间以空格分隔。

    输出：
        None。结果写入 output_path，并在控制台打印完成信息。
    """
    with open(input_path, 'r', encoding='utf-8') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            words = jieba.lcut(line)
            fout.write(' '.join(words) + '\n')
    print(f"分词完成，已保存至 {output_path}")


# ---------------------- 2. 读取语料 ----------------------
def load_corpus(path):
    """
    读取分词后的语料文件，返回按句子组织的词列表。

    输入：
        path (str | Path)：分词后的语料文件路径，每行以空格分隔的词序列。

    输出：
        sentences (List[List[str]])：句子列表，每个句子为词列表。
    """
    sentences = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            sentences.append(line.split())
    return sentences


# ---------------------- 3. 构建词表 ----------------------
def build_vocab(sentences, min_count):
    """
    根据语料构建词表，过滤低频词并按词频降序排列。

    输入：
        sentences (List[List[str]])：句子列表；
        min_count (int)：最小词频阈值，低于该值的词被丢弃。

    输出：
        word2id (Dict[str, int])：词到 id 的映射；
        id2word (Dict[int, str])：id 到词的映射；
        freq (np.ndarray)：按词表顺序排列的词频数组（float64）。
    """
    counter = Counter()
    for sent in sentences:
        counter.update(sent)
    words = [w for w, c in counter.items() if c >= min_count]
    words.sort(key=lambda w: -counter[w])
    word2id = {w: i for i, w in enumerate(words)}
    id2word = {i: w for w, i in word2id.items()}
    freq = np.array([counter[w] for w in words], dtype=np.float64)
    print(f"词表大小：{len(word2id)}（原始词数：{len(counter)}）")
    return word2id, id2word, freq


# ---------------------- 4. 负采样表 ----------------------
def build_negative_table(freq, table_size=1_000_000, power=0.75):
    """
    构建 table_size 大小的采样表
    按词频的 power 次方作为采样概率，按照 (f^075 / Σf^0.75) 权重作为概率采样。

    输入：
        freq (np.ndarray)：词频数组；
        table_size (int)：采样表长度；
        power (float)：平滑指数，通常取 0.75。

    输出：
        neg_table (np.ndarray)：长度为 table_size 的 id 数组，按概率分布采样。
    """
    weights = freq ** power
    weights = weights / weights.sum()
    return np.random.choice(len(weights), size=table_size, p=weights)


# ---------------------- 5. 子采样概率 ----------------------
def build_subsample_probs(freq, total, sample=1e-3):
    """
    计算每个词的保留概率（子采样）(sqrt(f/t) + 1) * (t/f) 
    高频词（"的"、"the"、"a"）提供的信息量很少，但在语料中出现次数极多，会拖慢训练速度。
    1. 词频远大于阈值（高频词，如 "the"）大概率被丢弃;
    2. 词频小于等于阈值则全部保留;


    输入：
        freq (np.ndarray)：词频数组；
        total (int | float)：语料总词数；
        sample (float)：子采样阈值，默认 1e-3。

    输出：
        p_keep (np.ndarray)：每个词的保留概率，范围 [0, 1]。
    """
    f = freq / total
    p_keep = (np.sqrt(f / sample) + 1) * (sample / f)
    return np.clip(p_keep, 0.0, 1.0)


# ---------------------- 6. 生成训练样本 ----------------------
def generate_skipgram_pairs(sentences: List[List[str]], word2id:Dict[str, int], p_keep:np.ndarray, window: int):
    """
    Skip-gram：生成 (中心词, 上下文词) 训练对。

    输入：
        sentences (List[List[str]])：句子列表；
        word2id (Dict[str, int])：词到 id 映射；
        p_keep (np.ndarray)：每个词的保留概率；
        window (int)：最大窗口大小，实际窗口在 [1, window] 随机。

    输出：
        pairs (List[Tuple[int, int]])：(center, context) 训练对列表。
    """
    pairs = []
    for sent in sentences:
        ids = [word2id[w] for w in sent if w in word2id]
        if len(ids) < 2:
            continue
        ids = [i for i in ids if np.random.rand() < p_keep[i]] # 根据保留概率保留部分 tokens
        if len(ids) < 2:
            continue
        for i, center in enumerate(ids):
            win = np.random.randint(1, window + 1) # 随机选取一个窗口大小
            start = max(0, i - win) # 窗口开始索引
            end = min(len(ids), i + win + 1) # 窗口结束索引
            for j in range(start, end): # 词队
                if j != i:
                    pairs.append((center, ids[j]))
    return pairs


def generate_cbow_pairs(sentences, word2id, p_keep, window):
    """
    CBOW：生成 (上下文词列表, 中心词) 训练对。

    输入：
        sentences (List[List[str]])：句子列表；
        word2id (Dict[str, int])：词到 id 映射；
        p_keep (np.ndarray)：每个词的保留概率；
        window (int)：最大窗口大小，实际窗口在 [1, window] 随机。

    输出：
        pairs (List[Tuple[List[int], int]])：(context_list, center) 训练对列表。
    """
    pairs = []
    for sent in sentences:
        ids = [word2id[w] for w in sent if w in word2id]
        if len(ids) < 3:
            continue
        ids = [i for i in ids if np.random.rand() < p_keep[i]]
        if len(ids) < 3:
            continue
        for i, center in enumerate(ids):
            win = np.random.randint(1, window + 1)
            start = max(0, i - win)
            end = min(len(ids), i + win + 1)
            ctx = [ids[j] for j in range(start, end) if j != i]
            if ctx:
                pairs.append((ctx, center))
    return pairs


# ---------------------- 7. 初始化 ----------------------
def init_weights(vocab_size, dim, seed=42):
    """
    初始化输入/输出词向量矩阵

    输入：
        vocab_size (int)：词表大小；
        dim (int)：词向量维度；
        seed (int)：随机种子。

    输出：
        W_in (np.ndarray)：输入词向量矩阵，形状 (vocab_size, dim)；
        W_out (np.ndarray)：输出词向量矩阵，形状 (vocab_size, dim)，初始为 0。
    """
    rng = np.random.default_rng(seed)
    W_in = (rng.random((vocab_size, dim)) - 0.5) / dim
    W_out = np.zeros((vocab_size, dim), dtype=np.float64)
    return W_in, W_out


def sigmoid(x):
    """
    Sigmoid 激活函数，带数值裁剪防止溢出。

    输入：
        x (np.ndarray | float)：输入值。

    输出：
        (np.ndarray | float)：sigmoid 结果，范围 (0, 1)。
    """
    x = np.clip(x, -30, 30)
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------- 8. 训练 Skip-gram ----------------------
def train_skipgram(sentences: List[List[str]], 
                   word2id: Dict[int, str], 
                   id2word: Dict[str, int], 
                   freq: np.ndarray, 
                   model_path: Path):
    """
    使用负采样训练 Skip-gram 模型。

    输入：
        sentences (List[List[str]])：句子列表；
        word2id (Dict[str, int])：词到 id 映射；
        id2word (Dict[int, str])：id 到词映射； 
        freq (np.ndarray)：词频数组；
        model_path (str | Path)：模型保存路径。

    输出：
        W_in (np.ndarray)：训练后的输入词向量矩阵；
        word2id (Dict[str, int])：词到 id 映射；
        id2word (Dict[int, str])：id 到词映射。
    """
    np.random.seed(SEED)
    random.seed(SEED)
    vocab_size = len(word2id)
    total_tokens = freq.sum()

    p_keep = build_subsample_probs(freq, total_tokens) # 词保留概率
    neg_table = build_negative_table(freq) # 采样表

    print("正在生成 Skip-gram 训练对...")
    pairs = generate_skipgram_pairs(sentences, word2id, p_keep, WINDOW)
    print(f"共生成 {len(pairs)} 个训练对")

    W_in, W_out = init_weights(vocab_size, VECTOR_SIZE, SEED)

    total_pairs = len(pairs)
    for epoch in range(EPOCHS):
        np.random.shuffle(pairs)
        loss_sum = 0.0
        lr = LEARNING_RATE
        for step, (center, context) in enumerate(pairs):
            v_c = W_in[center] # 获取中心词汇词向量
            u_o = W_out[context] # 获得上下文词汇词向量
            pred = sigmoid(np.dot(v_c, u_o)) # 点乘计算两个 token 相似度再进行香农函数激活
            g = (pred - 1.0) * lr # 计算梯度

            grad_c = g * u_o
            W_out[context] -= g * v_c
            W_in[center]    -= grad_c
            loss_sum += -np.log(pred + 1e-9)

            neg_ids = neg_table[np.random.randint(0, len(neg_table), NEGATIVE)]
            for neg in neg_ids:
                if neg == context:
                    continue
                u_neg = W_out[neg]
                pred_n = sigmoid(np.dot(v_c, u_neg))
                g_n = (pred_n - 0.0) * lr

                grad_c_n = g_n * u_neg
                W_out[neg] -= g_n * v_c
                W_in[center] -= grad_c_n
                loss_sum += -np.log(1.0 - pred_n + 1e-9)

            # 学习率线性衰减
            lr = max(LEARNING_RATE * (1.0 - (epoch * total_pairs + step) /
                                      (EPOCHS * total_pairs)), 1e-4)

        print(f"[Skip-gram] Epoch {epoch + 1}/{EPOCHS} 平均损失：{loss_sum / total_pairs:.4f}")

    save_model(model_path, W_in, W_out, word2id, id2word)
    return W_in, word2id, id2word


# ---------------------- 9. 训练 CBOW ----------------------
def train_cbow(sentences, word2id, id2word, freq, model_path):
    """
    使用负采样训练 CBOW 模型。

    说明：
        用上下文词向量的平均值预测中心词；
        负采样时，正样本是中心词，负样本随机采样。

    输入：
        sentences (List[List[str]])：句子列表；
        word2id (Dict[str, int])：词到 id 映射；
        id2word (Dict[int, str])：id 到词映射；
        freq (np.ndarray)：词频数组；
        model_path (str | Path)：模型保存路径。

    输出：
        W_in (np.ndarray)：训练后的输入词向量矩阵；
        word2id (Dict[str, int])：词到 id 映射；
        id2word (Dict[int, str])：id 到词映射。
    """
    np.random.seed(SEED)
    random.seed(SEED)
    vocab_size = len(word2id)
    total_tokens = freq.sum()

    p_keep = build_subsample_probs(freq, total_tokens)
    neg_table = build_negative_table(freq)

    print("正在生成 CBOW 训练对...")
    pairs = generate_cbow_pairs(sentences, word2id, p_keep, WINDOW)
    print(f"共生成 {len(pairs)} 个训练对")

    W_in, W_out = init_weights(vocab_size, VECTOR_SIZE, SEED)

    total_pairs = len(pairs)
    for epoch in range(EPOCHS):
        np.random.shuffle(pairs)
        loss_sum = 0.0
        lr = LEARNING_RATE
        for step, (ctx_ids, center) in enumerate(pairs):
            # ---- 上下文向量：取平均（投影层） ----
            ctx_vecs = W_in[ctx_ids]                # (C, dim)
            v_avg = ctx_vecs.mean(axis=0)           # (dim,)

            # ---- 正样本：中心词 ----
            u_o = W_out[center]
            pred = sigmoid(np.dot(v_avg, u_o))
            g = (pred - 1.0) * lr

            # 梯度：对输出向量和上下文向量
            grad_v = g * u_o
            W_out[center] -= g * v_avg
            loss_sum += -np.log(pred + 1e-9)

            # 平均池化的梯度：均摊到每个上下文词
            grad_per_ctx = grad_v / len(ctx_ids)

            # ---- 负采样 ----
            neg_ids = neg_table[np.random.randint(0, len(neg_table), NEGATIVE)]
            for neg in neg_ids:
                if neg == center:
                    continue
                u_neg = W_out[neg]
                pred_n = sigmoid(np.dot(v_avg, u_neg))
                g_n = (pred_n - 0.0) * lr

                grad_v_n = g_n * u_neg
                W_out[neg] -= g_n * v_avg
                grad_per_ctx += grad_v_n / len(ctx_ids)
                loss_sum += -np.log(1.0 - pred_n + 1e-9)

            # ---- 更新上下文词向量 ----
            for cid in ctx_ids:
                W_in[cid] -= grad_per_ctx

            lr = max(LEARNING_RATE * (1.0 - (epoch * total_pairs + step) /
                                      (EPOCHS * total_pairs)), 1e-4)

        print(f"[CBOW] Epoch {epoch + 1}/{EPOCHS} 平均损失：{loss_sum / total_pairs:.4f}")

    save_model(model_path, W_in, W_out, word2id, id2word)
    return W_in, word2id, id2word


# ---------------------- 10. 保存 ----------------------
def save_model(model_path, W_in, W_out, word2id, id2word):
    """
    保存模型参数与词向量文本文件。

    输入：
        model_path (str | Path)：模型保存路径（.npz）；
        W_in (np.ndarray)：输入词向量矩阵；
        W_out (np.ndarray)：输出词向量矩阵；
        word2id (Dict[str, int])：词到 id 映射；
        id2word (Dict[int, str])：id 到词映射。

    输出：
        None。生成 model_path（.npz）与 model_path + '.vector' 两个文件。
    """
    np.savez(
        model_path,
        W_in=W_in,
        W_out=W_out,
        words=np.array(list(word2id.keys()), dtype=object),
    )
    print(f"模型已保存至 {model_path}")

    with open(model_path.with_suffix('.vector'), 'w', encoding='utf-8') as f:
        f.write(f"{len(word2id)} {W_in.shape[1]}\n")
        for i in range(len(word2id)):
            vec = ' '.join(f"{x:.6f}" for x in W_in[i])
            f.write(f"{id2word[i]} {vec}\n")
    print(f"词向量文本格式已保存至 {model_path}.vector")


# ---------------------- 11. 评估 ----------------------
def most_similar(word, W_in, word2id, id2word, topn=5):
    """
    查找与给定词最相似的 topn 个词（余弦相似度）。

    输入：
        word (str)：查询词；
        W_in (np.ndarray)：输入词向量矩阵；
        word2id (Dict[str, int])：词到 id 映射；
        id2word (Dict[int, str])：id 到词映射；
        topn (int)：返回的相似词数量。

    输出：
        List[Tuple[str, float]] | None：相似词及相似度列表；
        若词不在词表中则返回 None。
    """
    if word not in word2id:
        return None
    idx = word2id[word]
    vec = W_in[idx]
    norms = np.linalg.norm(W_in, axis=1) + 1e-9
    sims = (W_in @ vec) / (norms * np.linalg.norm(vec) + 1e-9)
    sims[idx] = -np.inf
    top_idx = np.argsort(-sims)[:topn]
    return [(id2word[i], float(sims[i])) for i in top_idx]


def analogy(pos1, pos2, neg1, W_in, word2id, id2word, topn=3):
    """
    词向量类比：pos1 - neg1 + pos2，返回最相似的 topn 个词。

    输入：
        pos1 (str)：正例词 1；
        pos2 (str)：正例词 2；
        neg1 (str)：负例词；
        W_in (np.ndarray)：输入词向量矩阵；
        word2id (Dict[str, int])：词到 id 映射；
        id2word (Dict[int, str])：id 到词映射；
        topn (int)：返回的候选词数量。

    输出：
        List[Tuple[str, float]] | str：相似词及相似度列表；
        若缺少词汇则返回提示字符串。
    """
    for w in (pos1, pos2, neg1):
        if w not in word2id:
            return f"缺少词汇: {w}"
    vec = W_in[word2id[pos1]] - W_in[word2id[neg1]] + W_in[word2id[pos2]]
    norms = np.linalg.norm(W_in, axis=1) + 1e-9
    sims = (W_in @ vec) / (norms * np.linalg.norm(vec) + 1e-9)
    for w in (pos1, pos2, neg1):
        sims[word2id[w]] = -np.inf
    top_idx = np.argsort(-sims)[:topn]
    return [(id2word[i], float(sims[i])) for i in top_idx]


def evaluate_model(W_in, word2id, id2word, tag=''):
    """
    对模型进行简单评估：相似词查询与类比测试。

    输入：
        W_in (np.ndarray)：输入词向量矩阵；
        word2id (Dict[str, int])：词到 id 映射；
        id2word (Dict[int, str])：id 到词映射；
        tag (str)：评估标签，用于打印区分不同模型。

    输出：
        None。评估结果打印到控制台。
    """
    print(f"\n===== 模型评估 {tag} =====")
    for word in ['中国', '科技', '发展']:
        res = most_similar(word, W_in, word2id, id2word, topn=5)
        if res is None:
            print(f"「{word}」不在词汇表中")
        else:
            print(f"与「{word}」最相似的词：{res}")

    try:
        result = analogy('国王', '女人', '男人', W_in, word2id, id2word, topn=3)
        print(f"国王 - 男人 + 女人 = {result}")
    except Exception as e:
        print(f"类比测试失败: {e}")


if __name__ == '__main__':
    # 步骤1：分词
    if not os.path.exists(CORPUS_PATH):
        print(f"未找到分词语料 {CORPUS_PATH}，尝试对原始语料分词...")
        preprocess_corpus(RAW_CORPUS_PATH, CORPUS_PATH)

    # 步骤2：读取语料并构建词表（两者共享，避免重复）
    sentences = load_corpus(CORPUS_PATH)
    print(f"共读取 {len(sentences)} 个句子")
    word2id, id2word, freq = build_vocab(sentences, MIN_COUNT)

    # 步骤3：训练 Skip-gram
    W_in_sg, w2i_sg, i2w_sg = train_skipgram(
        sentences, word2id, id2word, freq, MODEL_PATH_SG
    )
    evaluate_model(W_in_sg, w2i_sg, i2w_sg, tag='Skip-gram')

    # 步骤4：训练 CBOW
    W_in_cbow, w2i_cbow, i2w_cbow = train_cbow(
        sentences, word2id, id2word, freq, MODEL_PATH_CBOW
    )
    evaluate_model(W_in_cbow, w2i_cbow, i2w_cbow, tag='CBOW')