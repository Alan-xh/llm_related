''' n元语法模型 n-gram Language Model

包含：
  - n-gram 计数与最大似然估计 (MLE)
  - 未登录词 <UNK> 处理（低频词与测试期未见词统一映射为 <unk>）
  - 平滑方法：拉普拉斯(add-k)、线性插值、Stupid Backoff
  - 困惑度 (Perplexity，对数空间计算避免下溢)
  - 文本生成（贪婪 / 温度采样 / top-k 采样）
  - 模型保存与加载
'''

import math
import random
import pickle
from collections import defaultdict, Counter


class NGramLM:
    def __init__(self, n=2, unk_threshold=1):
        """
        初始化N-gram语言模型
        :param n: n-gram的阶数，默认bigram
        :param unk_threshold: 训练语料中出现次数 <= 该阈值的词映射为 <unk>
        """
        self.n = n
        self.unk_threshold = unk_threshold
        # counts[order] = {context: Counter(word: count)}，order 取 1..n
        # order=k 表示用长度 k-1 的上下文预测下一个词；存储所有低阶便于插值/回退
        self.counts = {k: defaultdict(Counter) for k in range(1, n + 1)}
        self.totals = {k: Counter() for k in range(1, n + 1)}  # context -> 该上下文总计数
        self.vocab = set()
        self.total_tokens = 0
        # 插值平滑权重，默认各阶均匀；sum(lambdas) == 1
        self.lambdas = {k: 1.0 / n for k in range(1, n + 1)}
        # Stupid Backoff 的回退折扣系数
        self.discount = 0.4

    # ============ 预处理 ============

    def _build_vocab(self, corpus):
        """统计词频，构建词表（低频词归入 <unk>）"""
        raw = Counter()
        for sentence in corpus:
            for w in sentence.lower().split():
                raw[w] += 1
        self.vocab = {'<s>', '</s>', '<unk>'}
        for w, c in raw.items():
            if c > self.unk_threshold:        # 出现次数足够多才保留
                self.vocab.add(w)

    def tokenize(self, text):
        """分词：小写化 + 未登录词替换为 <unk> + 添加首尾标记"""
        words = text.lower().split()
        words = [w if w in self.vocab else '<unk>' for w in words]
        return ['<s>'] * (self.n - 1) + words + ['</s>']

    def train(self, corpus):
        """
        训练N-gram模型：统计 1~n 阶的所有 n-gram 计数
        :param corpus: 文本语料库，字符串列表或单个字符串
        """
        if isinstance(corpus, str):
            corpus = [corpus]
        self._build_vocab(corpus)

        for sentence in corpus:
            tokens = self.tokenize(sentence)
            self.total_tokens += len(tokens)
            # 统计每个阶数的 n-gram
            for order in range(1, self.n + 1):
                for i in range(len(tokens) - order + 1):
                    ngram = tuple(tokens[i:i + order])
                    context, word = ngram[:-1], ngram[-1]
                    self.counts[order][context][word] += 1
                    self.totals[order][context] += 1

    # ============ 概率计算 ============

    @staticmethod
    def _normalize_context(context):
        if isinstance(context, str):
            return (context,)
        if isinstance(context, list):
            return tuple(context)
        return context

    def _mle(self, context, word, order):
        """order 阶最大似然估计 P(word|context)，context 长度应为 order-1"""
        total = self.totals[order].get(context, 0)
        if total == 0:
            return 0.0
        return self.counts[order][context].get(word, 0) / total

    def _context_for_order(self, context, order):
        """取 context 末尾 order-1 个词作为该阶上下文"""
        if order == 1:
            return ()
        return context[-(order - 1):]

    def get_ngram_probability(self, context, word, method='none', alpha=1.0):
        """
        计算条件概率 P(word|context)
        :param context: 上下文（元组/列表/字符串），长度为 n-1
        :param word: 目标词
        :param method: 'none'(MLE) | 'laplace'(加alpha平滑) | 'interpolation'(线性插值) | 'backoff'(Stupid Backoff)
        :param alpha: 拉普拉斯平滑参数
        """
        context = self._normalize_context(context)
        if len(context) > self.n - 1:                       # 过长上下文截断到 n-1
            context = context[-(self.n - 1):] if self.n > 1 else ()

        if method == 'laplace':
            ctx = self._context_for_order(context, self.n)
            count = self.counts[self.n][ctx].get(word, 0)
            total = self.totals[self.n].get(ctx, 0)
            V = len(self.vocab)
            return (count + alpha) / (total + alpha * V)
        if method == 'interpolation':
            return self._prob_interpolation(context, word)
        if method == 'backoff':
            return self._prob_backoff(context, word)
        # 默认 MLE（无平滑）
        ctx = self._context_for_order(context, self.n)
        return self._mle(ctx, word, self.n)

    def _prob_interpolation(self, context, word):
        """线性插值：P = Σ λ_k * P_MLE_k(word | context 的末 k-1 个词)"""
        prob = 0.0
        for order in range(1, self.n + 1):
            ctx = self._context_for_order(context, order)
            prob += self.lambdas[order] * self._mle(ctx, word, order)
        return prob

    def _prob_backoff(self, context, word):
        """Stupid Backoff：高阶命中则返回 MLE，否则乘 discount 回退到低阶"""
        weight = 1.0
        for order in range(self.n, 0, -1):
            ctx = self._context_for_order(context, order)
            total = self.totals[order].get(ctx, 0)
            if total > 0:
                count = self.counts[order][ctx].get(word, 0)
                if count > 0:
                    return weight * count / total
            weight *= self.discount        # 回退一次，后续结果乘以折扣
        return 0.0                          # 各阶均未见过该词

    def _sentence_log_prob(self, sentence, method='none', alpha=1.0):
        """计算整句对数概率，返回 (log_prob, n-gram数)；对数空间避免下溢"""
        tokens = self.tokenize(sentence)
        log_prob, num = 0.0, 0
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context, word = ngram[:-1], ngram[-1]
            p = self.get_ngram_probability(context, word, method, alpha)
            p = max(p, 1e-12)               # 避免 log(0)
            log_prob += math.log(p)
            num += 1
        return log_prob, num

    def get_sentence_probability(self, sentence, method='none', alpha=1.0):
        """计算整句概率 P(w1, w2, ..., wn)"""
        log_prob, _ = self._sentence_log_prob(sentence, method, alpha)
        return math.exp(log_prob) if log_prob > -700 else 0.0

    def get_perplexity(self, sentence, method='none', alpha=1.0):
        """计算困惑度 PPL = exp(-1/N * Σ log P)"""
        log_prob, num = self._sentence_log_prob(sentence, method, alpha)
        if num == 0:
            return float('inf')
        return math.exp(-log_prob / num)

    # ============ 文本生成 ============

    def _candidate_probs(self, context, method='none', alpha=1.0):
        """计算上下文后所有候选词的概率，返回 {word: prob}"""
        context = self._normalize_context(context)
        if len(context) > self.n - 1:
            context = context[-(self.n - 1):] if self.n > 1 else ()
        probs = {}
        for word in self.vocab:
            if word == '<s>':               # <s> 不会作为下一个词出现
                continue
            p = self.get_ngram_probability(context, word, method, alpha)
            if p > 0:
                probs[word] = p
        return probs

    def generate_next_word(self, context, method='none', alpha=1.0):
        """根据上下文贪婪生成下一个词（概率最大）"""
        probs = self._candidate_probs(context, method, alpha)
        return max(probs, key=probs.get) if probs else None

    def generate_sentence(self, start='', max_words=20, method='none', alpha=1.0,
                          temperature=0.0, top_k=0):
        """
        生成完整句子，遇到 </s> 或达到 max_words 停止
        :param start: 起始文本
        :param temperature: 0=贪婪；>0 按温度采样（越大越随机）
        :param top_k: >0 时仅从概率最高的 top_k 个词中采样
        :return: 生成的词列表
        """
        tokens = ['<s>'] * (self.n - 1) + start.lower().split()
        for _ in range(max_words):
            context = tuple(tokens[-(self.n - 1):]) if self.n > 1 else ()
            probs = self._candidate_probs(context, method, alpha)
            if not probs:
                break
            if temperature <= 0:
                next_word = max(probs, key=probs.get)
            else:
                items = sorted(probs.items(), key=lambda x: -x[1])
                if top_k > 0:
                    items = items[:top_k]
                weights = [p ** (1.0 / temperature) for _, p in items]
                total = sum(weights)
                next_word = random.choices([w for w, _ in items],
                                           weights=[w / total for w in weights])[0]
            if next_word == '</s>':
                break
            tokens.append(next_word)
        return tokens[self.n - 1:]

    # ============ 工具方法 ============

    def set_lambdas(self, lambdas):
        """设置插值权重，lambdas 为长度 n 的列表，对应 1..n 阶（自动归一化）"""
        if len(lambdas) != self.n:
            raise ValueError(f"需要 {self.n} 个权重，收到 {len(lambdas)} 个")
        s = sum(lambdas)
        self.lambdas = {k: lambdas[k - 1] / s for k in range(1, self.n + 1)}

    def print_ngram_stats(self):
        """打印n-gram统计信息"""
        print(f"模型信息:")
        print(f"  N-gram阶数: {self.n}")
        print(f"  词汇表大小: {len(self.vocab)}")
        print(f"  总词数(含标记): {self.total_tokens}")
        for order in range(1, self.n + 1):
            n_ngrams = sum(len(c) for c in self.counts[order].values())
            print(f"  {order}-gram数: {n_ngrams}")

    def save(self, path):
        """保存模型到文件"""
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, path):
        """从文件加载模型"""
        with open(path, 'rb') as f:
            state = pickle.load(f)
        model = cls.__new__(cls)
        model.__dict__.update(state)
        return model


# ============ 简单案例演示 ============

def demo():
    # 1. 准备训练语料
    corpus = [
        "I love natural language processing",
        "I love machine learning",
        "natural language processing is fun",
        "machine learning is fun too"
    ]

    print("训练语料:")
    for i, sent in enumerate(corpus, 1):
        print(f"  {i}. {sent}")

    # 2. 训练Bigram模型
    print("\n训练Bigram模型 (n=2, unk_threshold=1)")
    model = NGramLM(n=2, unk_threshold=1)
    model.train(corpus)
    model.print_ngram_stats()

    # 3. <UNK> 处理说明
    print("\n【<UNK> 处理】")
    print(f"  词表: {sorted(model.vocab)}")
    print(f"  注: 语料中 'too' 仅出现 1 次 (<=阈值), 被映射为 <unk>")
    print(f"  测试期未见词 (如 'deep') 同样映射为 <unk>，从而获得非零概率")

    # 4. 查看Bigram计数详情
    print("\n【Bigram 计数详情】")
    print("  上下文 -> 后续词: 计数")
    print("  " + "-" * 40)
    for context, counter in sorted(model.counts[2].items()):
        for word, count in counter.items():
            print(f"  '{' '.join(context)}' -> '{word}': {count}")

    # 5. 条件概率：四种方法对比
    print("\n" + "=" * 60)
    print("条件概率计算示例 (四种方法对比)")
    print("=" * 60)

    examples = [
        (('i',), 'love'),
        (('love',), 'natural'),
        (('love',), '<unk>'),       # 训练期未见该转移
        (('<unk>',), 'learning'),   # 训练期未见该转移
    ]
    methods = ['none', 'laplace', 'interpolation', 'backoff']

    print(f"\n  {'P(word|context)':<28}", end='')
    for m in methods:
        print(f"{m:>14}", end='')
    print()
    print("  " + "-" * 84)
    for context, word in examples:
        ctx_str = ' '.join(context)
        label = f"P({word}|{ctx_str})"
        print(f"  {label:<28}", end='')
        for m in methods:
            prob = model.get_ngram_probability(context, word, method=m)
            print(f"{prob:>14.4f}", end='')
        print()
    print("  注: MLE 对未见转移返回 0；平滑方法赋予非零概率")

    # 6. 整句概率与困惑度
    print("\n" + "=" * 60)
    print("整句概率与困惑度 (对数空间计算，避免下溢)")
    print("=" * 60)

    test_sentences = [
        "I love natural language",   # (language, </s>) 未见
        "I love deep learning",      # 'deep' 未见 -> <unk>
        "natural language is fun",
    ]

    for sent in test_sentences:
        print(f"\n  句子: {sent}  ->  tokenize: {model.tokenize(sent)}")
        for m in methods:
            prob = model.get_sentence_probability(sent, method=m)
            ppl = model.get_perplexity(sent, method=m)
            print(f"    {m:<13} 概率: {prob:.2e}, 困惑度: {ppl:.2f}")

    # 7. 文本生成
    print("\n" + "=" * 60)
    print("文本生成演示")
    print("=" * 60)

    print("\n  贪婪生成 (temperature=0):")
    for start in ['i', 'natural', 'machine']:
        words = model.generate_sentence(start=start, method='interpolation', max_words=10)
        print(f"    '{start}' -> {' '.join(words)}")

    print("\n  温度采样生成 (temperature=1.5, top_k=3):")
    random.seed(42)
    for _ in range(3):
        words = model.generate_sentence(start='i', method='interpolation',
                                        temperature=1.5, top_k=3, max_words=10)
        print(f"    {' '.join(words)}")

    # 8. 插值权重调整
    print("\n" + "=" * 60)
    print("插值权重对困惑度的影响")
    print("=" * 60)
    sent = "I love deep learning"
    for lam in [(1.0, 0.0), (0.7, 0.3), (0.5, 0.5), (0.3, 0.7), (0.0, 1.0)]:
        model.set_lambdas(list(lam))   # (unigram权重, bigram权重)
        ppl = model.get_perplexity(sent, method='interpolation')
        print(f"  λ(unigram)={lam[0]:.1f}, λ(bigram)={lam[1]:.1f} -> PPL={ppl:.2f}")
    model.set_lambdas([0.5, 0.5])      # 恢复

    # 9. 详细计算过程展示
    print("\n" + "=" * 60)
    print("详细计算过程展示")
    print("=" * 60)

    print("\n  计算 P(love|i) [MLE]:")
    ctx, word = ('i',), 'love'
    ngram_count = model.counts[2][ctx].get(word, 0)
    context_count = model.totals[2].get(ctx, 0)
    print(f"    P(love|i) = Count(i,love) / Count(i) = {ngram_count} / {context_count} = {ngram_count/context_count:.4f}")

    print("\n  计算整句概率 'I love natural language' [Laplace]:")
    tokens = model.tokenize("I love natural language")
    print(f"    分词: {tokens}")
    print("    P = P(i|<s>) × P(love|i) × P(natural|love) × P(language|natural) × P(</s>|language)")
    log_prob = 0.0
    for i in range(len(tokens) - model.n + 1):
        ngram = tuple(tokens[i:i + model.n])
        context, word = tuple(ngram[:-1]), ngram[-1]
        p = model.get_ngram_probability(context, word, method='laplace', alpha=1.0)
        log_prob += math.log(p)
        print(f"      P({word}|{' '.join(context)}) = {p:.4f}")
    print(f"    总概率 = {math.exp(log_prob):.6e}")
    print(f"    困惑度 = {model.get_perplexity('I love natural language', method='laplace'):.2f}")

    # 10. 模型保存与加载
    print("\n" + "=" * 60)
    print("模型保存与加载")
    print("=" * 60)
    import os, tempfile
    tmp = tempfile.mktemp(suffix='.pkl')
    model.save(tmp)
    loaded = NGramLM.load(tmp)
    os.remove(tmp)
    print(f"  原模型 PPL(I love deep learning) = {model.get_perplexity('I love deep learning', method='interpolation'):.4f}")
    print(f"  加载后 PPL(I love deep learning) = {loaded.get_perplexity('I love deep learning', method='interpolation'):.4f}")
    print(f"  词汇表一致: {model.vocab == loaded.vocab}")


if __name__ == "__main__":
    demo()
