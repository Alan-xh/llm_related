''' n元语法模型 n-gram Language Model '''

import numpy as np
from collections import defaultdict, Counter
import math

class NGramLM:
    def __init__(self, n=2):
        """
        初始化N-gram语言模型
        :param n: n-gram的阶数，默认bigram
        """
        self.n = n
        self.ngram_counts = defaultdict(Counter)  # 存储n-gram计数
        self.context_counts = Counter()           # 存储上下文计数
        self.vocab = set()
        self.total_tokens = 0
        
    def tokenize(self, text):
        """简单分词，添加开始和结束标记"""
        # 添加开始和结束标记
        tokens = ['<s>'] * (self.n - 1) + text.lower().split() + ['</s>']
        return tokens
    
    def train(self, corpus):
        """
        训练N-gram模型
        :param corpus: 文本语料库，可以是字符串列表或单个字符串
        """
        if isinstance(corpus, str):
            corpus = [corpus]
            
        for sentence in corpus:
            tokens = self.tokenize(sentence)
            self.vocab.update(tokens)
            self.total_tokens += len(tokens)
            
            # 统计n-gram
            for i in range(len(tokens) - self.n + 1):
                # 获取n-gram
                ngram = tuple(tokens[i:i+self.n])
                context = tuple(tokens[i:i+self.n-1])
                
                # 更新计数
                self.ngram_counts[context][ngram[-1]] += 1
                self.context_counts[context] += 1
        print(f"ngram_counts: {self.ngram_counts}")
        print(f"context_counts: {self.context_counts}")
    
    def get_ngram_probability(self, context, word, smoothing=False, alpha=1.0):
        """
        计算条件概率 P(word|context)

        :param context: 上下文（元组或列表）,context 长度为 n-1
        :param word: 目标词
        :param smoothing: 是否使用拉普拉斯平滑
        :param alpha: 平滑参数
        """
        if isinstance(context, str):
            context = (context,)
        elif isinstance(context, list):
            context = tuple(context)
            
        ngram_count = self.ngram_counts[context][word] # 获得前面是 context，后面是 word 的次数
        context_count = self.context_counts[context] # 获得 context 出现的次数
        
        if smoothing:
            # 拉普拉斯平滑
            vocab_size = len(self.vocab)
            return (ngram_count + alpha) / (context_count + alpha * vocab_size)
        else:
            # 无平滑
            if context_count == 0:
                return 0.0
            return ngram_count / context_count
    
    def get_sentence_probability(self, sentence, smoothing=False, alpha=1.0):
        """
        计算整句的概率 P(w1, w2, ..., wn)
        """
        tokens = self.tokenize(sentence)
        prob = 1.0
        
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i+self.n])
            context = tuple(tokens[i:i+self.n-1])
            word = ngram[-1]
            
            p = self.get_ngram_probability(context, word, smoothing, alpha)
            prob *= p
            
        return prob
    
    def get_perplexity(self, sentence, smoothing=False, alpha=1.0):
        """
        计算困惑度 1/n * -log P(w1) - log P(w2) - ... - log P(wn)
        """
        prob = self.get_sentence_probability(sentence, smoothing, alpha)
        tokens = self.tokenize(sentence)
        n = len(tokens) - self.n + 1  # n-gram数量
        
        if prob == 0:
            return float('inf')
        
        return math.pow(prob, -1.0/n)
    
    def generate_next_word(self, context, smoothing=False, alpha=1.0):
        """
        根据上下文生成下一个词（贪婪选择）
        """
        if isinstance(context, str):
            context = (context,)
        elif isinstance(context, list):
            context = tuple(context)
            
        # 获取所有候选词的概率
        word_probs = {}
        for word in self.vocab:
            prob = self.get_ngram_probability(context, word, smoothing, alpha)
            if prob > 0:
                word_probs[word] = prob
        
        if not word_probs:
            return None
            
        # 选择概率最大的词
        return max(word_probs, key=word_probs.get)
    
    def print_ngram_stats(self):
        """打印n-gram统计信息"""
        print(f"模型信息:")
        print(f"  N-gram阶数: {self.n}")
        print(f"  词汇表大小: {len(self.vocab)}")
        print(f"  总词数: {self.total_tokens}")
        print(f"  不同上下文数: {len(self.context_counts)}")
        print(f"  不同N-gram数: {sum(len(c) for c in self.ngram_counts.values())}")


# ============ 简单案例演示 ============

def demo():
    # 1. 准备训练语料
    corpus = [
        "I love natural language processing",
        "I love machine learning",
        "natural language processing is fun",
        "machine learning is fun too"
    ]
    
    for i, sent in enumerate(corpus, 1):
        print(f"{i}. {sent}")
    
    # 2. 训练Bigram模型
    print("训练Bigram模型 (n=2)")
    
    model = NGramLM(n=2)
    model.train(corpus)
    model.print_ngram_stats()
    
    # 3. 查看具体的计数
    print("\n【N-gram计数详情】")
    print("上下文 -> 后续词: 计数")
    print("-" * 40)
    for context, counter in sorted(model.ngram_counts.items()):
        for word, count in counter.items():
            print(f"'{' '.join(context)}' -> '{word}': {count}")
    
    # 4. 计算条件概率
    print("条件概率计算示例: ")
    
    examples = [
        (('i',), 'love'),
        (('love',), 'natural'),
        (('natural',), 'language'),
        (('machine',), 'learning'),
        (('learning',), 'is'),
    ]
    
    print("\n无平滑:")
    for context, word in examples:
        prob = model.get_ngram_probability(context, word, smoothing=False)
        context_str = ' '.join(context)
        print(f"  P({word} | {context_str}) = {prob:.4f}")
    
    print("\n拉普拉斯平滑 (alpha=1.0):")
    for context, word in examples:
        prob = model.get_ngram_probability(context, word, smoothing=True, alpha=1.0)
        context_str = ' '.join(context)
        print(f"  P({word} | {context_str}) = {prob:.4f}")
    
    # 5. 整句概率计算
    print("\n" + "=" * 60)
    print("整句概率与困惑度计算")
    print("=" * 60)
    
    test_sentences = [
        "I love natural language",
        "I love deep learning",  # 包含未见词
        "natural language is fun"
    ]
    
    for sent in test_sentences:
        print(f"\n句子: {sent}")
        
        # 无平滑
        prob = model.get_sentence_probability(sent, smoothing=False)
        ppl = model.get_perplexity(sent, smoothing=False)
        print(f"  无平滑 - 概率: {prob:.6f}, 困惑度: {ppl:.2f}")
        
        # 有平滑
        prob = model.get_sentence_probability(sent, smoothing=True, alpha=1.0)
        ppl = model.get_perplexity(sent, smoothing=True, alpha=1.0)
        print(f"  平滑 - 概率: {prob:.6f}, 困惑度: {ppl:.2f}")
    
    # 6. 文本生成演示
    print("\n" + "=" * 60)
    print("文本生成演示")
    print("=" * 60)
    
    contexts = ['i', 'natural', 'machine', 'learning']
    for context in contexts:
        next_word = model.generate_next_word(context, smoothing=True)
        print(f"上下文 '{context}' -> 下一个词: '{next_word}'")
    
    # 7. 详细计算过程展示
    print("\n" + "=" * 60)
    print("详细计算过程展示")
    print("=" * 60)
    
    prob = model.get_ngram_probability(('i',), 'love', smoothing=False)
    print(f"    P(love|i) = Count(i,love) / Count(i) = {model.ngram_counts[('i',)]['love']} / {model.context_counts[('i',)]} = {prob:.4f}")
    
    print("\n计算整句概率: 'I love natural language'")
    print("  使用链式法则: P(I love natural language)")
    tokens = model.tokenize("I love natural language")
    print(f"  分词: {tokens}")
    print("  P(sentence) = P(love|I) × P(natural|love) × P(language|natural) × P(</s>|language)")
    
    prob = 1.0
    for i in range(len(tokens) - model.n + 1):
        ngram = tuple(tokens[i:i+model.n])
        context = tuple(tokens[i:i+model.n-1])
        word = ngram[-1]
        p = model.get_ngram_probability(context, word, smoothing=True, alpha=1.0)
        prob *= p
        print(f"    P({word}|{' '.join(context)}) = {p:.4f}")
    
    print(f"  总概率 = {prob:.6f}")
    ppl = model.get_perplexity("I love natural language", smoothing=True, alpha=1.0)
    print(f"  困惑度 = {ppl:.2f}")


if __name__ == "__main__":
    demo()