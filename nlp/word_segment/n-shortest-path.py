''' N最短路径分词方法 N-Shortest Path Word Segmentation

核心思想：
    1. 根据词典为句子构建词语网格(有向无环图 DAG)：
       节点 i 表示第 i 个字之前的位置，边 (i -> j) 表示词语 sentence[i:j]。
    2. 每条边赋予权重 = -log P(word)，权重越小代表词越常见(词频越高)。
    3. 一条切分路径的总权重 = 路径上所有词权重之和，最小化总权重等价于最大化切分概率。
    4. 在 DAG 上求前 N 条最短路径：每个节点仅保留累计权重最小的前 N 条候选路径，
       既得到最优切分，又保留多条候选用于后续歧义消解与词性标注。

参考：张华平, 刘群. 《基于N-最短路径方法的中文词语粗分模型》
'''

import math
from collections import defaultdict


class NShortestPathSegmenter:
    def __init__(self, dict_path=None, n=1):
        """
        初始化N最短路径分词器
        :param dict_path: 词典文件路径（每行格式：词 词频），为None则使用内置词典
        :param n: 保留的最短路径数量
        """
        self.n = n
        self.word_freq = defaultdict(int)   # 词 -> 词频
        self.max_word_len = 0               # 最大词长
        self.total_freq = 0                 # 词频总和
        self.graph = {}                     # 节点 -> [(下一节点, 权重, 词), ...]

        if dict_path:
            self.load_dict(dict_path)
        else:
            self._load_default_dict()

    def load_dict(self, path):
        """加载词典文件，每行格式：词 词频"""
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    word, freq = parts[0], int(parts[1])
                    self.word_freq[word] = freq
                    self.max_word_len = max(self.max_word_len, len(word))
        self.total_freq = sum(self.word_freq.values())

    def _load_default_dict(self):
        """加载内置示例词典"""
        default_dict = {
            '研': 100, '究': 100, '生': 200, '命': 100, '的': 5000,
            '起': 100, '源': 100,
            '研究': 800, '研究生': 300, '生命': 600, '起源': 400,
        }
        for word, freq in default_dict.items():
            self.word_freq[word] = freq
            self.max_word_len = max(self.max_word_len, len(word))
        self.total_freq = sum(self.word_freq.values())

    def word_weight(self, word):
        """
        计算词的权重：-log P(word)，权重越小越常见
        未登录词采用拉普拉斯平滑，赋予较小概率（较大权重）
        """
        freq = self.word_freq.get(word, 0)
        if freq > 0:
            prob = freq / self.total_freq
        else:
            # 未登录词平滑：给一个很小的概率
            prob = 1.0 / (self.total_freq + len(self.word_freq) + 1)
        return -math.log(prob)

    def build_graph(self, sentence):
        """
        为句子构建词语网格(DAG)
        节点 i 表示第 i 个字之前的位置，边 (i->j) 表示词语 sentence[i:j]
        单字一定作为候选边加入，保证图从起点到终点连通
        """
        L = len(sentence)
        num_nodes = L + 1
        self.graph = {i: [] for i in range(num_nodes)}

        for i in range(L):
            # 词的结束位置 j，长度不超过 max_word_len 和句子长度
            for j in range(i + 1, min(i + self.max_word_len, L) + 1):
                word = sentence[i:j]
                # 词典中的词，或单字（单字一定作为候选，保证图连通）
                if word in self.word_freq or len(word) == 1:
                    weight = self.word_weight(word)
                    self.graph[i].append((j, weight, word))
        return self.graph

    def n_shortest_paths(self, sentence):
        """
        在DAG上求前N条最短路径
        每个节点仅保留累计权重最小的前N条候选路径（N最短路径算法的核心）
        :return: 列表 [(总权重, [词...]), ...]，按权重升序
        """
        L = len(sentence)
        num_nodes = L + 1
        # paths[node] = [(累计权重, 路径节点列表), ...] 最多保留n条
        paths = [[] for _ in range(num_nodes)]
        paths[0] = [(0.0, [0])]  # 起点：累计权重0，路径仅含节点0

        # 按位置顺序(拓扑序)处理每个节点：边只向后指，前驱节点已全部处理完毕
        for node in range(num_nodes):
            # 当前节点候选路径按权重排序，仅保留前n条
            paths[node].sort(key=lambda x: x[0])
            paths[node] = paths[node][:self.n]

            # 用前n条路径向后续节点扩展
            for cum_weight, path in paths[node]:
                for next_node, edge_weight, word in self.graph.get(node, []):
                    new_weight = cum_weight + edge_weight
                    new_path = path + [next_node]
                    paths[next_node].append((new_weight, new_path))

        # 终点的n条最短路径，将节点路径转换为词序列
        results = []
        for cum_weight, path in paths[num_nodes - 1][:self.n]:
            words = []
            for k in range(len(path) - 1):
                i, j = path[k], path[k + 1]
                words.append(sentence[i:j])
            results.append((cum_weight, words))
        return results

    def segment(self, sentence, n=None):
        """
        对句子分词，返回前n条切分结果
        :param n: 临时指定保留路径数，为None则使用初始化时的n
        :return: 列表 [(总权重, [词...]), ...]
        """
        if n is not None:
            old_n, self.n = self.n, n
        self.build_graph(sentence)
        results = self.n_shortest_paths(sentence)
        if n is not None:
            self.n = old_n
        return results

    def print_dict_stats(self):
        """打印词典统计信息"""
        print(f"词典信息:")
        print(f"  词条数: {len(self.word_freq)}")
        print(f"  最大词长: {self.max_word_len}")
        print(f"  词频总和: {self.total_freq}")

    def print_graph(self, sentence):
        """打印词语网格(DAG)"""
        print("  节点 --[词 (权重)]--> 目标节点:")
        for i in sorted(self.graph):
            if not self.graph[i]:
                continue
            for j, weight, word in self.graph[i]:
                print(f"    {i} --[{word} (w={weight:.3f})]--> {j}")


# ============ 简单案例演示 ============

def demo():
    print("=" * 60)
    print("N最短路径分词方法 演示")
    print("=" * 60)

    # 1. 加载词典
    print("\n【1. 加载词典】")
    segmenter = NShortestPathSegmenter(n=3)
    segmenter.print_dict_stats()

    # 2. 构建词语网格
    sentence = "研究生命的起源"
    print(f"\n【2. 为句子构建词语网格 DAG】")
    print(f"句子: '{sentence}' (长度 {len(sentence)})")
    segmenter.build_graph(sentence)
    segmenter.print_graph(sentence)

    # 3. 词权重示例
    print(f"\n【3. 词权重计算 weight = -log P(word) = -log(freq / total_freq)】")
    print(f"  total_freq = {segmenter.total_freq}")
    for word in ['研究', '研究生', '生命', '命', '的', '起源']:
        w = segmenter.word_weight(word)
        freq = segmenter.word_freq.get(word, 0)
        print(f"  '{word}' 频次={freq:5d}  P={freq/segmenter.total_freq:.4f}  权重={w:.4f}")

    # 4. N最短路径
    print(f"\n【4. 求前 {segmenter.n} 条最短路径】")
    results = segmenter.n_shortest_paths(sentence)
    for rank, (weight, words) in enumerate(results, 1):
        print(f"  第{rank}条: 权重={weight:.4f}  切分: {' / '.join(words)}")

    # 5. 最优切分 (N=1)
    print(f"\n【5. 最优切分 (N=1，即最短路径)】")
    for weight, words in segmenter.segment(sentence, n=1):
        print(f"  {' / '.join(words)}  (权重 {weight:.4f})")

    # 6. 歧义保留演示
    print(f"\n【6. 歧义保留演示】")
    print("N最短路径保留多条候选，用于后续歧义消解与词性标注：")
    print("注意 '研究/生命' 与 '研究生/命' 的交叉歧义被同时保留")
    for n in [1, 2, 3]:
        res = segmenter.segment(sentence, n=n)
        print(f"  N={n}:")
        for weight, words in res:
            print(f"    {' / '.join(words)}  (权重 {weight:.4f})")

    # 7. 最优路径逐词权重分解
    print(f"\n【7. 最优路径逐词权重分解】")
    weight, words = results[0]
    print(f"  切分: {' / '.join(words)}")
    total = 0.0
    for w in words:
        ww = segmenter.word_weight(w)
        total += ww
        print(f"    {w}: -log P({w}) = {ww:.4f}")
    parts = ' + '.join(f"{segmenter.word_weight(w):.4f}" for w in words)
    print(f"  总权重 = {parts} = {total:.4f}")

    # 8. 更多切分示例
    print(f"\n【8. 更多切分示例 (N=3)】")
    examples = ["研究生命", "起源", "的起源"]
    for sent in examples:
        res = segmenter.segment(sent, n=3)
        print(f"  '{sent}':")
        for weight, words in res:
            print(f"    {' / '.join(words)}  (权重 {weight:.4f})")


if __name__ == "__main__":
    demo()
