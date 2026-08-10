import math

class HMM_Segmenter:
    def __init__(self):
        # 状态集合: B (Begin), M (Middle), E (End), S (Single)
        self.states = ['B', 'M', 'E', 'S']
        
        # 初始概率 Vector, 转移概率 Matrix, 发射概率 Matrix
        self.pi = {}
        self.A = {}
        self.B = {}
        
        # 词频计数器（用于极大似然估计）
        self.state_count = {s: 0 for s in self.states}

    def train(self, corpus_path):
        """
        通过标注语料库训练 HMM 参数 (以空格分隔词语，如: "小明 喜欢 看 电影")
        """
        # 初始化概率字典
        for s in self.states:
            self.pi[s] = 0.0
            self.A[s] = {s2: 0.0 for s2 in self.states}
            self.B[s] = {}

        # 1. 统计频数
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                if not words:
                    continue
                
                # 生成 BMES 标注序列
                char_states = []
                for word in words:
                    if len(word) == 1:
                        char_states.append((word[0], 'S'))
                    else:
                        tags = ['B'] + ['M'] * (len(word) - 2) + ['E']
                        for char, tag in zip(word, tags):
                            char_states.append((char, tag))
                
                # 统计初始概率、转移概率、发射概率
                for i, (char, tag) in enumerate(char_states):
                    self.state_count[tag] += 1
                    self.B[tag][char] = self.B[tag].get(char, 0) + 1
                    
                    if i == 0:
                        self.pi[tag] += 1
                    else:
                        prev_tag = char_states[i - 1][1]
                        self.A[prev_tag][tag] += 1

        # 2. 转换为 Log 概率 (取对数防止浮点下溢，并采用拉普拉斯平滑)
        total_sentences = sum(self.pi.values())
        for s in self.states:
            # 初始概率
            self.pi[s] = math.log((self.pi[s] + 1e-6) / total_sentences)
            
            # 转移概率
            s_count = sum(self.A[s].values())
            for s2 in self.states:
                self.A[s][s2] = math.log((self.A[s][s2] + 1e-6) / (s_count + 1e-6 * len(self.states)))
                
            # 发射概率
            b_count = self.state_count[s]
            for char in self.B[s]:
                self.B[s][char] = math.log(self.B[s][char] / b_count)

    def _viterbi(self, text):
        """
        维特比算法：根据观测文本解码出概率最大的 BMES 标注序列
        """
        if not text:
            return []

        V = [{}]  # DP 动态规划表
        path = {}  # 记录最优路径

        # 1. 初始化 (t = 0)
        for s in self.states:
            # 未登录字给出极小默认发射对数概率
            emit_p = self.B[s].get(text[0], -3.14e10)
            V[0][s] = self.pi[s] + emit_p
            path[s] = [s]

        # 2. 递推 (t > 0)
        for t in range(1, len(text)):
            V.append({})
            new_path = {}

            for curr_s in self.states:
                emit_p = self.B[curr_s].get(text[t], -3.14e10)
                
                # 寻找能产生当前状态最大概率的前一状态
                (prob, prev_s) = max(
                    (V[t - 1][prev_s] + self.A[prev_s][curr_s] + emit_p, prev_s)
                    for prev_s in self.states
                )
                
                V[t][curr_s] = prob
                new_path[curr_s] = path[prev_s] + [curr_s]

            path = new_path

        # 3. 终止：选择文本末尾概率最大的状态 (只在 E 或 S 状态中选择更符合语法逻辑)
        (prob, best_state) = max((V[-1][s], s) for s in ['E', 'S'])
        return path[best_state]

    def cut(self, text):
        """
        将概率最大的 BMES 序列转化为分词结果
        """
        tags = self._viterbi(text)
        words = []
        word = ""

        for char, tag in zip(text, tags):
            word += char
            if tag in ['E', 'S']:
                words.append(word)
                word = ""
        
        if word:  # 容错处理
            words.append(word)
            
        return words


# ==================== 测试示例 ====================
if __name__ == '__main__':
    import os

    # 1. 模拟生成一份简单标注语料库文件
    corpus_file = 'train_corpus.txt'
    with open(corpus_file, 'w', encoding='utf-8') as f:
        f.write("小明 喜欢 看 电影 \n")
        f.write("隐马尔可夫模型 在 自然语言处理 中 应用 广泛 \n")
        f.write("条件随机场 也是 一种 序列标注 算法 \n")
        f.write("深度学习 效果 更好 \n")

    # 2. 训练模型
    model = HMM_Segmenter()
    model.train(corpus_file)

    # 3. 预测分词
    test_text = "小明在学习自然语言处理"
    result = model.cut(test_text)

    print("待分词文本:", test_text)
    print("分词结果:  ", " / ".join(result))

    # 清理测试生成的临时文件
    if os.path.exists(corpus_file):
        os.remove(corpus_file)