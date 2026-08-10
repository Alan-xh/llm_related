import sklearn_crfsuite

class CRF_Segmenter:
    def __init__(self):
        self.model = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=0.1,             # L1 正则化系数
            c2=0.1,             # L2 正则化系数
            max_iterations=100, # 最大迭代次数
            all_possible_transitions=True
        )

    def _extract_features(self, sentence, i):
        """
        抽取第 i 个字符的上下文特征（窗口为 [-2, 2]）
        """
        word = sentence[i]
        
        features = {
            'bias': 1.0,
            'string': word,
            'is_first': i == 0,
            'is_last': i == len(sentence) - 1,
            
            # 前后单个字符特征
            'prev_char': '' if i == 0 else sentence[i - 1],
            'next_char': '' if i == len(sentence) - 1 else sentence[i + 1],
            'prev_2char': '' if i < 2 else sentence[i - 2],
            'next_2char': '' if i > len(sentence) - 3 else sentence[i + 2],
            
            # 组合特征（Bigram 特征，对中文分词至关重要）
            'prev_and_curr': '' if i == 0 else sentence[i - 1] + word,
            'curr_and_next': '' if i == len(sentence) - 1 else word + sentence[i + 1],
        }
        return features

    def _sentence_to_features(self, sentence):
        """将一句话转换为特征序列"""
        return [self._extract_features(sentence, i) for i in range(len(sentence))]

    def _sentence_to_labels(self, sentence_words):
        """将以空格分隔的词列表转换为 BMES 标签序列"""
        labels = []
        for word in sentence_words:
            if len(word) == 1:
                labels.append('S')
            else:
                labels.extend(['B'] + ['M'] * (len(word) - 2) + ['E'])
        return labels

    def train(self, corpus_path):
        """根据以空格分隔的语料文件训练模型"""
        X_train = []
        y_train = []

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                words = line.strip().split()
                if not words:
                    continue
                
                raw_sentence = "".join(words)
                labels = self._sentence_to_labels(words)
                features = self._sentence_to_features(raw_sentence)

                X_train.append(features)
                y_train.append(labels)

        # 拟合 CRF 模型
        self.model.fit(X_train, y_train)

    def cut(self, text):
        """输入原始句子，输出分词结果"""
        if not text:
            return []

        features = [self._sentence_to_features(text)]
        predicted_labels = self.model.predict(features)[0]

        # 根据预测出的 BMES 标签组词
        words = []
        word = ""
        for char, tag in zip(text, predicted_labels):
            word += char
            if tag in ['E', 'S']:
                words.append(word)
                word = ""

        if word:  # 边界容错处理
            words.append(word)

        return words


# ==================== 测试示例 ====================
if __name__ == '__main__':
    import os

    # 1. 生成简单的训练语料文件
    corpus_file = 'train_corpus_crf.txt'
    with open(corpus_file, 'w', encoding='utf-8') as f:
        f.write("小明 喜欢 看 电影 \n")
        f.write("隐马尔可夫模型 在 自然语言处理 中 应用 广泛 \n")
        f.write("条件随机场 也是 一种 序列标注 算法 \n")
        f.write("深度学习 效果 更好 \n")
        f.write("小明 在 学习 条件随机场 分词算法 \n")

    # 2. 训练 CRF 模型
    crf_segmenter = CRF_Segmenter()
    crf_segmenter.train(corpus_file)

    # 3. 预测分词
    test_text = "小明在学习条件随机场"
    result = crf_segmenter.cut(test_text)

    print("待分词文本:", test_text)
    print("CRF分词结果:", " / ".join(result))

    # 清理临时文件
    if os.path.exists(corpus_file):
        os.remove(corpus_file)