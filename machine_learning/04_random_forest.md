# 随机森林 (Random Forest)

## 1. 算法原理与概述

随机森林（Random Forest）是一种集成学习（Ensemble Learning）算法，基于 Bagging（Bootstrap Aggregating）框架。它通过并行构建多棵独立的决策树，并利用特征随机选择降低树之间的相关性，最后通过投票（分类）或取平均（回归）输出最终结果。

```


```

```
                    [ 原始训练数据集 D ]
                     /        |        \
           Bootstrap /   Bootstrap  \ Bootstrap
                   v          v          v
               [ 样本子集 D1 ] [ 样本子集 D2 ] [ 样本子集 D3 ]
                  |          |          |

```

(随机挑选 k 个特征)  v          v          v
[ 决策树 T1 ] [ 决策树 T2 ] [ 决策树 T3 ]
\          |          /
\         |         /
v        v        v
[ 多数投票 / 均值预测 ]
|
v
[ 最终输出 ]

```


```

---

## 2. 数学原理

### 2.1 双重随机性

1. **样本随机性（Bootstrap 抽样）**：有放回地采样 $N$ 次获得与原集合等大的训练子集。约 $36.8\%$ 的数据未被抽到（包外数据 Out-Of-Bag, OOB）。

* N: 样本总数/原始数据集中的样本数量

2. **特征随机性**：分类任务中从 $M$ 个特征中随机选取 $k = \sqrt{M}$ 个候选特征选择最佳分割节点。

* k: 随机选择的候选特征数量
* M: 数据集中的总特征数量

### 2.2 方差降低推导

对 $B$ 棵独立且方差为 $\sigma^2$ 的树，均值方差为 $\frac{\sigma^2}{B}$。若树之间相关系数为 $\rho$，集成模型总体方差为：


$$\text{Var}(\text{Forest}) = \rho \sigma^2 + \frac{1 - \rho}{B} \sigma^2$$

* Var(Forest): 随机森林集成模型的总体方差
* ρ(柔): 树与树之间的相关系数
* σ(西格玛): 单棵决策树的方差（σ² 代表单棵树的方差）
* B: 随机森林中决策树的总棵数

特征随机选择减小了 $\rho$，从而显著降低整体方差，提高泛化能力。

* ρ(柔): 树与树之间的相关系数

---

## 3. Python / NumPy / scikit-learn 实现

```python
import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier as SklearnRandomForest
from sklearn.metrics import accuracy_score

class CustomRandomForest:
    def __init__(self, n_trees=10, max_depth=10, min_samples_split=2, max_features=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.trees = []

    def fit(self, X, y):
        from 03_decision_tree import CustomDecisionTree  # 假设重用前文 DecisionTree
        self.trees = []
        n_samples, n_features = X.shape

        for _ in range(self.n_trees):
            # Bootstrap 抽样
            idxs = np.random.choice(n_samples, n_samples, replace=True)
            X_sample, y_sample = X[idxs], y[idxs]

            tree = CustomDecisionTree(
                max_depth=self.max_depth, 
                min_samples_split=self.min_samples_split
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        # 树的预测结果转置后取众数 (多数投票)
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [Counter(preds).most_common(1)[0][0] for preds in tree_preds]
        return np.array(y_pred)

if __name__ == "__main__":
    from sklearn.datasets import load_digits
    digits = load_digits()
    X, y = digits.data, digits.target

    sk_rf = SklearnRandomForest(n_estimators=10, max_depth=5, random_state=42)
    sk_rf.fit(X, y)
    print("Sklearn Random Forest Acc:", accuracy_score(y, sk_rf.predict(X)))


```