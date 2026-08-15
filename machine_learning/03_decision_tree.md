# 决策树 (Decision Tree)

## 1. 算法原理与概述

决策树（Decision Tree）是一种非参数监督学习方法，通过树状结构对数据建立递归条件分割模型。树的内部节点表示一个属性测试，分支表示测试结果，叶子节点对应决策结果。

```
                    [ 根节点: 年龄 <= 30? ]
                         /         \
                      是 /           \ 否
                        /             \
             [ 拥有房产? ]          ( 输出: 批准贷款 )
               /       \
            是 /         \ 否
              /           \
     ( 批准贷款 )      ( 拒绝贷款 )

```

---

## 2. 数学原理与指标

### 2.1 节点不纯度度量

#### A. 信息熵 (Entropy) & 信息增益 (ID3)

集合 $D$ 的熵：


$$H(D) = -\sum_{k=1}^{K} p_k \log_2(p_k)$$


特征 $A$ 对集合 $D$ 的信息增益：


$$Gain(D, A) = H(D) - \sum_{v=1}^{V} \frac{\vert{}D^v\vert{}}{\vert{}D\vert{}} H(D^v)$$

#### B. 信息增益比 (C4.5)

防止选择取值较多的特征：


$$Gain\_ratio(D, A) = \frac{Gain(D, A)}{IV(A)}, \quad IV(A) = -\sum_{v=1}^{V} \frac{\vert{}D^v\vert{}}{\vert{}D\vert{}} \log_2 \frac{\vert{}D^v\vert{}}{\vert{}D\vert{}}$$

#### C. 基尼指数 (Gini Index - CART)

$$Gini(D) = 1 - \sum_{k=1}^{K} p_k^2$$


特征 $A$ 分割下的基尼指数：


$$Gini\_index(D, A) = \frac{\vert{}D_1\vert{}}{\vert{}D\vert{}} Gini(D_1) + \frac{\vert{}D_2\vert{}}{\vert{}D\vert{}} Gini(D_2)$$

---

## 3. Python / NumPy / scikit-learn 实现

```python
import numpy as np
from collections import Counter
from sklearn.tree import DecisionTreeClassifier as SklearnTree
from sklearn.metrics import accuracy_score

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class CustomDecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def _gini(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return 1.0 - np.sum(ps ** 2)

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for thresh in thresholds:
                left_idxs, right_idxs = self._split(X_column, thresh)
                if len(left_idxs) == 0 or len(right_idxs) == 0:
                    continue
                
                # Gini gain
                parent_gini = self._gini(y)
                n = len(y)
                n_l, n_r = len(left_idxs), len(right_idxs)
                e_l, e_r = self._gini(y[left_idxs]), self._gini(y[right_idxs])
                child_gini = (n_l / n) * e_l + (n_r / n) * e_r
                gini_gain = parent_gini - child_gini

                if gini_gain > best_gain:
                    best_gain = gini_gain
                    split_idx = feat_idx
                    split_thresh = thresh

        return split_idx, split_thresh

    def _build_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, n_feats, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)

        if best_feat is None:
            leaf_value = Counter(y).most_common(1)[0][0]
            return Node(value=leaf_value)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        left = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feat, best_thresh, left, right)

    def fit(self, X, y):
        self.root = self._build_tree(X, y)

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    iris = load_iris()
    X, y = iris.data, iris.target

    clf = CustomDecisionTree(max_depth=5)
    clf.fit(X, y)
    preds = clf.predict(X)
    print("Custom Decision Tree Acc:", accuracy_score(y, preds))

    sk_tree = SklearnTree(max_depth=5)
    sk_tree.fit(X, y)
    print("Sklearn Decision Tree Acc:", accuracy_score(y, sk_tree.predict(X)))
