# AdaBoost (Adaptive Boosting)

## 1. 算法原理与概述

AdaBoost（自适应提升）是一种迭代 Boosting 算法。其基本逻辑是：在前轮弱分类器预测错误样本的基础上，增加其权重，并降低预测正确样本的权重；最终将多个弱分类器按各自预测表现的权重加权组合为强分类器。

```
 [ 均匀权重样本 W1 ] ---> [ 弱分类器 h1 ] ---> 计算分类误差 err1 ---> 确定分类器权重 alpha1
                                 |
                          更新样本权重 W2
                                 v
 [ 调整权重样本 W2 ] ---> [ 弱分类器 h2 ] ---> ... ---> 强分类器 H(X) = sign(sum(alpha_m * h_m))

```

---

## 2. 数学原理

### 2.1 算法流程

1. 初始化样本权重分布 $D_1(i) = \frac{1}{m}$。
2. 对 $m = 1, \dots, M$：
a. 使用权重 $D_m$ 训练弱分类器 $G_m(x)$。
b. 计算分类误差率：

$$e_m = \sum_{i=1}^{m} D_m(i) I(G_m(x_i) \neq y_i)$$



c. 计算分类器权重系数：

$$\alpha_m = \frac{1}{2} \ln \left( \frac{1 - e_m}{e_m} \right)$$



d. 更新样本权重：

$$D_{m+1}(i) = \frac{D_m(i) \exp(-\alpha_m y_i G_m(x_i))}{Z_m}$$



其中 $Z_m$ 为归一化常数。
3. 最终分类器：

$$G(x) = \text{sign}\left( \sum_{m=1}^{M} \alpha_m G_m(x) \right)$$



---

## 3. Python / NumPy / scikit-learn 实现

```python
import numpy as np
from sklearn.ensemble import AdaBoostClassifier as SklearnAdaBoost
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

class CustomAdaBoost:
    def __init__(self, n_clf=5):
        self.n_clf = n_clf
        self.clfs = []
        self.alpha = []

    def fit(self, X, y):
        n_samples, _ = X.shape
        # 初始化分布
        w = np.full(n_samples, (1 / n_samples))
        y_signed = np.where(y <= 0, -1, 1)

        for _ in range(self.n_clf):
            # 使用单层决策树 (Decision Stump)
            clf = DecisionTreeClassifier(max_depth=1)
            clf.fit(X, y_signed, sample_weight=w)
            predictions = clf.predict(X)

            # 误差率
            error = np.sum(w[y_signed != predictions])
            if error >= 0.5:
                break

            # 计算分类器权重
            alpha = 0.5 * np.log((1.0 - error) / (error + 1e-10))

            # 更新权重
            w *= np.exp(-alpha * y_signed * predictions)
            w /= np.sum(w)  # 归一化

            self.clfs.append(clf)
            self.alpha.append(alpha)

    def predict(self, X):
        clf_preds = [alpha * clf.predict(X) for alpha, clf in zip(self.alpha, self.clfs)]
        y_pred = np.sum(clf_preds, axis=0)
        return np.where(np.sign(y_pred) <= 0, 0, 1)

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)

    ada = CustomAdaBoost(n_clf=10)
    ada.fit(X, y)
    print("Custom AdaBoost Acc:", accuracy_score(y, ada.predict(X)))

    sk_ada = SklearnAdaBoost(n_estimators=10, random_state=42)
    sk_ada.fit(X, y)
    print("Sklearn AdaBoost Acc:", accuracy_score(y, sk_ada.predict(X)))

```

