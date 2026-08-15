# 朴素贝叶斯 (Naive Bayes)

## 1. 算法原理与概述

朴素贝叶斯（Naive Bayes）是一种基于**贝叶斯定理**与**特征条件独立假设**的概率分类方法。虽然特征独立性假设在实际场景中难以完全满足，但该模型结构简单，计算极其高效，在文本分类与垃圾邮件识别中表现出色。

```
                  P(Y) (先验概率)  x  P(X|Y) (条件似然)
 P(Y|X) (后验概率) = -----------------------------------------
                                P(X) (证据)

```

---

## 2. 数学原理

### 2.1 贝叶斯定理与条件独立假设

由贝叶斯公式：


$$P(y\vert{}\mathbf{x}) = \frac{P(y) P(\mathbf{x}\vert{}y)}{P(\mathbf{x})}$$

条件独立性假设：


$$P(\mathbf{x}\vert{}y) = P(x_1, x_2, \dots, x_n \vert{} y) = \prod_{i=1}^{n} P(x_i \vert{} y)$$

因此分类目标为：


$$\hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \vert{} y)$$

### 2.2 高斯朴素贝叶斯 (连续特征)

连续特征假设服从高斯分布：


$$P(x_i \vert{} y) = \frac{1}{\sqrt{2\pi \sigma_y^2}} \exp\left( -\frac{(x_i - \mu_y)^2}{2\sigma_y^2} \right)$$

---

## 3. Python / NumPy / scikit-learn 实现

```python
import numpy as np
from sklearn.naive_bayes import GaussianNB as SklearnGaussianNB
from sklearn.metrics import accuracy_score

class CustomGaussianNaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        self._mean = np.zeros((n_classes, n_features))
        self._var = np.zeros((n_classes, n_features))
        self._priors = np.zeros(n_classes)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0) + 1e-9  # 防止除以 0
            self._priors[idx] = X_c.shape[0] / float(n_samples)

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._priors[idx])
            posterior = np.sum(np.log(self._pdf(idx, x)))
            posterior = prior + posterior
            posteriors.append(posterior)
        return self._classes[np.argmax(posteriors)]

    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)
        return numerator / denominator

if __name__ == "__main__":
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)

    gnb = CustomGaussianNaiveBayes()
    gnb.fit(X, y)
    print("Custom Naive Bayes Acc:", accuracy_score(y, gnb.predict(X)))

    sk_gnb = SklearnGaussianNB()
    sk_gnb.fit(X, y)
    print("Sklearn Naive Bayes Acc:", accuracy_score(y, sk_gnb.predict(X)))

```

