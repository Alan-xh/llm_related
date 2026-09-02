# 孤立森林 (Isolation Forest)

## 1. 算法原理

孤立森林（Isolation Forest, iForest）是一种基于集成学习的**无监督异常检测算法**。

其核心假设是：**异常值少且特征值与正常数据差异大，因此更容易被孤立（Separated）**。

算法构建若干棵二叉树（Isolation Tree, iTree）：随机选择特征和分割点，异常点在树中离根节点更近（路径长度较小），而正常点位于深层节点。

---

## 2. 数学公式与推导

1. **平均路径长度归一化因子**：

$$c(n) = 2 \left( \ln(n - 1) + 0.5772156649 \right) - \frac{2(n - 1)}{n}$$

* c(n): 样本大小为 $n$ 时的平均路径长度归一化因子（模拟二叉查找树未成功搜索的平均路径长度）
* n: 构建孤立树时抽取的样本数量/节点数据量
* $\ln$: 自然对数函数
* 0.5772156649: 欧拉-马歇罗尼常数 (Euler-Mascheroni constant)，通常记作 $\gamma$(伽马)

2. **异常得分 (Anomaly Score)**：

$$s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}$$

* s(x, n): 样本 $x$ 基于样本量 $n$ 计算得出的异常得分（取值在 0 到 1 之间，越接近 1 说明越可能是异常点）
* x: 待评估的目标样本点
* n: 构建孤立树时的样本总体规模/样本量
* E(h(x)): 样本 $x$ 在森林中所有孤立树上的路径长度的期望值（平均深度）
* h(x): 样本 $x$ 在单棵孤立树上的路径长度（即从根节点到叶子节点经过的边数/分割次数）
* c(n): 样本量为 $n$ 时的平均路径长度归一化因子

其中 $E(h(x))$ 为样本 $x$ 在森林中路径长度的期望值。

---

## 3. ASCII 结构图

```
      正常点路径 (深层):                     异常点路径 (浅层):

             (根)                                   (根)
            /    \                                 /    \
          (*)    ...                              (*)   [异常点 x_anomaly!]
         /   \                                   (孤立)
       (*)   ...
      /
 [正常点 x_normal]


```

---

## 4. Python 代码实现 (基于 Scikit-Learn)

```python
import numpy as np
from sklearn.ensemble import IsolationForest

np.random.seed(42)
X_normal = 0.3 * np.random.randn(100, 2)
X_outliers = np.random.uniform(low=-4, high=4, size=(10, 2))
X = np.vstack([X_normal, X_outliers])

clf = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
clf.fit(X)

y_pred = clf.predict(X)
scores = clf.decision_function(X)

print("预测为异常的点数量:", np.sum(y_pred == -1))
print("前5个样本的异常得分:", scores[:5].round(3))

```