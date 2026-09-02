# 马尔可夫决策过程 (Markov Decision Process, MDP)

## 1. 算法原理

马尔可夫决策过程（MDP）是强化学习中对序列决策问题进行数学建模的标准形式。

核心是**马尔可夫性质**：下一状态的概率仅取决于当前状态和当前采取的动作，而与过去的历史状态无关。

五元组描述：$\mathcal{M} = (S, A, P, R, \gamma)$。

* $\mathcal{M}$: 马尔可夫决策过程模型
* S: 状态集合 (State Space)
* A: 动作集合 (Action Space)
* P: 状态转移概率矩阵/函数
* R: 奖励函数 (Reward Function)
* $\gamma$(伽马): 折扣因子 (Discount Factor, $0 \le \gamma \le 1$)

---

## 2. 数学公式与推导

1. **贝尔曼期望方程 (Bellman Expectation Equation)**：

$$V^\pi(s) = \sum_{a \in A} \pi(a\vert{}s) \sum_{s' \in S} P(s'\vert{}s,a) \left[ R(s,a,s') + \gamma V^\pi(s') \right]$$

* $V^\pi(s)$: 在策略 $\pi$ 下，状态 $s$ 的状态价值 (State-Value Function)
* $\pi$(派): 策略 (Policy)，$\pi(a\vert{}s)$ 表示在状态 $s$ 下采取动作 $a$ 的概率
* a: 当前采取的动作
* A: 动作集合
* s: 当前状态
* S: 状态集合
* s': 下一个状态
* $P(s'\vert{}s,a)$: 状态转移概率，即在状态 $s$ 采取动作 $a$ 转移到状态 $s'$ 的概率
* $R(s,a,s')$: 奖励函数，表示从状态 $s$ 采取动作 $a$ 转移到状态 $s'$ 获得的即时奖励
* $\gamma$(伽马): 折扣因子
* $V^\pi(s')$: 在策略 $\pi$ 下，下一个状态 $s'$ 的状态价值

2. **贝尔曼最优方程 (Bellman Optimality Equation)**：

$$V^*(s) = \max_{a \in A} \sum_{s' \in S} P(s'\vert{}s,a) \left[ R(s,a,s') + \gamma V^*(s') \right]$$

* $V^*(s)$: 状态 $s$ 的最优状态价值 (Optimal State-Value Function)
* $\max_{a \in A}$: 对所有可能的动作 $a$ 取最大值
* a: 当前采取的动作
* A: 动作集合
* s: 当前状态
* S: 状态集合
* s': 下一个状态
* $P(s'\vert{}s,a)$: 状态转移概率
* $R(s,a,s')$: 奖励函数
* $\gamma$(伽马): 折扣因子
* $V^*(s')$: 下一个状态 $s'$ 的最优状态价值

---

## 3. ASCII 交互关系图

```
                 +-------------------+
                 |     环境 (Env)     |
                 +-------------------+
                   /               \
       状态 S_t   /                 \   奖励 R_t
                 v                   v
            +-----------------------------+
            |        智能体 (Agent)        |
            +-----------------------------+
                         |
                         | 动作 A_t
                         v
                 +-------------------+
                 |     环境 (Env)     |
                 +-------------------+


```

---

## 4. Python 代码实现 (基于 NumPy 值迭代法)

```python
import numpy as np

S = [0, 1, 2]
A = [0, 1]
gamma = 0.9

P = np.zeros((3, 2, 3))
P[0, 1, 1] = 1.0; P[0, 0, 0] = 1.0
P[1, 1, 2] = 1.0; P[1, 0, 0] = 1.0
P[2, 1, 2] = 1.0; P[2, 0, 1] = 1.0

R = np.array([
    [0, 0],
    [0, 10],
    [0, 0]
])

def value_iteration(P, R, gamma, theta=1e-6):
    n_states, n_actions = R.shape
    V = np.zeros(n_states)
    
    while True:
        delta = 0
        for s in range(n_states):
            v_old = V[s]
            q_values = [R[s, a] + gamma * np.sum(P[s, a, :] * V) for a in range(n_actions)]
            V[s] = max(q_values)
            delta = max(delta, abs(v_old - V[s]))
            
        if delta < theta:
            break
            
    return V

V_optimal = value_iteration(P, R, gamma)
print("MDP 最优状态价值 V*(s):", V_optimal.round(3))


```