# 16. Q-学习 (Q-Learning)

## 1. 核心原理

Q-learning 是一种**无模型 (Model-Free)**、**离策略 (Off-Policy)** 的强化学习算法。

* **无模型**：智能体（Agent）不需要提前了解环境的转移概率模型 $P(s'\vert{}s, a)$ 和回报函数 $R(s, a)$。

* $P(s'\vert{}s, a)$: 在状态 $s$ 采取动作 $a$ 转移到下一个状态 $s'$ 的概率
* $s$: 当前状态 (State)
* $s'$: 下一个状态 (Next State)
* $a$: 采取的动作 (Action)
* $R(s, a)$: 在状态 $s$ 采取动作 $a$ 所获得的立即回报 (Reward)

* **离策略**：智能体在探索环境时使用的策略（如 $\epsilon$-greedy 策略）与目标更新所遵循的动作选择策略（纯贪婪策略 $\max_{a'} Q(s', a')$）可以不同。

* $\epsilon$(艾普西隆/伊普西隆): 探索率（$\epsilon$-greedy 策略中的随机动作概率）
* $Q(s', a')$: 在下一个状态 $s'$ 采取动作 $a'$ 的 $Q$ 值
* $a'$: 下一个状态可选的动作
* $s'$: 下一个状态

核心思想是构建并迭代维护一张 $Q$ 表（Q-Table），表项 $Q(s, a)$ 表示在状态 $s$ 采取动作 $a$ 能够获得的期望累积折扣回报。

* $Q(s, a)$: 在状态 $s$ 采取动作 $a$ 的动作价值（Q值）
* $s$: 当前状态
* $a$: 当前动作

## 2. 算法与数学公式

### 2.1 贝尔曼最优方程 (Bellman Optimality Equation)

$$Q^*(s, a) = R(s, a) + \gamma \max_{a'} Q^*(s', a')$$

* $Q^*(s, a)$: 在状态 $s$ 采取动作 $a$ 的最优 $Q$ 值（最优动作价值函数）
* $R(s, a)$: 在状态 $s$ 采取动作 $a$ 所获得的立即回报
* $\gamma$(伽马): 折扣因子 (Discount Factor)，用于对未来收益进行折现
* $a'$: 在下一个状态 $s'$ 时可选的所有动作
* $Q^*(s', a')$: 下一个状态 $s'$ 采取动作 $a'$ 的最优 $Q$ 值
* $s'$: 转移后的下一个状态

### 2.2 Q-Learning 差分更新公式 (TD-Update)

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t) \right]$$

* $Q(s_t, a_t)$: 当前时间步 $t$ 在状态 $s_t$ 执行动作 $a_t$ 的 $Q$ 值
* $s_t$: 时间步 $t$ 的状态
* $a_t$: 时间步 $t$ 执行的动作
* $\alpha$(阿尔法): 学习率 (Learning Rate)
* $r_t$: 时间步 $t$ 执行动作后获得的立即回报
* $\gamma$(伽马): 折扣因子 (Discount Factor)
* $a$: 在下一个状态 $s_{t+1}$ 可选择的候选动作
* $s_{t+1}$: 时间步 $t+1$ 的下一个状态
* $Q(s_{t+1}, a)$: 下一个状态 $s_{t+1}$ 采取动作 $a$ 的 $Q$ 值

其中：

* $\alpha \in (0, 1]$ 是学习率（Learning Rate）。
* $\gamma \in [0, 1)$ 是折扣因子（Discount Factor）。
* $r_t + \gamma \max_{a} Q(s_{t+1}, a) - Q(s_t, a_t)$ 称为 TD 误差（Temporal Difference Error）。

## 3. ASCII 流程框架图

```
+------------------------------------------------------+
|                   初始化 Q(s, a)                     |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
|  在状态 s 根据 epsilon-greedy 选择动作 a              |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
|  执行动作 a，观测回报 r 和新状态 s'                    |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
|  更新 Q(s, a):                                       |
|  Q(s,a) += alpha * [r + gamma * max_a' Q(s',a') - Q(s,a)]
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
|  更新状态 s <- s'                                     |
+------------------------------------------------------+


```

## 4. NumPy 纯代码实现

```python
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # 初始化 Q 表
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        # epsilon-greedy 策略
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        else:
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (0 if done else self.gamma * self.q_table[next_state, best_next_action])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

if __name__ == "__main__":
    agent = QLearningAgent(n_states=5, n_actions=2)
    state = 0
    action = agent.choose_action(state)
    next_state, reward, done = 1, 1.0, False
    agent.update(state, action, reward, next_state, done)
    print("Updated Q-Table:")
    print(agent.q_table)


```