
# 17. SARSA (State-Action-Reward-State-Action)

## 1. 核心原理

SARSA 是一种**无模型 (Model-Free)**、**同策略 (On-Policy)** 的时序差分（Temporal Difference, TD）强化学习算法。

* **名称来源**：其更新过程依赖于五元组元组 $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$。
* **同策略 (On-Policy)**：智能体用于与环境交互探索动作的策略（如 $\epsilon$-greedy 策略）与更新 $Q$ 值的策略是完全相同的。即更新时直接使用下一步实际**已经选择**的动作 $a_{t+1}$，而非 Q-learning 中的最大动作 $\max_{a'} Q(s', a')$。

因此 SARSA 的行为比 Q-learning 更加“保守”和安全。

## 2. 算法与数学公式

### 2.1 SARSA 更新公式

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$

与 Q-learning 的本质区别：

* **Q-Learning (Off-policy)**: $TD_{target} = r_t + \gamma \max_{a'} Q(s_{t+1}, a')$
* **SARSA (On-policy)**: $TD_{target} = r_t + \gamma Q(s_{t+1}, a_{t+1})$

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
|  在状态 s' 根据 epsilon-greedy 选取下一个动作 a'      |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
|  更新 Q(s, a):                                       |
|  Q(s, a) += alpha * [r + gamma * Q(s', a') - Q(s, a)]  |
+------------------------------------------------------+
                           |
                           v
+------------------------------------------------------+
|  更新 s <- s', a <- a'                                |
+------------------------------------------------------+

```

## 4. NumPy 纯代码实现

```python
import numpy as np

class SARSAAgent:
    def __init__(self, n_states, n_actions, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_actions)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, next_action, done):
        td_target = reward + (0 if done else self.gamma * self.q_table[next_state, next_action])
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error

if __name__ == "__main__":
    agent = SARSAAgent(n_states=5, n_actions=2)
    state = 0
    action = agent.choose_action(state)
    
    # 模拟环境交互一步
    next_state, reward, done = 1, 1.0, False
    next_action = agent.choose_action(next_state)
    
    agent.update(state, action, reward, next_state, next_action, done)
    print("SARSA Q-Table updated successfully:")
    print(agent.q_table)

```

