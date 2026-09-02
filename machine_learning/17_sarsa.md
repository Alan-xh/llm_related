# 17. SARSA (State-Action-Reward-State-Action)

## 1. 核心原理

SARSA 是一种**无模型 (Model-Free)**、**同策略 (On-Policy)** 的时序差分（Temporal Difference, TD）强化学习算法。

* **名称来源**：其更新过程依赖于五元组元组 $(s_t, a_t, r_t, s_{t+1}, a_{t+1})$。
* $s_t$: $t$ 时刻的环境状态
* $a_t$: $t$ 时刻智能体执行的动作
* $r_t$: 执行动作 $a_t$ 后获得的即时奖励
* $s_{t+1}$: 执行动作 $a_t$ 后转移到的下一个状态
* $a_{t+1}$: 在状态 $s_{t+1}$ 下根据当前策略实际选择的下一个动作


* **同策略 (On-Policy)**：智能体用于与环境交互探索动作的策略（如 $\epsilon$-greedy 策略）与更新 $Q$ 值的策略是完全相同的。即更新时直接使用下一步实际**已经选择**的动作 $a_{t+1}$，而非 Q-learning 中的最大动作 $\max_{a'} Q(s', a')$。
* $\epsilon$(艾普西隆): 探索率，即以该概率随机选择动作以探索环境
* $Q$: 状态-动作价值函数，用于估计在某状态下采取某动作的预期累计回报
* $s'$: 下一个状态 (对应 $s_{t+1}$)
* $a'$: 在状态 $s'$ 下的可选动作



因此 SARSA 的行为比 Q-learning 更加“保守”和安全。

## 2. 算法与数学公式

### 2.1 SARSA 更新公式

$$Q(s_t, a_t) \leftarrow Q(s_t, a_t) + \alpha \left[ r_t + \gamma Q(s_{t+1}, a_{t+1}) - Q(s_t, a_t) \right]$$

* Q(s, a): 状态-动作价值函数（Q值），表示在状态 s 采取动作 a 的预期累计回报
* $s_t$: $t$ 时刻的状态
* $a_t$: $t$ 时刻选择的动作
* $\alpha$(阿尔法): 学习率（Learning Rate），控制新信息覆盖旧信息的速率 ($0 < \alpha \le 1$)
* $r_t$: $t$ 时刻获得的即时奖励（Reward）
* $\gamma$(伽马): 折扣因子（Discount Factor），用于衡量未来奖励对当前价值的影响 ($0 \le \gamma \le 1$)
* $s_{t+1}$: $t+1$ 时刻（下一个）的状态
* $a_{t+1}$: $t+1$ 时刻（下一个）实际选择并准备执行的动作

与 Q-learning 的本质区别：

* **Q-Learning (Off-policy)**: $TD_{target} = r_t + \gamma \max_{a'} Q(s_{t+1}, a')$
* $TD_{target}$: 时序差分目标值（Temporal Difference Target）
* $r_t$: $t$ 时刻获得的即时奖励
* $\gamma$(伽马): 折扣因子
* $s_{t+1}$: 下一个状态
* $a'$: 下一个状态下的候选动作
* $\max_{a'} Q(s_{t+1}, a')$: 下一个状态 $s_{t+1}$ 下所有可选动作中的最大 Q 值
* $Q$: 状态-动作价值函数


* **SARSA (On-policy)**: $TD_{target} = r_t + \gamma Q(s_{t+1}, a_{t+1})$
* $TD_{target}$: 时序差分目标值（Temporal Difference Target）
* $r_t$: $t$ 时刻获得的即时奖励
* $\gamma$(伽马): 折扣因子
* $s_{t+1}$: 下一个状态
* $a_{t+1}$: 下一个状态下根据当前策略实际选择的动作
* $Q$: 状态-动作价值函数



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