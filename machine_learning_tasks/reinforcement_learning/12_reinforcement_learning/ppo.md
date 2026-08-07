根据你提供的通用 PyTorch 规范与标准，我已将原有的 PPO 算法代码进行了重构与扩展，完善了**任务 Header 描述**、**模块化层级结构**、**张量 Shape 追踪**、**数学与代码映射注释**以及**配套 Markdown 架构文档**。

---

### Part 1: Python 可执行代码

```python
"""
========================================================================================
任务与理论 Header (Task & Theory Header)
========================================================================================
任务编号/名称: 任务 12 - 强化学习 (Reinforcement Learning)
领域分类: 在策略强化学习 (On-Policy Reinforcement Learning) / 连续状态与离散动作控制
代表架构/算法: PPO (Proximal Policy Optimization - Clip Variants)
论文来源: Schulman et al., "Proximal Policy Optimization Algorithms", arXiv:1707.06347 (2017)

核心思想与机制:
1. PPO 通过引入策略裁剪机制 (Clipping Mechanism)，将新旧策略的比率 r_t(θ) 限制在 [1-ε, 1+ε] 区间内，
   从而防止在单次采样数据上更新幅度过大导致策略崩溃。
2. 采用广义优势估计 (Generalized Advantage Estimation, GAE) 平衡偏差 (Bias) 与方差 (Variance)。
3. 使用 Actor-Critic 共享/解耦网络架构，同时输出动作概率分布 (Policy) 和状态价值估计 (Value Function)。

数学公式 / 优化目标:
1. 策略比率 (Ratio):
   r_t(θ) = π_θ(a_t | s_t) / π_θ_old(a_t | s_t) = exp(log_π_θ(a_t | s_t) - log_π_θ_old(a_t | s_t))

2. 裁剪策略损失 (Clipped Surrogate Objective):
   L_CLIP(θ) = E_t [ min( r_t(θ) * A_t, clip(r_t(θ), 1-ε, 1+ε) * A_t ) ]

3. 值函数 MSE 损失 (Value Loss):
   L_VF(θ) = E_t [ (V_θ(s_t) - V_target_t)^2 ]

4. 熵正则项 (Entropy Regularization):
   S[π_θ](s_t) = E_a [ -log π_θ(a | s_t) ]

5. PPO 联合总目标函数 (Total Loss to Minimize):
   L_TOTAL(θ) = - L_CLIP(θ) + c_1 * L_VF(θ) - c_2 * S[π_θ](s_t)

6. GAE 优势估计 (GAE Formula):
   δ_t^V = r_t + γ * V(s_{t+1}) * (1 - d_t) - V(s_t)
   A_t^{GAE(γ, λ)} = Σ_{l=0}^{∞} (γ * λ)^l * δ_{t+l}^V

数据输入与输出规范:
Input Observation:  Shape [B, obs_dim]  (CartPole-v1 中 obs_dim = 4: 位置, 速度, 角度, 角速度)
Output Action:       Shape [B]          (离散动作 0 或 1)
Output State-Value: Shape [B]          (标量状态价值估计)
========================================================================================
"""

# ======================================================================================
# 2. 依赖导入 (Imports)
# ======================================================================================
import math
import random
from typing import Dict, List, Tuple, Union, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical

# 优先导入 gymnasium，未安装则回退至 gym
try:
    import gymnasium as gym
except ImportError:
    import gym

# ======================================================================================
# 3. 超参数与全局配置 (Hyperparameters & Config)
# ======================================================================================
class PPOConfig:
    """PPO 训练全局超参数配置项"""
    ENV_ID: str = "CartPole-v1"
    SEED: int = 42
    
    # 采样与训练迭代控制
    EPOCHS: int = 50                  # 总训练轮数 (Outer Epochs)
    STEPS_PER_EPOCH: int = 2048       # 每轮采样的 Rollout 步数
    PPO_EPOCHS: int = 4               # 每次 Rollout 数据的 PPO 重复优化轮数
    MINIBATCH_SIZE: int = 64          # PPO 更新的 Mini-batch 大小
    
    # 算法数学超参数
    LR: float = 3e-4                  # Adam 优化器学习率
    GAMMA: float = 0.99               # 折扣因子 γ
    GAE_LAMBDA: float = 0.95          # GAE 平滑因子 λ
    CLIP_EPS: float = 0.2             # 策略裁剪范围 ε
    VALUE_COEF: float = 0.5           # 价值损失权重 c1
    ENTROPY_COEF: float = 0.01        # 熵正则权重 c2
    MAX_GRAD_NORM: float = 0.5        # 梯度裁剪阈值
    
    DEVICE: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CONFIG = PPOConfig()

# 设置随机种子保证可复现性
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(CONFIG.SEED)

# ======================================================================================
# 4. 数据处理与 Rollout 管道 (Data Pipeline & Utils)
# ======================================================================================
class RolloutBuffer:
    """
    On-Policy 经验回放缓冲区，用于存储单个 Epoch 收集的轨迹数据，
    并计算 GAE 优势 (Advantage) 与 Target Returns。
    """
    def __init__(self, steps: int, obs_dim: int, device: torch.device):
        self.steps = steps
        self.device = device
        
        # 预分配 NumPy 数组内存
        self.obs = np.zeros((steps, obs_dim), dtype=np.float32)
        self.actions = np.zeros(steps, dtype=np.int64)
        self.logprobs = np.zeros(steps, dtype=np.float32)
        self.rewards = np.zeros(steps, dtype=np.float32)
        self.values = np.zeros(steps, dtype=np.float32)
        self.dones = np.zeros(steps, dtype=np.float32)
        
        self.returns = np.zeros(steps, dtype=np.float32)
        self.advantages = np.zeros(steps, dtype=np.float32)
        
        self.ptr = 0

    def store(self, obs: np.ndarray, act: int, logp: float, rew: float, val: float, done: bool):
        """存储单步转置数据 (Step Transition)"""
        assert self.ptr < self.steps, "Buffer 溢出，请重置 Buffer！"
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = act
        self.logprobs[self.ptr] = logp
        self.rewards[self.ptr] = rew
        self.values[self.ptr] = val
        self.dones[self.ptr] = float(done)
        self.ptr += 1

    def compute_gae_and_returns(self, last_val: float, last_done: bool, gamma: float, gae_lambda: float):
        """
        根据 GAE 递归公式计算优势值 (Advantage) 和目标回报 (Returns)。
        
        数学公式:
            δ_t = r_t + γ * V(s_{t+1}) * (1 - d_t) - V(s_t)
            A_t = δ_t + γ * λ * (1 - d_t) * A_{t+1}
            R_t = A_t + V(s_t)
        """
        gae = 0.0
        for t in reversed(range(self.steps)):
            if t == self.steps - 1:
                next_non_terminal = 1.0 - float(last_done)
                next_value = last_val
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = self.values[t + 1]
            
            # δ_t 项计算 (TD Error)
            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            # GAE 优势累加
            gae = delta + gamma * gae_lambda * next_non_terminal * gae
            
            self.advantages[t] = gae
            self.returns[t] = gae + self.values[t]  # Target Return: R_t = A_t + V_t

    def get_torch_tensors(self) -> Tuple[torch.Tensor, ...]:
        """将缓存数据转换为 PyTorch Tensor 并移至指定计算设备"""
        obs_t = torch.tensor(self.obs, dtype=torch.float32, device=self.device)         # [N, obs_dim]
        act_t = torch.tensor(self.actions, dtype=torch.int64, device=self.device)       # [N]
        logp_t = torch.tensor(self.logprobs, dtype=torch.float32, device=self.device)   # [N]
        ret_t = torch.tensor(self.returns, dtype=torch.float32, device=self.device)     # [N]
        adv_t = torch.tensor(self.advantages, dtype=torch.float32, device=self.device)  # [N]
        
        # 重置指针以备下次使用
        self.ptr = 0
        return obs_t, act_t, logp_t, ret_t, adv_t


# ======================================================================================
# 5. 核心子模块 / Encoder / Actor-Critic 网络 (Sub-components)
# ======================================================================================
class ActorCritic(nn.Module):
    """
    Actor-Critic 联合网络，共享特征提取骨干网络，解耦输出 Action Logits 和 State Value。

    数学原理 / 变换逻辑:
        h = SiLU(W2 * SiLU(W1 * x + b1) + b2)
        logits = W_actor * h + b_actor
        value  = W_critic * h + b_critic

    Args:
        obs_dim (int): 状态观察维度。
        action_dim (int): 动作空间维度。
        hidden_dim (int): 隐藏层特征维度，默认 64。

    Inputs:
        x (Tensor): 状态观察张量，shape: [B, obs_dim]

    Outputs:
        logits (Tensor): 未归一化的动作概率对数，shape: [B, action_dim]
        value (Tensor): 状态价值估计，shape: [B, 1]
    """
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        # 共享特征提取主干
        self.shared_backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),  # 采用现代高效激活函数 SiLU (Swish)
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        # Actor 头 (Policy Network)
        self.actor_head = nn.Linear(hidden_dim, action_dim)
        
        # Critic 头 (Value Network)
        self.critic_head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播计算

        Inputs:
            x (Tensor): [B, obs_dim]
        
        Outputs:
            logits (Tensor): [B, action_dim]
            value (Tensor):  [B, 1]
        """
        # x: [B, obs_dim]
        h = self.shared_backbone(x)     # -> [B, hidden_dim]
        logits = self.actor_head(h)     # -> [B, action_dim]
        value = self.critic_head(h)     # -> [B, 1]
        return logits, value

    def get_action_and_value(
        self, 
        x: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        根据当前状态采样动作并计算 log_prob、熵与价值。若传入 action，则评估该 action 的策略响应。

        Inputs:
            x (Tensor): 状态张量，shape: [B, obs_dim]
            action (Tensor, optional): 指定动作，shape: [B]

        Outputs:
            action (Tensor): 采样或指定的动作，shape: [B]
            log_prob (Tensor): 动作对数概率 log π(a|s)，shape: [B]
            entropy (Tensor): 策略分布熵 S[π](s)，shape: [B]
            value (Tensor): 状态价值估计 V(s)，shape: [B]
        """
        logits, value = self.forward(x)                         # logits: [B, action_dim], value: [B, 1]
        dist = Categorical(logits=logits)                       # 离散 Categorical 分布
        
        if action is None:
            action = dist.sample()                              # 采样动作 -> [B]
            
        log_prob = dist.log_prob(action)                        # 计算对数概率 -> [B]
        entropy = dist.entropy()                                # 计算分布熵 -> [B]
        value = value.squeeze(-1)                               # 压缩尾部维度 -> [B]
        
        return action, log_prob, entropy, value


# ======================================================================================
# 6. 顶层模型 / Pipeline 主体 (Top-level Architecture / Model)
# ======================================================================================
class PPOAgent:
    """
    PPO 算法 Pipeline 主控类：解耦模型管理、环境交互 Rollout 采集、损失计算与参数优化。
    """
    def __init__(self, obs_dim: int, action_dim: int, config: PPOConfig):
        self.cfg = config
        self.model = ActorCritic(obs_dim, action_dim).to(self.cfg.DEVICE)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.cfg.LR, eps=1e-5)
        self.buffer = RolloutBuffer(self.cfg.STEPS_PER_EPOCH, obs_dim, self.cfg.DEVICE)

    def collect_rollout(self, env: gym.Env, current_obs: np.ndarray) -> np.ndarray:
        """
        在环境中运行当前策略，采集 STEPS_PER_EPOCH 步的轨迹数据充填 Buffer。
        """
        obs = current_obs
        for step in range(self.cfg.STEPS_PER_EPOCH):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.cfg.DEVICE).unsqueeze(0) # [1, obs_dim]
            
            with torch.no_grad():
                # 评估动作与价值
                action, log_prob, _, value = self.model.get_action_and_value(obs_tensor)
                
            action_np = action.cpu().numpy()[0]
            log_prob_np = log_prob.cpu().numpy()[0]
            value_np = value.cpu().numpy()[0]
            
            # 环境交互
            next_obs, reward, terminated, truncated, _ = env.step(action_np)
            done = terminated or truncated
            
            # 存入缓冲区
            self.buffer.store(obs, action_np, log_prob_np, reward, value_np, done)
            
            obs = next_obs
            if done:
                obs, _ = env.reset()
                
        # 计算最后一步的价值以进行 GAE 截断估计
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.cfg.DEVICE).unsqueeze(0)
        with torch.no_grad():
            _, _, _, last_value = self.model.get_action_and_value(obs_tensor)
        
        self.buffer.compute_gae_and_returns(
            last_val=last_value.cpu().item(),
            last_done=False,
            gamma=self.cfg.GAMMA,
            gae_lambda=self.cfg.GAE_LAMBDA
        )
        return obs

# ======================================================================================
# 7. 损失函数与评估指标 (Loss & Metrics)
# ======================================================================================
class PPOLoss(nn.Module):
    """
    PPO 联合计算损失模块 (Clipped Surrogate Policy Loss + Value Loss - Entropy Bonus)
    """
    def __init__(self, clip_eps: float, value_coef: float, entropy_coef: float):
        super().__init__()
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef

    def forward(
        self, 
        new_logp: torch.Tensor, 
        old_logp: torch.Tensor, 
        advantages: torch.Tensor, 
        values: torch.Tensor, 
        returns: torch.Tensor, 
        entropy: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        计算 PPO 各项损失

        Inputs:
            new_logp (Tensor): 当前策略下动作对数概率，shape: [B_mb]
            old_logp (Tensor): 采样策略下动作对数概率，shape: [B_mb]
            advantages (Tensor): 归一化后的优势值 A_t，shape: [B_mb]
            values (Tensor): 当前 Critic 网络预测的状态价值，shape: [B_mb]
            returns (Tensor): GAE 目标回报 R_t，shape: [B_mb]
            entropy (Tensor): 当前策略的分布熵 S[π]，shape: [B_mb]

        Outputs:
            total_loss (Tensor): 标量总损失
            policy_loss (Tensor): 策略损失项
            value_loss (Tensor): 价值 MSE 损失项
            entropy_loss (Tensor): 熵正则项
        """
        # 1. 计算重要性采样比率 ratio: r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))
        # log_prob_diff: [B_mb] -> ratio: [B_mb]
        log_prob_diff = new_logp - old_logp
        ratio = torch.exp(log_prob_diff)
        
        # 2. PPO Clipped Surrogate Loss
        surr1 = ratio * advantages                                                    # [B_mb]
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages # [B_mb]
        policy_loss = -torch.min(surr1, surr2).mean()                                 # 标量
        
        # 3. Value Function Loss (MSE)
        value_loss = nn.functional.mse_loss(values, returns)                           # 标量
        
        # 4. Entropy Loss (鼓励探索)
        entropy_loss = entropy.mean()                                                 # 标量
        
        # 5. 总损失融合
        total_loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
        
        return total_loss, policy_loss, value_loss, entropy_loss


# ======================================================================================
# 8. 训练/推理逻辑与入口 (Training/Inference Execution)
# ======================================================================================
def train_ppo(agent: PPOAgent, env: gym.Env):
    """PPO 训练循环执行入口"""
    loss_fn = PPOLoss(agent.cfg.CLIP_EPS, agent.cfg.VALUE_COEF, agent.cfg.ENTROPY_COEF)
    current_obs, _ = env.reset(seed=agent.cfg.SEED)
    
    print("=" * 70)
    print(f"开始 PPO 训练 Pipeline | 设备: {agent.cfg.DEVICE} | 环境: {agent.cfg.ENV_ID}")
    print("=" * 70)

    for epoch in range(agent.cfg.EPOCHS):
        # 1. 采集 Rollout 经验数据
        current_obs = agent.collect_rollout(env, current_obs)
        
        # 2. 获取 Buffer 张量数据
        obs_t, act_t, old_logp_t, ret_t, adv_t = agent.buffer.get_torch_tensors()
        
        # 3. 优势值 Advantage 批次归一化 (Mini-batch Normalization, 提升数值稳定性)
        adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)
        
        dataset_size = agent.cfg.STEPS_PER_EPOCH
        indices = np.arange(dataset_size)
        
        epoch_policy_loss = 0.0
        epoch_value_loss = 0.0
        epoch_total_loss = 0.0
        updates = 0

        # 4. PPO 多轮 Epoch 参数更新 (Multiple Pass Update)
        for _ in range(agent.cfg.PPO_EPOCHS):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, agent.cfg.MINIBATCH_SIZE):
                end = start + agent.cfg.MINIBATCH_SIZE
                mb_idx = indices[start:end]
                
                # 获取 Mini-batch 切片数据
                mb_obs = obs_t[mb_idx]          # [B_mb, obs_dim]
                mb_act = act_t[mb_idx]          # [B_mb]
                mb_old_logp = old_logp_t[mb_idx]# [B_mb]
                mb_ret = ret_t[mb_idx]          # [B_mb]
                mb_adv = adv_t[mb_idx]          # [B_mb]
                
                # 前向传播重新估算策略对数概率、分布熵与状态价值
                _, new_logp, entropy, values = agent.model.get_action_and_value(mb_obs, mb_act)
                
                # 计算联合损失
                total_loss, pol_loss, val_loss, _ = loss_fn(
                    new_logp=new_logp,
                    old_logp=mb_old_logp,
                    advantages=mb_adv,
                    values=values,
                    returns=mb_ret,
                    entropy=entropy
                )
                
                # 反向传播与梯度更新
                agent.optimizer.zero_grad()
                total_loss.backward()
                # 梯度裁剪 (Gradient Clipping)
                nn.utils.clip_grad_norm_(agent.model.parameters(), agent.cfg.MAX_GRAD_NORM)
                agent.optimizer.step()
                
                epoch_total_loss += total_loss.item()
                epoch_policy_loss += pol_loss.item()
                epoch_value_loss += val_loss.item()
                updates += 1

        # 打印日志信息
        avg_return = ret_t.mean().item()
        avg_loss = epoch_total_loss / updates
        avg_pol_loss = epoch_policy_loss / updates
        avg_val_loss = epoch_value_loss / updates

        print(f"Epoch [{epoch + 1:02d}/{agent.cfg.EPOCHS:02d}] | "
              f"Mean Return: {avg_return:6.2f} | "
              f"Total Loss: {avg_loss:6.4f} | "
              f"Policy Loss: {avg_pol_loss:6.4f} | "
              f"Value Loss: {avg_val_loss:6.4f}")


def main():
    """程序主运行入口"""
    env = gym.make(CONFIG.ENV_ID)
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPOAgent(obs_dim, action_dim, CONFIG)
    train_ppo(agent, env)
    
    env.close()


if __name__ == "__main__":
    main()

```

---

### Part 2: Markdown 技术说明文档

# PPO (Proximal Policy Optimization) 技术架构与接口文档

## 1. 架构总览

PPO 是一种**在策略 (On-Policy)** 的 Actor-Critic 强化学习 Pipeline。整体交互与训练数据流如下：

```
+-----------------------------------------------------------------------------------+
|                                 PPO Pipeline                                      |
+-----------------------------------------------------------------------------------+
                                                                                     
  +------------------+         Action a_t         +-------------------+              
  |                  | -------------------------> |                   |              
  |  Actor-Critic    |                            |    Environment    |              
  |     Network      | <------------------------- |   (CartPole-v1)   |              
  +------------------+      Obs s_t, Reward r_t   +-------------------+              
       |          |                                         |                        
       |          +--------------------+                    |                        
 logits| V(s)                          | Transition         |                        
       v                               v                    v                        
+---------------+             +---------------------------------------+              
| Categorical   |             |            Rollout Buffer             |              
| Distribution  |             | (s_t, a_t, r_t, V_t, log_p_t, d_t)    |              
+---------------+             +---------------------------------------+              
                                                  |                                  
                                                  v  GAE Compute                     
                                      +-----------------------+                      
                                      | Advantage A_t & Ret R |                      
                                      +-----------------------+                      
                                                  |                                  
                                                  v  Mini-batch                      
                                      +-----------------------+                      
                                      |   PPO Loss & Update   |                      
                                      +-----------------------+                      

```

---

## 2. 张量 Shape 流动追踪 (Tensor Flow Table)

| 节点/模块 | 输入 Shape | 输出 Shape | 说明 / 维度变化原因 |
| --- | --- | --- | --- |
| **Environment Observation** | - | `[4]` | CartPole 原始 4 维状态输入 |
| **Batch Observation Layer** | `[4]` | `[1, 4]` | 扩展 Batch 维度以传入网络计算 |
| **Shared Backbone Layer 1** | `[B, 4]` | `[B, 64]` | 线性映射 `Linear(4, 64)` + SiLU |
| **Shared Backbone Layer 2** | `[B, 64]` | `[B, 64]` | 线性映射 `Linear(64, 64)` + SiLU |
| **Actor Head Layer** | `[B, 64]` | `[B, 2]` | 计算离散动作 logits (`action_dim=2`) |
| **Critic Head Layer** | `[B, 64]` | `[B, 1]` | 状态价值 $V(s)$ 标量预测 |
| **Categorical Distribution** | `[B, 2]` | `Distribution` | 构建离散分布对象 |
| **Action Sampling** | `Distribution` | `[B]` | 从动作概率分布中采样的动作索引 |
| **Log Probability (`log_prob`)** | `Distribution`, `[B]` | `[B]` | 对应动作的对数概率 $\log \pi(a \mid s)$ |
| **Value Squeeze (`squeeze(-1)`)** | `[B, 1]` | `[B]` | 压缩尾部维度，便于和 Return 进行 MSE Loss 计算 |
| **Mini-batch Split** | `[N, 4]` (`N=2048`) | `[B_mb, 4]` (`B_mb=64`) | 训练时拆分为 $64$ 大小的 Mini-batch 数据块 |

---

## 3. 核心公式与代码映射

| 数学原理 / 目标公式 | 对应代码实现名称 / 位置 | 代码表达式 |
| --- | --- | --- |
| **重要性采样比率**<br>

<br>$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{old}}(a_t \mid s_t)}$ | `PPOLoss.forward` | `ratio = torch.exp(new_logp - old_logp)` |
| **PPO 裁剪目标**<br>

<br>$\min(r_t A_t, \text{clip}(r_t, 1-\epsilon, 1+\epsilon)A_t)$ | `PPOLoss.forward` | `surr1 = ratio * advantages`<br>

<br>`surr2 = torch.clamp(ratio, 1-eps, 1+eps) * adv`<br>

<br>`policy_loss = -torch.min(surr1, surr2).mean()` |
| **值函数损失**<br>

<br>$L_{VF} = (V_\theta(s) - R_t)^2$ | `PPOLoss.forward` | `value_loss = nn.functional.mse_loss(values, returns)` |
| **GAE 优势估算**<br>

<br>$\delta_t = r_t + \gamma V(s_{t+1})(1-d_t) - V(s_t)$ | `RolloutBuffer.compute_gae_and_returns` | `delta = rew + gamma * next_val * next_non_term - val` |
| **GAE 递归累加**<br>

<br>$A_t = \delta_t + (\gamma \lambda)(1-d_t) A_{t+1}$ | `RolloutBuffer.compute_gae_and_returns` | `gae = delta + gamma * gae_lambda * next_non_term * gae` |
| **优势值标准化**<br>

<br>$\hat{A}_t = \frac{A_t - \mu_A}{\sigma_A + 10^{-8}}$ | `train_ppo` | `adv_t = (adv_t - adv_t.mean()) / (adv_t.std() + 1e-8)` |