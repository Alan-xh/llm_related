"""
任务 12：强化学习（Reinforcement Learning）
代表模型：PPO（近端策略优化）
损失函数：策略损失 + 值函数损失
在 CartPole-v1 环境上演示 PPO 训练流程。
"""
import torch
import torch.nn as nn
import numpy as np

# 尝试导入 gymnasium，未安装则回退到 gym
try:
    import gymnasium as gym
except ImportError:
    import gym

# 超参数
EPOCHS = 100
STEPS_PER_EPOCH = 2048
MINIBATCH_SIZE = 64
LR = 3e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
VALUE_COEF = 0.5
ENTROPY_COEF = 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dim):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(obs_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
        )
        self.actor = nn.Linear(64, action_dim)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_action_and_value(self, x, action=None):
        logits, value = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        return action, log_prob, entropy, value.squeeze(-1)


def collect_rollout(env, model, steps):
    obs_list, act_list, logp_list, rew_list, val_list, done_list = [], [], [], [], [], []
    obs, _ = env.reset()

    for _ in range(steps):
        obs_t = torch.tensor(obs, dtype=torch.float32).to(DEVICE)
        with torch.no_grad():
            action, log_prob, _, value = model.get_action_and_value(obs_t)

        action_np = action.cpu().numpy()
        next_obs, reward, terminated, truncated, _ = env.step(action_np)
        done = terminated or truncated

        obs_list.append(obs)
        act_list.append(action_np)
        logp_list.append(log_prob.cpu().numpy())
        rew_list.append(reward)
        val_list.append(value.cpu().numpy())
        done_list.append(float(done))

        obs = next_obs
        if done:
            obs, _ = env.reset()

    obs_list = np.array(obs_list, dtype=np.float32)
    act_list = np.array(act_list, dtype=np.int64)
    logp_list = np.array(logp_list, dtype=np.float32)
    rew_list = np.array(rew_list, dtype=np.float32)
    val_list = np.array(val_list, dtype=np.float32)
    done_list = np.array(done_list, dtype=np.float32)

    # 计算回报和优势（GAE）
    returns = np.zeros_like(rew_list)
    advantages = np.zeros_like(rew_list)
    gae = 0.0
    next_value = 0.0

    for t in reversed(range(steps)):
        if t == steps - 1:
            next_non_terminal = 1.0 - done_list[t]
        else:
            next_non_terminal = 1.0 - done_list[t]
            next_value = val_list[t + 1]

        delta = rew_list[t] + GAMMA * next_value * next_non_terminal - val_list[t]
        gae = delta + GAMMA * GAE_LAMBDA * next_non_terminal * gae
        advantages[t] = gae
        returns[t] = gae + val_list[t]

    return obs_list, act_list, logp_list, returns, advantages


def main():
    env = gym.make("CartPole-v1")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model = ActorCritic(obs_dim, action_dim).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        obs, acts, old_logps, returns, advantages = collect_rollout(
            env, model, STEPS_PER_EPOCH
        )

        obs_t = torch.tensor(obs).to(DEVICE)
        acts_t = torch.tensor(acts).to(DEVICE)
        old_logps_t = torch.tensor(old_logps).to(DEVICE)
        returns_t = torch.tensor(returns).to(DEVICE)
        advantages_t = torch.tensor(advantages).to(DEVICE)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        dataset_size = STEPS_PER_EPOCH
        indices = np.arange(dataset_size)

        total_loss = 0.0
        for _ in range(4):  # 每个 epoch 更新 4 轮
            np.random.shuffle(indices)
            for start in range(0, dataset_size, MINIBATCH_SIZE):
                end = start + MINIBATCH_SIZE
                mb = indices[start:end]

                _, new_logp, entropy, values = model.get_action_and_value(
                    obs_t[mb], acts_t[mb]
                )
                ratio = torch.exp(new_logp - old_logps_t[mb])

                adv = advantages_t[mb]
                surr1 = ratio * adv
                surr2 = torch.clamp(ratio, 1 - CLIP_EPS, 1 + CLIP_EPS) * adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = nn.functional.mse_loss(values, returns_t[mb])

                loss = policy_loss + VALUE_COEF * value_loss - ENTROPY_COEF * entropy.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

        avg_return = returns.mean()
        print(f"Epoch [{epoch + 1}/{EPOCHS}]  Avg Return: {avg_return:.2f}")

    env.close()


if __name__ == "__main__":
    main()
