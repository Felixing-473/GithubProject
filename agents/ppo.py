import math
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, action_dims):
        super().__init__()
        # action_dims: list of discrete sizes (here [3,3,3])
        self.obs_dim = obs_dim
        self.action_dims = action_dims
        self.shared = nn.Sequential(nn.Linear(obs_dim, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU())
        # actor heads
        self.actors = nn.ModuleList([nn.Linear(64, a) for a in action_dims])
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        h = self.shared(x)
        logits = [head(h) for head in self.actors]
        value = self.critic(h).squeeze(-1)
        return logits, value


class PPOTrainer:
    def __init__(self, env, lr=3e-4, gamma=0.99, clip_eps=0.2, epochs=4, batch_size=1024, minibatch=256, device=None):
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.action_dims = [3, 3, 3]
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ActorCritic(self.obs_dim, self.action_dims).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.epochs = epochs
        self.batch_size = batch_size
        self.minibatch = minibatch

    def select_action(self, obs):
        obs_v = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.model(obs_v)
        actions = []
        logp = 0.0
        for logit in logits:
            probs = torch.softmax(logit, dim=-1)
            dist = Categorical(probs)
            a = dist.sample()
            actions.append(int(a.item()))
            logp += dist.log_prob(a)
        return np.array(actions, dtype=np.int32), float(value.item()), float(logp.item())

    def compute_gae(self, rewards, values, dones, last_value, lam=0.95):
        values = np.append(values, last_value)
        gae = 0
        returns = np.zeros_like(rewards)
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * values[step + 1] * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * lam * (1 - dones[step]) * gae
            returns[step] = gae + values[step]
        return returns

    def update(self, trajectories):
        obs = torch.from_numpy(np.vstack(trajectories["obs"])).float().to(self.device)
        actions = torch.from_numpy(np.vstack(trajectories["actions"]).astype(np.int64)).to(self.device)
        old_logps = torch.from_numpy(np.vstack(trajectories["logps"]).astype(np.float32)).to(self.device)
        returns = torch.from_numpy(np.vstack(trajectories["returns"]).astype(np.float32)).to(self.device).squeeze(-1)
        advantages = returns - torch.from_numpy(np.vstack(trajectories["values"]).astype(np.float32)).to(self.device).squeeze(-1)

        dataset_size = obs.shape[0]
        for _ in range(self.epochs):
            idxs = np.arange(dataset_size)
            np.random.shuffle(idxs)
            for start in range(0, dataset_size, self.minibatch):
                mb_idx = idxs[start:start + self.minibatch]
                mb_obs = obs[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_logp = old_logps[mb_idx]
                mb_returns = returns[mb_idx]
                mb_adv = advantages[mb_idx]

                logits, values = self.model(mb_obs)
                # compute logp for multi-discrete actions
                logp = 0
                for i, head in enumerate(logits):
                    dist = Categorical(logits=head)
                    logp = logp + dist.log_prob(mb_actions[:, i])
                ratio = torch.exp(logp - mb_old_logp.squeeze(-1))
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = (mb_returns - values).pow(2).mean()
                loss = policy_loss + 0.5 * value_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

    def train(self, total_steps=50000, log_interval=1000, save_path=None):
        obs = self.env.reset()
        ep_reward = 0
        trajectories = {"obs": [], "actions": [], "logps": [], "rewards": [], "values": [], "dones": [], "returns": []}
        steps = 0
        while steps < total_steps:
            actions, value, logp = self.select_action(obs)
            action_dict = {"step_action": actions, "pre_action": 0}
            next_obs, reward, done, info = self.env.step(action_dict)
            trajectories["obs"].append(obs)
            trajectories["actions"].append(actions)
            trajectories["logps"].append([logp])
            trajectories["rewards"].append(reward)
            trajectories["values"].append(value)
            trajectories["dones"].append(1 if done else 0)

            obs = next_obs
            ep_reward += reward
            steps += 1

            if done or len(trajectories["obs"]) >= self.batch_size:
                # bootstrap last value
                last_value = 0.0
                if not done:
                    _, last_value, _ = self.select_action(obs)
                returns = self.compute_gae(np.array(trajectories["rewards"]), np.array(trajectories["values"]), np.array(trajectories["dones"]), last_value)
                trajectories["returns"] = returns.tolist()
                # prepare returns shape
                self.update(trajectories)
                trajectories = {"obs": [], "actions": [], "logps": [], "rewards": [], "values": [], "dones": [], "returns": []}
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
            if done:
                obs = self.env.reset()
                ep_reward = 0

        if save_path:
            torch.save(self.model.state_dict(), save_path)
