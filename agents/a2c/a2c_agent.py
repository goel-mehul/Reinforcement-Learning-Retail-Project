import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from pathlib import Path
from typing import Dict, Any, Optional, List

from agents.base_agent import BaseRLAgent
from agents.registry import register


class A2CNetwork(nn.Module):
    """
    Actor-Critic network for A2C.
    Simpler than PPO — no clipping, updates every step.
    """

    def __init__(self, obs_size: int, n_products: int, n_actions: int):
        super().__init__()
        self.n_products = n_products
        self.n_actions  = n_actions

        self.backbone = nn.Sequential(
            nn.Linear(obs_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.actor  = nn.Linear(128, n_products * n_actions)
        self.critic = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor):
        f      = self.backbone(x)
        logits = self.actor(f).view(-1, self.n_products, self.n_actions)
        value  = self.critic(f).squeeze(-1)
        return logits, value


@register("a2c")
class A2CAgent(BaseRLAgent):
    """
    Advantage Actor-Critic agent.

    Updates every n_steps using n-step returns.
    Simpler than PPO: no replay buffer, no clipping.
    Faster iteration but higher variance.

    Assigned to: Amazon Fresh (market_share reward)
    """

    def __init__(
        self,
        agent_id:   int,
        agent_name: str,
        obs_size:   int,
        n_products: int,
        reward_fn:  str,
        config:     Dict[str, Any],
        seed:       int = 42,
    ):
        super().__init__(agent_id, agent_name, obs_size, n_products, reward_fn, config, seed)

        torch.manual_seed(seed)

        self.lr        = config.get('lr',        7e-4)
        self.gamma     = config.get('gamma',     0.99)
        self.vf_coef   = config.get('vf_coef',   0.5)
        self.ent_coef  = config.get('ent_coef',  0.01)
        self.max_grad  = config.get('max_grad',  0.5)
        self.n_steps   = config.get('n_steps',   5)
        self.n_actions = 7

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net       = A2CNetwork(obs_size, n_products, self.n_actions).to(self.device)
        self.optimizer = optim.RMSprop(
            self.net.parameters(), lr=self.lr, eps=1e-5, alpha=0.99
        )

        self._reset_buffer()

        self.logger.info(
            f"A2C initialized | device={self.device} | "
            f"lr={self.lr} | n_steps={self.n_steps}"
        )

    def _reset_buffer(self):
        self.buf = {
            "obs": [], "actions": [], "rewards": [],
            "values": [], "log_probs": [], "dones": []
        }

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = self.preprocess_obs(observation)
        self.total_steps += 1

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, value = self.net(obs_t)
            dist     = Categorical(logits=logits)
            actions  = dist.sample()
            log_prob = dist.log_prob(actions).sum(dim=-1)

        action_np = actions.squeeze(0).cpu().numpy()

        # store for use in learn()
        self._last_obs      = obs
        self._last_action   = action_np
        self._last_value    = value.item()
        self._last_log_prob = log_prob.item()

        return action_np

    def learn(
        self,
        reward:   float,
        done:     bool,
        next_obs: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, float]]:
        # store transition now using values captured in act()
        self.buf["obs"].append(self._last_obs)
        self.buf["actions"].append(self._last_action)
        self.buf["values"].append(self._last_value)
        self.buf["log_probs"].append(self._last_log_prob)
        self.buf["rewards"].append(reward)
        self.buf["dones"].append(float(done))

        if len(self.buf["rewards"]) < self.n_steps and not done:
            return None

        return self._update(next_obs, done)

    def _update(self, next_obs, done) -> Dict[str, float]:
        rewards = np.array(self.buf["rewards"], dtype=np.float32)
        values  = np.array(self.buf["values"],  dtype=np.float32)
        dones   = np.array(self.buf["dones"],   dtype=np.float32)

        # bootstrap
        if next_obs is not None and not done:
            next_t = torch.FloatTensor(
                self.preprocess_obs(next_obs)
            ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, nv = self.net(next_t)
            next_value = nv.item()
        else:
            next_value = 0.0

        # n-step returns
        returns = np.zeros_like(rewards)
        R = next_value
        for t in reversed(range(len(rewards))):
            R = rewards[t] + self.gamma * R * (1 - dones[t])
            returns[t] = R

        advantages = returns - values

        obs_t  = torch.FloatTensor(np.array(self.buf["obs"])).to(self.device)
        act_t  = torch.LongTensor(np.array(self.buf["actions"])).to(self.device)
        ret_t  = torch.FloatTensor(returns).to(self.device)
        adv_t  = torch.FloatTensor(advantages).to(self.device)

        logits, values_pred = self.net(obs_t)
        dist     = Categorical(logits=logits)
        log_prob = dist.log_prob(act_t).sum(dim=-1)
        entropy  = dist.entropy().sum(dim=-1).mean()

        pg_loss  = -(log_prob * adv_t).mean()
        vf_loss  = nn.MSELoss()(values_pred, ret_t)
        loss     = pg_loss + self.vf_coef * vf_loss - self.ent_coef * entropy

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
        self.optimizer.step()

        self._reset_buffer()
        return {
            "loss":    loss.item(),
            "pg_loss": pg_loss.item(),
            "vf_loss": vf_loss.item(),
            "entropy": entropy.item(),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "net":         self.net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path / f"a2c_{self.agent_name}.pt")

    def load(self, path: Path) -> None:
        ckpt = torch.load(
            Path(path) / f"a2c_{self.agent_name}.pt",
            map_location=self.device
        )
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]