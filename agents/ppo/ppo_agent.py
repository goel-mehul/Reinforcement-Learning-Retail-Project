import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

from agents.base_agent import BaseRLAgent
from agents.registry import register


class ActorCriticNetwork(nn.Module):
    """
    Shared backbone with separate actor and critic heads.

    Actor:  outputs logits for each product's price action → (n_products, n_actions)
    Critic: outputs scalar state value → (1,)

    Shared backbone encourages learning useful representations
    that benefit both policy and value estimation.
    """

    def __init__(self, obs_size: int, n_products: int, n_actions: int):
        super().__init__()
        self.n_products = n_products
        self.n_actions  = n_actions

        # shared feature extractor
        self.backbone = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
        )

        # actor head — one set of logits per product
        self.actor = nn.Linear(256, n_products * n_actions)

        # critic head — scalar value estimate
        self.critic = nn.Linear(256, 1)

        # initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
                nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, x: torch.Tensor):
        features    = self.backbone(x)
        logits      = self.actor(features).view(-1, self.n_products, self.n_actions)
        value       = self.critic(features).squeeze(-1)
        return logits, value

    def get_action(self, x: torch.Tensor):
        """Sample actions and compute log probs for PPO update."""
        logits, value = self.forward(x)
        dist    = Categorical(logits=logits)
        actions = dist.sample()                        # (batch, n_products)
        log_prob = dist.log_prob(actions).sum(dim=-1)  # sum over products
        entropy  = dist.entropy().sum(dim=-1)
        return actions, log_prob, entropy, value


@register("ppo")
class PPOAgent(BaseRLAgent):
    """
    Proximal Policy Optimization agent for retail pricing.

    Uses:
    - Actor-critic network with shared backbone
    - GAE (Generalized Advantage Estimation) for variance reduction
    - Clipped surrogate objective (epsilon=0.2)
    - Entropy bonus to encourage exploration
    - Multiple epochs of minibatch updates per rollout

    Assigned to: Target (profit_margin reward)
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

        # hyperparameters
        self.lr           = config.get('lr',           3e-4)
        self.gamma        = config.get('gamma',        0.99)
        self.gae_lambda   = config.get('gae_lambda',   0.95)
        self.clip_eps     = config.get('clip_eps',     0.2)
        self.n_epochs     = config.get('n_epochs',     4)
        self.batch_size   = config.get('batch_size',   64)
        self.vf_coef      = config.get('vf_coef',      0.5)
        self.ent_coef     = config.get('ent_coef',     0.01)
        self.max_grad     = config.get('max_grad',     0.5)
        self.rollout_len  = config.get('rollout_len',  128)
        self.n_actions    = 7

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.net       = ActorCriticNetwork(obs_size, n_products, self.n_actions).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=self.lr, eps=1e-5)

        # rollout buffer
        self._reset_rollout()

        self.logger.info(
            f"PPO initialized | device={self.device} | "
            f"lr={self.lr} | clip={self.clip_eps} | "
            f"rollout={self.rollout_len}"
        )

    def _reset_rollout(self):
        self.rollout = {
            "obs":      [],
            "actions":  [],
            "log_probs": [],
            "rewards":  [],
            "values":   [],
            "dones":    [],
        }

    def act(self, observation: np.ndarray) -> np.ndarray:
        obs = self.preprocess_obs(observation)
        self.total_steps += 1

        obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            actions, log_prob, _, value = self.net.get_action(obs_t)

        self._last_obs      = obs
        self._last_action   = actions.squeeze(0).cpu().numpy()
        self._last_log_prob = log_prob.squeeze(0).cpu().item()
        self._last_value    = value.squeeze(0).cpu().item()

        return self._last_action

    def learn(
        self,
        reward: float,
        done:   bool,
        next_obs: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, float]]:
        # store transition using values from last act()
        self.rollout["obs"].append(self._last_obs)
        self.rollout["actions"].append(self._last_action)
        self.rollout["log_probs"].append(self._last_log_prob)
        self.rollout["values"].append(self._last_value)
        self.rollout["rewards"].append(reward)
        self.rollout["dones"].append(float(done))

        if len(self.rollout["rewards"]) < self.rollout_len and not done:
            return None

        return self._update(next_obs)

    def _update(self, next_obs: Optional[np.ndarray]) -> Dict[str, float]:
        """Run PPO update on collected rollout."""
        # compute GAE advantages
        rewards  = np.array(self.rollout["rewards"],   dtype=np.float32)
        values   = np.array(self.rollout["values"],    dtype=np.float32)
        dones    = np.array(self.rollout["dones"],     dtype=np.float32)

        # bootstrap value for last state
        if next_obs is not None and not dones[-1]:
            next_t = torch.FloatTensor(
                self.preprocess_obs(next_obs)
            ).unsqueeze(0).to(self.device)
            with torch.no_grad():
                _, next_val = self.net(next_t)
            next_value = next_val.item()
        else:
            next_value = 0.0

        advantages = self._compute_gae(rewards, values, dones, next_value)
        returns    = advantages + values

        # normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # convert rollout to tensors
        obs_t    = torch.FloatTensor(np.array(self.rollout["obs"])).to(self.device)
        act_t    = torch.LongTensor(np.array(self.rollout["actions"])).to(self.device)
        old_lp_t = torch.FloatTensor(np.array(self.rollout["log_probs"])).to(self.device)
        adv_t    = torch.FloatTensor(advantages).to(self.device)
        ret_t    = torch.FloatTensor(returns).to(self.device)

        # PPO epochs
        total_loss = total_pg = total_vf = total_ent = 0.0
        n = len(rewards)

        for _ in range(self.n_epochs):
            idx = np.random.permutation(n)
            for start in range(0, n, self.batch_size):
                mb = idx[start:start + self.batch_size]

                _, new_lp, entropy, values_pred = self.net.get_action(obs_t[mb])
                new_lp = new_lp

                ratio    = (new_lp - old_lp_t[mb]).exp()
                pg_loss1 = -adv_t[mb] * ratio
                pg_loss2 = -adv_t[mb] * ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps)
                pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

                vf_loss  = nn.MSELoss()(values_pred, ret_t[mb])
                ent_loss = -entropy.mean()

                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * ent_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad)
                self.optimizer.step()

                total_loss += loss.item()
                total_pg   += pg_loss.item()
                total_vf   += vf_loss.item()
                total_ent  += ent_loss.item()

        self._reset_rollout()
        n_updates = self.n_epochs * max(1, n // self.batch_size)

        return {
            "loss":     total_loss / n_updates,
            "pg_loss":  total_pg   / n_updates,
            "vf_loss":  total_vf   / n_updates,
            "entropy":  -total_ent / n_updates,
        }

    def _compute_gae(
        self,
        rewards:    np.ndarray,
        values:     np.ndarray,
        dones:      np.ndarray,
        next_value: float,
    ) -> np.ndarray:
        """Generalized Advantage Estimation."""
        n          = len(rewards)
        advantages = np.zeros(n, dtype=np.float32)
        last_gae   = 0.0

        for t in reversed(range(n)):
            next_val = next_value if t == n - 1 else values[t + 1]
            next_done = dones[t + 1] if t < n - 1 else 0.0
            delta    = rewards[t] + self.gamma * next_val * (1 - next_done) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1 - next_done) * last_gae
            advantages[t] = last_gae

        return advantages

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "net":         self.net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }, path / f"ppo_{self.agent_name}.pt")

    def load(self, path: Path) -> None:
        ckpt = torch.load(
            Path(path) / f"ppo_{self.agent_name}.pt",
            map_location=self.device
        )
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]