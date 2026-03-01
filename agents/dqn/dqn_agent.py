import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
import random

from agents.base_agent import BaseRLAgent
from agents.registry import register


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    Stores (obs, action, reward, next_obs, done) tuples.
    Sampling breaks temporal correlations in training data.
    """

    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)

    def push(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ):
        self.buffer.append((obs, action, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        obs, actions, rewards, next_obs, dones = zip(*batch)
        return (
            np.array(obs,      dtype=np.float32),
            np.array(actions,  dtype=np.int64),
            np.array(rewards,  dtype=np.float32),
            np.array(next_obs, dtype=np.float32),
            np.array(dones,    dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class QNetwork(nn.Module):
    """
    Shared Q-network backbone.
    Takes observation and outputs Q-values for all 50 products × 7 actions.

    Architecture:
        Input  (266,)
        Linear  → 256 → ReLU
        Linear  → 256 → ReLU
        Linear  → 128 → ReLU
        Output  → 50 * 7 = 350  (Q-values for every product-action pair)

    We reshape output to (50, 7) and take argmax per product.
    """

    def __init__(self, obs_size: int, n_products: int, n_actions: int):
        super().__init__()
        self.n_products = n_products
        self.n_actions  = n_actions

        self.net = nn.Sequential(
            nn.Linear(obs_size, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, n_products * n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # returns shape (batch, n_products, n_actions)
        q = self.net(x)
        return q.view(-1, self.n_products, self.n_actions)


@register("dqn")
class DQNAgent(BaseRLAgent):
    """
    Deep Q-Network agent for retail pricing.

    Uses:
    - Shared Q-network over factored action space (50 products × 7 actions)
    - Experience replay buffer (50k transitions)
    - Target network with periodic hard updates
    - Epsilon-greedy exploration with linear decay

    Assigned to: Walmart (pure_revenue reward)
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
        random.seed(seed)

        # hyperparameters from config
        self.lr             = config.get('lr',             1e-4)
        self.gamma          = config.get('gamma',          0.99)
        self.buffer_size    = config.get('buffer_size',    50_000)
        self.batch_size     = config.get('batch_size',     256)
        self.target_update  = config.get('target_update',  500)   # steps
        self.epsilon_start  = config.get('epsilon_start',  1.0)
        self.epsilon_end    = config.get('epsilon_end',    0.05)
        self.epsilon_decay  = config.get('epsilon_decay',  10_000)
        self.min_buffer     = config.get('min_buffer',     1_000)
        self.n_actions      = 7

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # networks
        self.q_net      = QNetwork(obs_size, n_products, self.n_actions).to(self.device)
        self.target_net = QNetwork(obs_size, n_products, self.n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.buffer    = ReplayBuffer(self.buffer_size)

        # epsilon tracking
        self.epsilon = self.epsilon_start

        # training metrics
        self.losses: List[float] = []

        self.logger.info(
            f"DQN initialized | device={self.device} | "
            f"lr={self.lr} | buffer={self.buffer_size} | "
            f"batch={self.batch_size}"
        )

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Epsilon-greedy action selection.
        With prob epsilon: random actions (explore)
        With prob 1-epsilon: greedy Q-network actions (exploit)
        """
        obs = self.preprocess_obs(observation)
        self.total_steps += 1
        self._update_epsilon()

        if self.training and np.random.random() < self.epsilon:
            return np.random.randint(0, self.n_actions, size=self.n_products)

        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            q_vals = self.q_net(obs_t)   # (1, n_products, n_actions)
            actions = q_vals.argmax(dim=2).squeeze(0).cpu().numpy()
        return actions

    def learn(
        self,
        obs:      np.ndarray,
        action:   np.ndarray,
        reward:   float,
        next_obs: np.ndarray,
        done:     bool,
    ) -> Optional[Dict[str, float]]:
        """
        Store transition in replay buffer and train if buffer is ready.

        Returns training metrics if a learning step occurred, else None.
        """
        self.buffer.push(obs, action, reward, next_obs, done)

        if len(self.buffer) < self.min_buffer:
            return None

        # sample batch and compute loss
        obs_b, act_b, rew_b, next_b, done_b = self.buffer.sample(self.batch_size)

        obs_t    = torch.FloatTensor(obs_b).to(self.device)
        act_t    = torch.LongTensor(act_b).to(self.device)
        rew_t    = torch.FloatTensor(rew_b).to(self.device)
        next_t   = torch.FloatTensor(next_b).to(self.device)
        done_t   = torch.FloatTensor(done_b).to(self.device)

        # current Q values — gather action taken for each product
        q_current = self.q_net(obs_t)   # (batch, n_products, n_actions)
        q_taken   = q_current.gather(
            2, act_t.unsqueeze(2)
        ).squeeze(2)   # (batch, n_products)

        # target Q values via Bellman equation
        with torch.no_grad():
            q_next   = self.target_net(next_t).max(dim=2).values
            q_target = rew_t.unsqueeze(1) + self.gamma * q_next * (1 - done_t.unsqueeze(1))

        loss = nn.MSELoss()(q_taken, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
        self.optimizer.step()

        # periodic target network update
        if self.total_steps % self.target_update == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        loss_val = loss.item()
        self.losses.append(loss_val)
        return {"loss": loss_val, "epsilon": self.epsilon}

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({
            "q_net":       self.q_net.state_dict(),
            "target_net":  self.target_net.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "epsilon":     self.epsilon,
        }, path / f"dqn_{self.agent_name}.pt")
        self.logger.info(f"DQN saved to {path}")

    def load(self, path: Path) -> None:
        ckpt = torch.load(
            Path(path) / f"dqn_{self.agent_name}.pt",
            map_location=self.device
        )
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]
        self.epsilon     = ckpt["epsilon"]
        self.logger.info(f"DQN loaded from {path}")

    def _update_epsilon(self):
        """Linear epsilon decay from epsilon_start to epsilon_end."""
        progress = min(self.total_steps / self.epsilon_decay, 1.0)
        self.epsilon = self.epsilon_start + progress * (
            self.epsilon_end - self.epsilon_start
        )