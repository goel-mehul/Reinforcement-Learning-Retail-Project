import numpy as np
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any
from utils.logger import get_logger

logger = get_logger(__name__)


class BaseRLAgent(ABC):
    """
    Abstract base class for all RL agents in RetailRL.

    All four algorithms (DQN, PPO, A2C, Q-Table) inherit from this.
    The training loop only calls methods defined here, so agents
    are fully interchangeable from the training loop's perspective.

    Subclasses must implement:
        act()   — select action given observation
        learn() — update policy from experience
        save()  — persist model to disk
        load()  — restore model from disk
    """

    def __init__(
        self,
        agent_id:    int,
        agent_name:  str,
        obs_size:    int,
        n_products:  int,
        reward_fn:   str,
        config:      Dict[str, Any],
        seed:        int = 42,
    ):
        self.agent_id   = agent_id
        self.agent_name = agent_name
        self.obs_size   = obs_size
        self.n_products = n_products
        self.reward_fn  = reward_fn
        self.config     = config
        self.seed       = seed

        # training state
        self.total_steps    = 0
        self.total_episodes = 0
        self.training       = True

        self.logger = get_logger(f"{__name__}.{agent_name}")
        self.logger.info(
            f"{self.__class__.__name__} initialized | "
            f"id={agent_id} | reward_fn={reward_fn}"
        )

    # ── Required interface ────────────────────────────────────────

    @abstractmethod
    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Select an action given the current observation.

        Args:
            observation: numpy array of shape (obs_size,)

        Returns:
            action: numpy array of shape (n_products,)
                    each element in [0, 6] representing price adjustment
        """

    @abstractmethod
    def learn(self, *args, **kwargs) -> Optional[Dict[str, float]]:
        """
        Update the policy from collected experience.

        Returns:
            dict of training metrics (loss, etc.) or None
        """

    @abstractmethod
    def save(self, path: Path) -> None:
        """Persist model weights and state to disk."""

    @abstractmethod
    def load(self, path: Path) -> None:
        """Restore model weights and state from disk."""

    # ── Optional hooks ────────────────────────────────────────────

    def on_episode_start(self) -> None:
        """Called at the start of each episode. Override if needed."""
        pass

    def on_episode_end(self, episode_reward: float) -> None:
        """Called at the end of each episode with total reward."""
        self.total_episodes += 1

    def set_training(self, training: bool) -> None:
        """Switch between training and evaluation mode."""
        self.training = training

    # ── Shared utilities ──────────────────────────────────────────

    def preprocess_obs(self, observation: np.ndarray) -> np.ndarray:
        """
        Clip and normalize observation to prevent numerical issues.
        Shared across all agents.
        """
        return np.clip(observation, -10.0, 10.0).astype(np.float32)

    def get_stats(self) -> Dict[str, Any]:
        """Returns training stats for logging."""
        return {
            "agent_id":      self.agent_id,
            "agent_name":    self.agent_name,
            "algorithm":     self.__class__.__name__,
            "reward_fn":     self.reward_fn,
            "total_steps":   self.total_steps,
            "total_episodes": self.total_episodes,
        }

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"id={self.agent_id}, "
            f"name={self.agent_name}, "
            f"reward={self.reward_fn}, "
            f"steps={self.total_steps})"
        )