import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import pickle

from agents.base_agent import BaseRLAgent
from agents.registry import register


def discretize_obs(observation: np.ndarray, n_bins: int = 5) -> tuple:
    """
    Reduce 266-dim observation to a small discrete state for Q-table.

    Uses only the most informative features:
        - own avg price level       (5 bins)
        - competitor avg price      (5 bins)  
        - avg stock level           (3 bins: low/med/high)
        - any stockout flag         (2 bins: yes/no)
        - is weekend                (2 bins)
        - days to next promo        (3 bins: soon/medium/far)

    Total states: 5 * 5 * 3 * 2 * 2 * 3 = 900 states
    """
    own_prices  = observation[0:50]
    comp_prices = observation[50:100]
    stock       = observation[100:150]
    stockouts   = observation[150:200]
    promo_vec   = observation[200:205]

    own_avg  = np.mean(own_prices)
    comp_avg = np.mean(comp_prices)
    stk_avg  = np.mean(stock)
    any_out  = int(stockouts.sum() > 0)
    weekend  = int(promo_vec[0] > 0.5)
    days_promo = promo_vec[3]   # normalized days to next promo

    own_bin  = min(int(own_avg  * n_bins), n_bins - 1)
    comp_bin = min(int(comp_avg * n_bins), n_bins - 1)
    stk_bin  = 0 if stk_avg < 0.3 else (1 if stk_avg < 0.7 else 2)
    promo_bin = 0 if days_promo < 0.2 else (1 if days_promo < 0.6 else 2)

    return (own_bin, comp_bin, stk_bin, any_out, weekend, promo_bin)


@register("qtable")
class QTableAgent(BaseRLAgent):
    """
    Tabular Q-learning agent with discretized state space.

    Uses a shared action for all products (one price decision
    applies to all 50 products simultaneously). This is a
    simplification but makes the table tractable.

    Action space: single action in [0,6] applied to all products.

    Assigned to: QFC (revenue_with_inventory reward)
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

        self.rng       = np.random.default_rng(seed)
        self.lr        = config.get('lr',        0.1)
        self.gamma     = config.get('gamma',     0.99)
        self.epsilon   = config.get('epsilon_start', 1.0)
        self.eps_end   = config.get('epsilon_end',   0.05)
        self.eps_decay = config.get('epsilon_decay', 5000)
        self.n_actions = 7
        self.n_bins    = config.get('n_bins', 5)

        # Q-table: dict mapping state tuple -> Q-values array (n_actions,)
        self.q_table: Dict[tuple, np.ndarray] = {}

        # last transition for learning
        self._last_state  = None
        self._last_action = None

        self.logger.info(
            f"QTable initialized | "
            f"lr={self.lr} | epsilon={self.epsilon} | "
            f"n_bins={self.n_bins}"
        )

    def _get_q(self, state: tuple) -> np.ndarray:
        """Get Q-values for state, initializing to zeros if unseen."""
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.n_actions, dtype=np.float32)
        return self.q_table[state]

    def act(self, observation: np.ndarray) -> np.ndarray:
        """
        Epsilon-greedy over discretized state.
        Returns same action for all 50 products.
        """
        obs   = self.preprocess_obs(observation)
        state = discretize_obs(obs, self.n_bins)
        self.total_steps += 1
        self._update_epsilon()

        if self.training and self.rng.random() < self.epsilon:
            action_idx = int(self.rng.integers(0, self.n_actions))
        else:
            action_idx = int(np.argmax(self._get_q(state)))

        self._last_state  = state
        self._last_action = action_idx

        return np.full(self.n_products, action_idx, dtype=int)

    def learn(
        self,
        reward:   float,
        done:     bool,
        next_obs: Optional[np.ndarray] = None,
    ) -> Optional[Dict[str, float]]:
        """Tabular Q-learning update."""
        if self._last_state is None:
            return None

        next_state = discretize_obs(
            self.preprocess_obs(next_obs), self.n_bins
        ) if next_obs is not None else self._last_state

        q_vals      = self._get_q(self._last_state)
        next_q_vals = self._get_q(next_state)

        td_target = reward + self.gamma * np.max(next_q_vals) * (1 - float(done))
        td_error  = td_target - q_vals[self._last_action]
        q_vals[self._last_action] += self.lr * td_error

        return {
            "td_error": float(td_error),
            "epsilon":  self.epsilon,
            "n_states": len(self.q_table),
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / f"qtable_{self.agent_name}.pkl", "wb") as f:
            pickle.dump({
                "q_table":     self.q_table,
                "total_steps": self.total_steps,
                "epsilon":     self.epsilon,
            }, f)

    def load(self, path: Path) -> None:
        with open(Path(path) / f"qtable_{self.agent_name}.pkl", "rb") as f:
            ckpt = pickle.load(f)
        self.q_table     = ckpt["q_table"]
        self.total_steps = ckpt["total_steps"]
        self.epsilon     = ckpt["epsilon"]

    def _update_epsilon(self):
        progress     = min(self.total_steps / self.eps_decay, 1.0)
        self.epsilon = 1.0 + progress * (self.eps_end - 1.0)