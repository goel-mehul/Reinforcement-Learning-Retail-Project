import numpy as np
from typing import Dict
from environment.product_catalog import ProductCatalog


class BaseAgent:
    """Abstract base for all rule-based agents."""

    def __init__(self, agent_id: int, agent_name: str, n_products: int):
        self.agent_id   = agent_id
        self.agent_name = agent_name
        self.n_products = n_products

    def act(self, observation: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(id={self.agent_id}, name={self.agent_name})"


class RandomAgent(BaseAgent):
    """
    Picks a random price adjustment for each product every step.
    Useful as a lower-bound baseline — any trained agent should beat this.
    """

    def __init__(self, agent_id: int, agent_name: str, n_products: int, seed: int = 42):
        super().__init__(agent_id, agent_name, n_products)
        self.rng = np.random.default_rng(seed + agent_id)

    def act(self, observation: np.ndarray) -> np.ndarray:
        return self.rng.integers(0, 7, size=self.n_products)


class FixedMarginAgent(BaseAgent):
    """
    Always prices at base_cost + fixed margin percentage.
    Ignores competitors entirely — pure cost-plus pricing.

    Real-world analog: small retailers who set prices once and forget.
    Action: always return action 3 (hold / 0% change) since prices
    are initialized at retail and this agent never changes them.
    """

    def __init__(
        self,
        agent_id:   int,
        agent_name: str,
        n_products: int,
        margin:     float = 0.30,   # 30% above cost
    ):
        super().__init__(agent_id, agent_name, n_products)
        self.margin = margin

    def act(self, observation: np.ndarray) -> np.ndarray:
        # always hold price — fixed margin is set at initialization
        return np.full(self.n_products, 3, dtype=int)   # action 3 = 0% change


class AlwaysCheapestAgent(BaseAgent):
    """
    Always tries to be the cheapest store in the market.
    Reads competitor average prices from observation and undercuts.

    Observation layout (from RetailEnv):
        obs[0:50]   = own prices (normalized)
        obs[50:100] = competitor avg prices (normalized)

    Strategy: if own price > competitor avg, cut by 10% (action 1)
              if own price <= competitor avg, cut by 5% (action 2)
              never raise prices
    """

    def __init__(self, agent_id: int, agent_name: str, n_products: int):
        super().__init__(agent_id, agent_name, n_products)

    def act(self, observation: np.ndarray) -> np.ndarray:
        own_prices  = observation[0:50]
        comp_prices = observation[50:100]

        actions = np.zeros(self.n_products, dtype=int)
        for i in range(self.n_products):
            if own_prices[i] > comp_prices[i]:
                actions[i] = 1   # -10% (more aggressive cut)
            else:
                actions[i] = 2   # -5%  (maintain cheapest position)
        return actions