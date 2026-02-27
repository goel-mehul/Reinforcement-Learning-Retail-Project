import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field

from environment.product_catalog import ProductCatalog
from utils.config_loader import load_env_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class InventoryState:
    """Snapshot of one agent's inventory at a point in time."""
    agent_id:        int
    stock:           Dict[int, int]    # product_id -> units on hand
    pipeline:        Dict[int, int]    # product_id -> units on order (in transit)
    stockout_days:   Dict[int, int]    # product_id -> consecutive days out of stock
    holding_cost:    float             # total holding cost today
    stockout_penalty: float            # total stockout penalty today


class InventoryManager:
    """
    Tracks and manages inventory for all agents across all products.

    Key mechanics:
    - Stock depletes as units are sold each day
    - When stock hits reorder_point, a replenishment order is placed
    - Orders arrive after a stochastic lead time (handled by SupplyChain)
    - Holding cost accrues daily on all units in stock
    - Stockout penalty applies when an agent has 0 units of a product
      and demand exists for it — lost sales hurt the reward signal

    This creates the core inventory tension:
        High stock  → high holding costs, capital tied up
        Low stock   → risk of stockout, lost sales, penalty
    """

    def __init__(
        self,
        catalog:   ProductCatalog,
        n_agents:  int,
        config=None,
        seed:      int = 42,
    ):
        self.catalog  = catalog
        self.n_agents = n_agents
        self.config   = config or load_env_config()
        self.rng      = np.random.default_rng(seed)

        # load config parameters
        self.initial_stock     = self.config.get('inventory.initial_stock',     500)
        self.reorder_point     = self.config.get('inventory.reorder_point',     100)
        self.reorder_quantity  = self.config.get('inventory.reorder_quantity',  400)
        self.holding_cost_per  = self.config.get('inventory.holding_cost_per_unit', 0.01)
        self.stockout_penalty  = self.config.get('inventory.stockout_penalty',  5.0)

        self.product_ids = [p.product_id for p in catalog.get_all_products()]
        self.n_products  = len(self.product_ids)

        # core state — initialized in reset()
        self.stock:         Dict[int, Dict[int, int]] = {}  # agent -> product -> units
        self.pipeline:      Dict[int, Dict[int, int]] = {}  # agent -> product -> units incoming
        self.stockout_days: Dict[int, Dict[int, int]] = {}  # agent -> product -> days out

        self.reset()

        logger.info(
            f"InventoryManager initialized | "
            f"agents={n_agents} | products={self.n_products} | "
            f"initial_stock={self.initial_stock} | "
            f"reorder_point={self.reorder_point}"
        )

    def reset(self):
        """Reset all inventory to initial state. Called at episode start."""
        self.stock = {
            aid: {pid: self.initial_stock for pid in self.product_ids}
            for aid in range(self.n_agents)
        }
        self.pipeline = {
            aid: {pid: 0 for pid in self.product_ids}
            for aid in range(self.n_agents)
        }
        self.stockout_days = {
            aid: {pid: 0 for pid in self.product_ids}
            for aid in range(self.n_agents)
        }

    def step(
        self,
        units_sold: Dict[int, Dict[int, int]],  # from DemandResult
        incoming:   Dict[int, Dict[int, int]],  # units arriving today from supply chain
    ) -> Dict[int, InventoryState]:
        """
        Process one day of inventory:
          1. Receive incoming stock from supply chain
          2. Deplete stock by units sold
          3. Update stockout tracking
          4. Trigger reorders for low-stock products
          5. Compute holding costs and stockout penalties
          6. Return per-agent inventory states

        Args:
            units_sold: units sold per agent per product (from demand model)
            incoming:   units arriving today (from supply chain)

        Returns:
            Dict of InventoryState per agent
        """
        states = {}

        for aid in range(self.n_agents):
            daily_holding  = 0.0
            daily_stockout = 0.0
            reorder_list   = []

            for pid in self.product_ids:
                # ── 1. receive incoming stock ──────────────────────
                arrived = incoming.get(aid, {}).get(pid, 0)
                self.stock[aid][pid]    += arrived
                self.pipeline[aid][pid] = max(0, self.pipeline[aid][pid] - arrived)

                # ── 2. deplete stock ───────────────────────────────
                sold = units_sold.get(aid, {}).get(pid, 0)
                actual_sold = min(sold, self.stock[aid][pid])  # can't sell what you don't have
                self.stock[aid][pid] -= actual_sold

                # ── 3. track stockouts ─────────────────────────────
                if self.stock[aid][pid] == 0 and sold > 0:
                    # had demand but ran out
                    self.stockout_days[aid][pid] += 1
                    daily_stockout += self.stockout_penalty
                else:
                    self.stockout_days[aid][pid] = 0

                # ── 4. trigger reorder if needed ───────────────────
                total_available = self.stock[aid][pid] + self.pipeline[aid][pid]
                if total_available <= self.reorder_point:
                    reorder_list.append(pid)

                # ── 5. holding cost on remaining stock ─────────────
                daily_holding += self.stock[aid][pid] * self.holding_cost_per

            states[aid] = InventoryState(
                agent_id=aid,
                stock=dict(self.stock[aid]),
                pipeline=dict(self.pipeline[aid]),
                stockout_days=dict(self.stockout_days[aid]),
                holding_cost=daily_holding,
                stockout_penalty=daily_stockout,
            )

            if reorder_list:
                logger.debug(
                    f"Agent {aid} reordering {len(reorder_list)} products: "
                    f"{reorder_list[:5]}{'...' if len(reorder_list) > 5 else ''}"
                )

        return states

    def place_reorders(self) -> Dict[int, Dict[int, int]]:
        """
        Identify which products each agent needs to reorder.
        Returns reorder quantities — supply chain will handle the delay.
        """
        reorders = {}
        for aid in range(self.n_agents):
            reorders[aid] = {}
            for pid in self.product_ids:
                total_available = self.stock[aid][pid] + self.pipeline[aid][pid]
                if total_available <= self.reorder_point:
                    reorders[aid][pid]          = self.reorder_quantity
                    self.pipeline[aid][pid]     += self.reorder_quantity
        return reorders

    # ── Observation helpers ───────────────────────────────────────

    def get_stock_vector(self, agent_id: int) -> np.ndarray:
        """Returns normalized stock levels as a numpy array for RL observation."""
        raw = np.array([self.stock[agent_id][pid] for pid in self.product_ids],
                       dtype=np.float32)
        return raw / self.initial_stock   # normalize to [0, ~1]

    def get_stockout_vector(self, agent_id: int) -> np.ndarray:
        """Returns binary stockout flags per product."""
        return np.array(
            [1.0 if self.stock[agent_id][pid] == 0 else 0.0
             for pid in self.product_ids],
            dtype=np.float32
        )

    def get_state_summary(self, agent_id: int) -> dict:
        """Human-readable summary of an agent's inventory position."""
        stock   = self.stock[agent_id]
        total   = sum(stock.values())
        out     = sum(1 for v in stock.values() if v == 0)
        low     = sum(1 for v in stock.values() if 0 < v <= self.reorder_point)
        return {
            "total_units":    total,
            "products_out":   out,
            "products_low":   low,
            "products_healthy": self.n_products - out - low,
        }

    def __repr__(self) -> str:
        return (f"InventoryManager("
                f"agents={self.n_agents}, products={self.n_products})")