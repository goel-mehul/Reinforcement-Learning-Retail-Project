import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from utils.config_loader import load_env_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PendingOrder:
    """An order placed by an agent that hasn't arrived yet."""
    agent_id:    int
    product_id:  int
    quantity:    int
    order_day:   int        # day the order was placed
    arrival_day: int        # day the order will arrive


class SupplyChain:
    """
    Manages supply chain delays for all agents.

    When an agent places a reorder:
      1. A lead time is sampled (2-7 days, stochastic)
      2. The order is placed in a pipeline queue
      3. Each day, orders whose arrival_day == today are delivered
      4. Occasionally, supply disruptions extend lead times

    This forces agents to anticipate inventory needs in advance
    rather than just reacting to stockouts.

    Calibrated config:
        lead_time_min:           2 days
        lead_time_max:           7 days
        disruption_probability:  0.02 (2% chance per day)
        disruption_multiplier:   3.0x lead time on disruption
    """

    def __init__(self, n_agents: int, config=None, seed: int = 42):
        self.n_agents = n_agents
        self.config   = config or load_env_config()
        self.rng      = np.random.default_rng(seed)

        # load config
        self.lead_min        = self.config.get('supply_chain.lead_time_min',         2)
        self.lead_max        = self.config.get('supply_chain.lead_time_max',         7)
        self.disruption_prob = self.config.get('supply_chain.disruption_probability', 0.02)
        self.disruption_mult = self.config.get('supply_chain.disruption_multiplier',  3.0)

        # pipeline: list of all pending orders
        self.pending_orders: List[PendingOrder] = []
        self.current_day = 0

        # track disruption events for logging/analysis
        self.disruption_log: List[dict] = []

        logger.info(
            f"SupplyChain initialized | "
            f"lead_time={self.lead_min}-{self.lead_max} days | "
            f"disruption_prob={self.disruption_prob}"
        )

    def reset(self):
        """Clear all pending orders. Called at episode start."""
        self.pending_orders = []
        self.disruption_log = []
        self.current_day    = 0

    def place_orders(
        self,
        reorders: Dict[int, Dict[int, int]],  # agent_id -> product_id -> quantity
    ):
        """
        Place reorders into the supply chain pipeline.
        Each order gets a stochastic lead time.
        Disruptions can randomly extend lead times.

        Args:
            reorders: reorder quantities from InventoryManager.place_reorders()
        """
        for aid, products in reorders.items():
            for pid, quantity in products.items():
                if quantity <= 0:
                    continue

                lead_time = self._sample_lead_time(aid, pid)
                arrival   = self.current_day + lead_time

                order = PendingOrder(
                    agent_id=aid,
                    product_id=pid,
                    quantity=quantity,
                    order_day=self.current_day,
                    arrival_day=arrival,
                )
                self.pending_orders.append(order)

                logger.debug(
                    f"Order placed | agent={aid} product={pid} "
                    f"qty={quantity} arrives_day={arrival} "
                    f"(lead={lead_time}d)"
                )

    def step(self) -> Dict[int, Dict[int, int]]:
        """
        Advance one day. Deliver orders that arrive today.

        Returns:
            incoming: agent_id -> product_id -> units arriving today
                      (passed directly to InventoryManager.step())
        """
        incoming: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

        delivered = []
        remaining = []

        for order in self.pending_orders:
            if order.arrival_day <= self.current_day:
                incoming[order.agent_id][order.product_id] += order.quantity
                delivered.append(order)
                logger.debug(
                    f"Order delivered | agent={order.agent_id} "
                    f"product={order.product_id} qty={order.quantity}"
                )
            else:
                remaining.append(order)

        self.pending_orders = remaining
        self.current_day   += 1

        # convert defaultdict to plain dict
        return {
            aid: dict(products)
            for aid, products in incoming.items()
        }

    def get_days_until_arrival(
        self, agent_id: int, product_id: int
    ) -> int:
        """
        Returns days until the next order arrives for a given agent/product.
        Returns -1 if no order is pending.
        Useful for RL observation space.
        """
        relevant = [
            o for o in self.pending_orders
            if o.agent_id == agent_id and o.product_id == product_id
        ]
        if not relevant:
            return -1
        earliest = min(o.arrival_day for o in relevant)
        return max(0, earliest - self.current_day)

    def get_pipeline_vector(
        self, agent_id: int, product_ids: List[int]
    ) -> np.ndarray:
        """
        Returns normalized expected incoming units per product.
        Used as part of the RL observation vector.
        """
        pipeline = np.zeros(len(product_ids), dtype=np.float32)
        for order in self.pending_orders:
            if order.agent_id == agent_id:
                if order.product_id in product_ids:
                    idx = product_ids.index(order.product_id)
                    pipeline[idx] += order.quantity
        return pipeline / 400.0   # normalize by reorder_quantity

    def n_pending_orders(self, agent_id: int = None) -> int:
        """Count pending orders, optionally filtered by agent."""
        if agent_id is None:
            return len(self.pending_orders)
        return sum(1 for o in self.pending_orders if o.agent_id == agent_id)

    # ── Private helpers ───────────────────────────────────────────

    def _sample_lead_time(self, agent_id: int, product_id: int) -> int:
        """
        Sample a lead time with optional disruption.
        Normal: Uniform(lead_min, lead_max)
        Disruption: multiplies lead time by disruption_multiplier
        """
        base_lead = int(self.rng.integers(self.lead_min, self.lead_max + 1))

        # check for supply disruption
        if self.rng.random() < self.disruption_prob:
            disrupted_lead = int(base_lead * self.disruption_mult)
            self.disruption_log.append({
                "day":        self.current_day,
                "agent_id":   agent_id,
                "product_id": product_id,
                "base_lead":  base_lead,
                "actual_lead": disrupted_lead,
            })
            logger.warning(
                f"Supply disruption! agent={agent_id} product={product_id} "
                f"lead_time={base_lead}d -> {disrupted_lead}d"
            )
            return disrupted_lead

        return base_lead

    def __repr__(self) -> str:
        return (
            f"SupplyChain(agents={self.n_agents}, "
            f"pending_orders={len(self.pending_orders)}, "
            f"day={self.current_day})"
        )