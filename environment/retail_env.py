import numpy as np
from typing import Dict, Optional
from gymnasium import spaces
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import AgentSelector

from environment.product_catalog import ProductCatalog
from environment.demand_model import MNLDemandModel
from environment.inventory import InventoryManager
from environment.supply_chain import SupplyChain
from environment.promotions import PromotionCalendar
from utils.config_loader import load_env_config, load_agent_config
from utils.logger import get_logger

logger = get_logger(__name__)

# price adjustment actions
PRICE_ACTIONS = {
    0: 0.85,   # -15%
    1: 0.90,   # -10%
    2: 0.95,   #  -5%
    3: 1.00,   #   0% (hold)
    4: 1.05,   #  +5%
    5: 1.10,   # +10%
    6: 1.15,   # +15%
}
N_ACTIONS = len(PRICE_ACTIONS)


class RetailEnv(AECEnv):
    """
    Multi-agent retail pricing environment.

    10 heterogeneous agents (retailers) compete to sell 50 products
    over a 365-day episode. Each day, agents set prices, consumers
    choose where to shop (basket-level MNL), inventory depletes,
    supply chain delivers restocks, and rewards are calculated.

    Observation space (per agent): Box(266,)
        - own prices:              (50,)  normalized by base retail price
        - competitor avg prices:   (50,)  normalized
        - stock levels:            (50,)  normalized by initial stock
        - stockout flags:          (50,)  binary
        - pipeline (incoming):     (50,)  normalized
        - promo observation:        (5,)  [weekend, holiday, mult, days_to_next, next_mult]
        - day of year:              (1,)  normalized to [0,1]
        - agent id one-hot:        (10,)
        Total: 266

    Action space (per agent): MultiDiscrete([7] * 50)
        Each product gets one of 7 price adjustments:
        -15%, -10%, -5%, 0%, +5%, +10%, +15%

    Reward functions (configurable per agent):
        - pure_revenue
        - profit_margin
        - market_share
        - revenue_with_inventory
        - long_term_value
        - promo_aware_profit
        - premium_floor
        - prestige_reward
        - discount_maximization
        - bulk_volume
    """

    metadata = {"render_modes": ["human"], "name": "retail_env_v1"}

    AGENT_NAMES = [
        "walmart", "target", "amazon_fresh", "qfc", "safeway",
        "kroger", "trader_joes", "whole_foods", "aldi", "costco"
    ]

    def __init__(self, config_path=None, render_mode=None):
        super().__init__()

        self.env_config   = load_env_config(config_path)
        self.agent_config = load_agent_config(config_path)
        self.render_mode  = render_mode

        # core components
        self.catalog  = ProductCatalog(seed=self.env_config.get('environment.seed', 42))
        self.demand   = MNLDemandModel(self.catalog,  self.env_config)
        self.inv      = InventoryManager(self.catalog, len(self.AGENT_NAMES), self.env_config)
        self.sc       = SupplyChain(len(self.AGENT_NAMES), self.env_config)
        self.promo_cal = PromotionCalendar(self.env_config)

        # episode config
        self.episode_length = self.env_config.get('environment.episode_length', 365)
        self.n_products     = len(self.catalog)
        self.n_agents_count = len(self.AGENT_NAMES)

        # PettingZoo required attributes
        self.agents         = list(self.AGENT_NAMES)
        self.possible_agents = list(self.AGENT_NAMES)
        self._agent_ids     = {name: i for i, name in enumerate(self.AGENT_NAMES)}

        # spaces
        obs_size = (
            self.n_products * 5 +   # prices, comp prices, stock, stockout, pipeline
            5 +                      # promo vector
            1 +                      # day normalized
            self.n_agents_count      # agent id one-hot
        )
        self.observation_spaces = {
            agent: spaces.Box(
                low=0.0, high=10.0,
                shape=(obs_size,),
                dtype=np.float32
            )
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.MultiDiscrete([N_ACTIONS] * self.n_products)
            for agent in self.agents
        }

        # state — initialized in reset()
        self.current_day   = 0
        self.current_prices: Dict[int, Dict[int, float]] = {}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}
        self._agent_selector = AgentSelector(self.agents)

        # episode metrics for analysis
        self.episode_metrics = []

        logger.info(
            f"RetailEnv initialized | "
            f"agents={self.n_agents_count} | "
            f"products={self.n_products} | "
            f"obs_size={obs_size} | "
            f"episode_length={self.episode_length}"
        )

    # ── PettingZoo API ────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        """Reset environment to start of new episode."""
        if seed is not None:
            np.random.seed(seed)

        self.agents    = list(self.possible_agents)
        self.current_day = 0

        self.inv.reset()
        self.sc.reset()

        # initialize prices at base retail for all agents
        self.current_prices = {
            aid: {
                p.product_id: p.base_retail_price
                for p in self.catalog.get_all_products()
            }
            for aid in range(self.n_agents_count)
        }

        self.rewards      = {a: 0.0 for a in self.agents}
        self.terminations = {a: False for a in self.agents}
        self.truncations  = {a: False for a in self.agents}
        self.infos        = {a: {} for a in self.agents}
        self._cumulative_rewards = {a: 0.0 for a in self.agents}

        self._agent_selector = AgentSelector(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.episode_metrics = []

        observations = {
            agent: self._get_observation(agent)
            for agent in self.agents
        }

        return observations, self.infos

    def step(self, action):
        """
        Process one agent's action.

        In AEC (Agent-Environment-Cycle) format, agents act one at a time.
        After all agents have acted, we simulate one day and compute rewards.
        """
        if (self.terminations[self.agent_selection] or
                self.truncations[self.agent_selection]):
            self._was_dead_step(action)
            return

        agent     = self.agent_selection
        agent_id  = self._agent_ids[agent]

        # apply price actions
        self._apply_action(agent_id, action)

        # if all agents have acted, simulate the day
        if self._agent_selector.is_last():
            self._simulate_day()

        self._cumulative_rewards[agent] += self.rewards.get(agent, 0.0)
        self.agent_selection = self._agent_selector.next()

    def observe(self, agent: str) -> np.ndarray:
        return self._get_observation(agent)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def render(self):
        if self.render_mode == "human":
            self._render_human()

    def close(self):
        pass

    # ── Core simulation ───────────────────────────────────────────

    def _simulate_day(self):
        """
        Simulate one full day after all agents have set prices:
        1. Get promo info for today
        2. Compute demand (basket-level MNL)
        3. Process supply chain deliveries
        4. Update inventory
        5. Place reorders
        6. Compute rewards
        7. Log metrics
        8. Advance day
        """
        day_info = self.promo_cal.get_day_info(self.current_day)

        # step 1 — compute demand
        demand_result = self.demand.compute_demand(
            prices=self.current_prices,
            agent_names={i: name for i, name in enumerate(self.AGENT_NAMES)},
            day=self.current_day,
            is_weekend=day_info.is_weekend,
            is_holiday=day_info.is_holiday,
            holiday_multiplier=day_info.demand_multiplier,
        )

        # step 2 — supply chain delivers today's orders
        incoming = self.sc.step()

        # step 3 — update inventory
        inv_states = self.inv.step(demand_result.units_sold, incoming)

        # step 4 — place reorders for low stock
        reorders = self.inv.place_reorders()
        self.sc.place_orders(reorders)

        # step 5 — compute rewards per agent
        for agent in self.agents:
            aid = self._agent_ids[agent]
            self.rewards[agent] = self._compute_reward(
                agent, aid, demand_result, inv_states[aid]
            )

        # step 6 — log daily metrics
        self.episode_metrics.append({
            "day":           self.current_day,
            "total_demand":  demand_result.total_demand,
            "market_shares": dict(demand_result.market_shares),
            "revenues":      dict(demand_result.revenue),
        })

        # step 7 — check episode termination
        self.current_day += 1
        if self.current_day >= self.episode_length:
            for agent in self.agents:
                self.terminations[agent] = True

    def _apply_action(self, agent_id: int, action: np.ndarray):
        """Convert discrete actions to price multipliers and update prices."""
        for i, product in enumerate(self.catalog.get_all_products()):
            pid         = product.product_id
            action_idx  = int(action[i])
            multiplier  = PRICE_ACTIONS[action_idx]
            current     = self.current_prices[agent_id][pid]
            new_price   = round(current * multiplier, 2)

            # enforce price floor — never sell below cost
            min_price = product.base_cost * 1.01
            self.current_prices[agent_id][pid] = max(new_price, min_price)

    # ── Reward functions ──────────────────────────────────────────

    def _compute_reward(self, agent, agent_id, demand_result, inv_state):
        """Route to the correct reward function for this agent."""
        reward_fn = self.agent_config.get(f"agents.{agent}.reward", "pure_revenue")
        revenue   = demand_result.revenue[agent_id]
        holding   = inv_state.holding_cost
        stockout  = inv_state.stockout_penalty

        if reward_fn == "pure_revenue":
            return revenue

        elif reward_fn == "profit_margin":
            # compute actual profit (revenue - cost of goods sold)
            profit = self._compute_profit(agent_id, demand_result)
            return profit - holding

        elif reward_fn == "market_share":
            # reward market share directly, penalize for being last
            share = demand_result.market_shares[agent_id]
            return share * 1000   # scale to similar magnitude as revenue

        elif reward_fn == "revenue_with_inventory":
            return revenue - holding - stockout

        elif reward_fn == "long_term_value":
            # LTV: revenue + loyalty bonus for repeat customers
            visits = demand_result.store_visits[agent_id]
            return revenue + (visits * 2.0) - holding

        elif reward_fn == "promo_aware_profit":
            profit  = self._compute_profit(agent_id, demand_result)
            day_info = self.promo_cal.get_day_info(self.current_day)
            promo_bonus = profit * 0.2 if day_info.is_holiday else 0.0
            return profit + promo_bonus - holding

        elif reward_fn == "premium_floor":
            # penalize heavily for pricing below 85% of base retail
            profit = self._compute_profit(agent_id, demand_result)
            penalty = self._premium_floor_penalty(agent_id)
            return profit - penalty - holding

        elif reward_fn == "prestige_reward":
            # prestige score drops if priced too low vs market average
            profit  = self._compute_profit(agent_id, demand_result)
            prestige = self._prestige_score(agent_id)
            return profit + prestige - holding

        elif reward_fn == "discount_maximization":
            return revenue - holding * 0.5   # lower weight on holding cost

        elif reward_fn == "bulk_volume":
            # reward total units sold, not revenue
            units = sum(demand_result.units_sold[agent_id].values())
            return units * 5.0 - holding

        return revenue   # fallback

    def _compute_profit(self, agent_id: int, demand_result) -> float:
        """Revenue minus cost of goods sold."""
        profit = 0.0
        for product in self.catalog.get_all_products():
            pid   = product.product_id
            units = demand_result.units_sold[agent_id].get(pid, 0)
            price = self.current_prices[agent_id][pid]
            profit += units * (price - product.base_cost)
        return profit

    def _premium_floor_penalty(self, agent_id: int) -> float:
        """Penalty for pricing below the premium floor."""
        penalty    = 0.0
        floor_pct  = self.agent_config.get(
            f"agents.{self.AGENT_NAMES[agent_id]}.price_floor_pct", 0.85
        )
        for product in self.catalog.get_all_products():
            pid       = product.product_id
            price     = self.current_prices[agent_id][pid]
            floor     = product.base_retail_price * floor_pct
            if price < floor:
                penalty += (floor - price) * 10
        return penalty

    def _prestige_score(self, agent_id: int) -> float:
        """Bonus when priced above market average, penalty when below."""
        score = 0.0
        for product in self.catalog.get_all_products():
            pid        = product.product_id
            own_price  = self.current_prices[agent_id][pid]
            avg_price  = np.mean([
                self.current_prices[a][pid]
                for a in range(self.n_agents_count)
            ])
            if own_price >= avg_price:
                score += 1.0
            else:
                score -= 2.0
        return score

    # ── Observation builder ───────────────────────────────────────

    def _get_observation(self, agent: str) -> np.ndarray:
        """Build the 266-dimensional observation vector for an agent."""
        aid = self._agent_ids[agent]

        # own prices normalized
        own_prices = np.array([
            self.current_prices[aid][p.product_id] / p.base_retail_price
            for p in self.catalog.get_all_products()
        ], dtype=np.float32)

        # competitor average prices normalized
        other_ids = [i for i in range(self.n_agents_count) if i != aid]
        comp_prices = np.array([
            np.mean([self.current_prices[o][p.product_id] for o in other_ids])
            / p.base_retail_price
            for p in self.catalog.get_all_products()
        ], dtype=np.float32)

        # inventory
        stock_vec   = self.inv.get_stock_vector(aid)
        stockout_vec = self.inv.get_stockout_vector(aid)
        pipeline_vec = self.sc.get_pipeline_vector(aid, list(range(self.n_products)))

        # promo
        promo_vec = self.promo_cal.get_observation_vector(self.current_day)

        # time
        day_norm = np.array([self.current_day / self.episode_length], dtype=np.float32)

        # agent identity one-hot
        agent_onehot = np.zeros(self.n_agents_count, dtype=np.float32)
        agent_onehot[aid] = 1.0

        return np.concatenate([
            own_prices, comp_prices, stock_vec, stockout_vec,
            pipeline_vec, promo_vec, day_norm, agent_onehot
        ])

    def _render_human(self):
        """Simple text render of current state."""
        print(f"\n{'='*60}")
        print(f"Day {self.current_day} / {self.episode_length}")
        print(f"{'Agent':<15} {'Avg Price':>10} {'Stock':>10}")
        print(f"{'-'*40}")
        for agent in self.AGENT_NAMES:
            aid       = self._agent_ids[agent]
            avg_price = np.mean(list(self.current_prices[aid].values()))
            stock     = self.inv.get_state_summary(aid)['total_units']
            print(f"{agent:<15} {avg_price:>10.2f} {stock:>10,}")