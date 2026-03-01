import numpy as np
from typing import Dict, List
from dataclasses import dataclass, field

from environment.product_catalog import Product, ProductCatalog
from utils.config_loader import load_env_config
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DemandResult:
    """Output of one demand calculation step."""
    units_sold:    Dict[int, Dict[int, int]]  # agent_id -> product_id -> units
    revenue:       Dict[int, float]            # agent_id -> total revenue
    market_shares: Dict[int, float]            # agent_id -> share of total units
    total_demand:  int                         # total units sold across all agents
    store_visits:  Dict[int, int]              # agent_id -> number of customer visits


class MNLDemandModel:
    """
    Basket-level Multinomial Logit (MNL) consumer choice model.

    KEY DESIGN DECISION — Basket-level shopping:
        Each customer picks ONE store for their entire basket (~10 items).
        They compare the total basket cost across all stores and choose
        probabilistically — cheaper overall basket = higher probability,
        but brand loyalty and quality bias also factor in.

        This mirrors real grocery behavior: ~70-80% of shoppers do
        one-stop grocery trips (Nielsen 2019). Split-shopping is the
        exception, not the rule.

    Calibrated from Instacart EDA:
        - brand_loyalty:        0.59
        - mean_basket_size:     10 items
        - weekend_multiplier:   1.347
    """

    # retailer name -> quality/brand attractiveness bonus
    QUALITY_BIAS = {
        "whole_foods":  0.12,
        "trader_joes":  0.10,
        "walmart":      0.08,
        "costco":       0.07,
        "target":       0.06,
        "kroger":       0.05,
        "safeway":      0.05,
        "qfc":          0.04,
        "amazon_fresh": 0.05,
        "aldi":         0.03,
    }

    def __init__(self, catalog: ProductCatalog, config=None, seed: int = 42):
        self.catalog = catalog
        self.config  = config or load_env_config()
        self.rng     = np.random.default_rng(seed)

        # demand parameters
        self.base_daily_customers = self.config.get('demand.base_daily_customers', 10000)
        self.price_sensitivity    = self.config.get('demand.price_sensitivity', 2.5)
        self.brand_loyalty        = self.config.get('demand.brand_loyalty', 0.59)
        self.noise_std            = self.config.get('demand.noise_std', 0.05)
        self.mean_basket_size     = self.config.get('demand.mean_basket_size', 10)

        # product list for basket sampling
        self.all_products = catalog.get_all_products()
        self.n_products   = len(self.all_products)

        logger.info(
            f"MNLDemandModel (basket-level) initialized | "
            f"customers={self.base_daily_customers} | "
            f"price_sensitivity={self.price_sensitivity} | "
            f"brand_loyalty={self.brand_loyalty} | "
            f"mean_basket_size={self.mean_basket_size}"
        )

    # ── Public API ────────────────────────────────────────────────

    def compute_demand(
        self,
        prices:           Dict[int, Dict[int, float]],  # agent_id -> product_id -> price
        agent_names:      Dict[int, str],               # agent_id -> retailer name
        day:              int,                          # 0-364
        is_weekend:       bool  = False,
        is_holiday:       bool  = False,
        holiday_multiplier: float = 1.0,
    ) -> DemandResult:
        """
        Simulate a full day of shopping.

        Each customer:
          1. Samples a basket of ~10 products they want to buy today
          2. Computes the total basket cost at each store
          3. Picks ONE store using MNL over basket costs
          4. Buys their entire basket at that store

        Args:
            prices:             current prices for all agents and products
            agent_names:        maps agent_id -> retailer name
            day:                day of year for seasonality
            is_weekend:         applies 1.347x multiplier
            is_holiday:         applies holiday_multiplier
            holiday_multiplier: demand boost on holidays

        Returns:
            DemandResult with per-agent units, revenue, market shares, visits
        """
        agent_ids      = list(prices.keys())
        daily_customers = self._get_daily_customers(
            is_weekend, is_holiday, holiday_multiplier
        )

        # initialize accumulators
        units_sold   = {aid: {p.product_id: 0 for p in self.all_products}
                        for aid in agent_ids}
        revenue      = {aid: 0.0 for aid in agent_ids}
        store_visits = {aid: 0    for aid in agent_ids}

        # simulate each customer independently
        for _ in range(daily_customers):
            # step 1 — sample this customer's basket (which products they want)
            basket = self._sample_basket(day)

            if not basket:
                continue

            # step 2 — compute total basket cost at each store
            basket_costs = self._compute_basket_costs(basket, prices, agent_ids)

            # step 3 — pick one store using MNL over basket costs
            chosen_agent = self._choose_store(basket_costs, agent_names, agent_ids)

            # step 4 — buy the whole basket at the chosen store
            store_visits[chosen_agent] += 1
            for product in basket:
                pid   = product.product_id
                price = prices[chosen_agent].get(pid, product.base_retail_price)
                units_sold[chosen_agent][pid] += 1
                revenue[chosen_agent]         += price

        # compute market shares by revenue
        total_revenue = sum(revenue.values())
        market_shares = {
            aid: revenue[aid] / total_revenue if total_revenue > 0 else 0.0
            for aid in agent_ids
        }

        total_units = sum(
            sum(pid_units.values())
            for pid_units in units_sold.values()
        )

        return DemandResult(
            units_sold=units_sold,
            revenue=revenue,
            market_shares=market_shares,
            total_demand=total_units,
            store_visits=store_visits,
        )

    # ── Private helpers ───────────────────────────────────────────

    def _sample_basket(self, day: int) -> List[Product]:
        """
        Sample which products a customer wants to buy today.

        Basket size is Poisson-distributed around mean_basket_size.
        Each product's inclusion probability is weighted by its
        seasonality on this day — produce in summer, frozen in winter.
        """
        basket_size = max(1, self.rng.poisson(self.mean_basket_size))

        # weight each product by its seasonality today
        weights = np.array([
            p.seasonality[day % 365] for p in self.all_products
        ])
        weights = weights / weights.sum()   # normalize to probabilities

        # sample without replacement (customer buys each product at most once)
        n_sample = min(basket_size, self.n_products)
        # use faster replacement sampling for large customer counts
        chosen_ix = self.rng.choice(self.n_products, size=n_sample,
                                    replace=True, p=weights)
        seen = set()
        unique_ix = []
        for ix in chosen_ix:
            if ix not in seen:
                seen.add(ix)
                unique_ix.append(ix)
        return [self.all_products[i] for i in unique_ix]

    def _compute_basket_costs(
        self,
        basket:    List[Product],
        prices:    Dict[int, Dict[int, float]],
        agent_ids: List[int],
    ) -> Dict[int, float]:
        """
        Compute total basket cost at each store.
        If a store doesn't have a price for a product, use base retail price.
        """
        return {
            aid: sum(
                prices[aid].get(p.product_id, p.base_retail_price)
                for p in basket
            )
            for aid in agent_ids
        }

    def _choose_store(
        self,
        basket_costs: Dict[int, float],
        agent_names:  Dict[int, str],
        agent_ids:    List[int],
    ) -> int:
        """
        MNL store choice based on total basket cost.

        Utility for store i:
            U_i = -α * basket_cost_i + quality_bias_i + loyalty_i + ε_i

        Where:
            α             = price_sensitivity (2.5)
            quality_bias  = store-level attractiveness (e.g. Whole Foods = 0.12)
            loyalty_i     = random brand loyalty draw
            ε_i           = Gumbel noise (standard in MNL)

        Brand loyalty: with probability brand_loyalty (0.59), the customer
        has a preferred store and gets a large utility bonus for it,
        partially overriding price. This is the key calibrated insight
        from Instacart — 59% of purchases are loyalty-driven.
        """
        # determine if this customer is a loyal shopper
        is_loyal       = self.rng.random() < self.brand_loyalty
        loyal_agent_id = self.rng.choice(agent_ids) if is_loyal else None

        utilities = {}
        for aid in agent_ids:
            name         = agent_names.get(aid, "")
            quality      = self.QUALITY_BIAS.get(name, 0.0)
            cost         = basket_costs[aid]

            # price utility — log scale so doubling price isn't catastrophic
            price_util   = -self.price_sensitivity * np.log(max(cost, 0.01))

            # loyalty bonus — loyal customers strongly prefer their store
            loyalty_util = 3.0 if (is_loyal and aid == loyal_agent_id) else 0.0

            # gumbel noise — standard MNL random utility term
            gumbel       = self.rng.gumbel(0, 1)

            utilities[aid] = price_util + quality + loyalty_util + gumbel

        # softmax → choice probabilities → sample one store
        max_u  = max(utilities.values())
        exp_u  = {aid: np.exp(utilities[aid] - max_u) for aid in agent_ids}
        sum_eu = sum(exp_u.values())
        probs  = {aid: exp_u[aid] / sum_eu for aid in agent_ids}

        # sample from the probability distribution
        chosen = self.rng.choice(
            agent_ids,
            p=[probs[aid] for aid in agent_ids]
        )
        return chosen

    def _get_daily_customers(
        self,
        is_weekend:         bool,
        is_holiday:         bool,
        holiday_multiplier: float,
    ) -> int:
        """Apply weekend/holiday multipliers to base customer count."""
        customers = self.base_daily_customers
        if is_weekend:
            customers = int(customers * 1.347)
        if is_holiday:
            customers = int(customers * holiday_multiplier)
        noise = self.rng.normal(1.0, 0.03)
        return max(1, int(customers * noise))