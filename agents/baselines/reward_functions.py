import numpy as np
from dataclasses import dataclass
from typing import Dict

from environment.inventory import InventoryState
from environment.demand_model import DemandResult
from environment.product_catalog import ProductCatalog
from environment.promotions import DayInfo


@dataclass
class RewardInput:
    """Everything needed to compute any reward function."""
    agent_id:      int
    agent_name:    str
    demand_result: DemandResult
    inv_state:     InventoryState
    day_info:      DayInfo
    current_prices: Dict[int, float]   # product_id -> price
    catalog:       ProductCatalog


def pure_revenue(inp: RewardInput) -> float:
    """
    Raw revenue — total sales regardless of cost.
    Simple, aggressive. Agents maximize topline.
    Used by: Walmart
    """
    return inp.demand_result.revenue[inp.agent_id]


def profit_margin(inp: RewardInput) -> float:
    """
    Gross profit (revenue - COGS) minus holding costs.
    Agents care about margin, not just volume.
    Used by: Target
    """
    profit  = _compute_profit(inp)
    holding = inp.inv_state.holding_cost
    return profit - holding


def market_share(inp: RewardInput) -> float:
    """
    Reward proportional to market share.
    Agents will sacrifice margin to capture customers.
    Scaled to similar magnitude as revenue rewards.
    Used by: Amazon Fresh
    """
    share = inp.demand_result.market_shares[inp.agent_id]
    return share * 10000


def revenue_with_inventory(inp: RewardInput) -> float:
    """
    Revenue minus holding costs and stockout penalties.
    Must balance aggressive pricing with inventory risk.
    Used by: QFC
    """
    revenue  = inp.demand_result.revenue[inp.agent_id]
    holding  = inp.inv_state.holding_cost
    stockout = inp.inv_state.stockout_penalty
    return revenue - holding - stockout


def long_term_value(inp: RewardInput) -> float:
    """
    Revenue + bonus for store visits (customer retention proxy).
    Encourages building loyal customer base over raw revenue.
    Used by: Safeway
    """
    revenue = inp.demand_result.revenue[inp.agent_id]
    visits  = inp.demand_result.store_visits[inp.agent_id]
    holding = inp.inv_state.holding_cost
    return revenue + (visits * 2.0) - holding


def promo_aware_profit(inp: RewardInput) -> float:
    """
    Profit with a bonus during holiday/promo periods.
    Agents learn to prepare inventory and price aggressively on holidays.
    Used by: Kroger
    """
    profit     = _compute_profit(inp)
    holding    = inp.inv_state.holding_cost
    promo_bonus = profit * 0.2 if inp.day_info.is_holiday else 0.0
    return profit + promo_bonus - holding


def premium_floor(inp: RewardInput) -> float:
    """
    Profit with heavy penalty for pricing below 85% of base retail.
    Agents maintain premium positioning — never race to the bottom.
    Used by: Trader Joe's
    """
    profit  = _compute_profit(inp)
    holding = inp.inv_state.holding_cost
    penalty = _premium_floor_penalty(inp, floor_pct=0.85)
    return profit - penalty - holding


def prestige_reward(inp: RewardInput) -> float:
    """
    Profit plus prestige score.
    Prestige drops if priced below market average.
    Whole Foods can't be seen as cheap.
    Used by: Whole Foods
    """
    profit   = _compute_profit(inp)
    holding  = inp.inv_state.holding_cost
    prestige = _prestige_score(inp)
    return profit + prestige - holding


def discount_maximization(inp: RewardInput) -> float:
    """
    Revenue with reduced holding cost weight.
    Aldi-style: volume over margin, lean inventory.
    Used by: Aldi
    """
    revenue = inp.demand_result.revenue[inp.agent_id]
    holding = inp.inv_state.holding_cost
    return revenue - holding * 0.3


def bulk_volume(inp: RewardInput) -> float:
    """
    Reward total units sold, not revenue.
    Costco-style: move as much product as possible.
    Used by: Costco
    """
    units   = sum(inp.demand_result.units_sold[inp.agent_id].values())
    holding = inp.inv_state.holding_cost
    return units * 5.0 - holding


# ── Reward registry ───────────────────────────────────────────────

REWARD_FUNCTIONS = {
    "pure_revenue":          pure_revenue,
    "profit_margin":         profit_margin,
    "market_share":          market_share,
    "revenue_with_inventory": revenue_with_inventory,
    "long_term_value":       long_term_value,
    "promo_aware_profit":    promo_aware_profit,
    "premium_floor":         premium_floor,
    "prestige_reward":       prestige_reward,
    "discount_maximization": discount_maximization,
    "bulk_volume":           bulk_volume,
}


def compute_reward(reward_fn_name: str, inp: RewardInput) -> float:
    """
    Dispatch to the correct reward function by name.
    Falls back to pure_revenue if name not found.
    """
    fn = REWARD_FUNCTIONS.get(reward_fn_name, pure_revenue)
    return fn(inp)


# ── Private helpers ───────────────────────────────────────────────

def _compute_profit(inp: RewardInput) -> float:
    profit = 0.0
    for product in inp.catalog.get_all_products():
        pid   = product.product_id
        units = inp.demand_result.units_sold[inp.agent_id].get(pid, 0)
        price = inp.current_prices.get(pid, product.base_retail_price)
        profit += units * (price - product.base_cost)
    return profit


def _premium_floor_penalty(inp: RewardInput, floor_pct: float) -> float:
    penalty = 0.0
    for product in inp.catalog.get_all_products():
        pid   = product.product_id
        price = inp.current_prices.get(pid, product.base_retail_price)
        floor = product.base_retail_price * floor_pct
        if price < floor:
            penalty += (floor - price) * 10
    return penalty


def _prestige_score(inp: RewardInput) -> float:
    """Bonus for pricing above market average, penalty for below."""
    score = 0.0
    for product in inp.catalog.get_all_products():
        pid       = product.product_id
        own_price = inp.current_prices.get(pid, product.base_retail_price)
        market_avg = product.base_retail_price   # simplified — env has full avg
        if own_price >= market_avg:
            score += 1.0
        else:
            score -= 2.0
    return score