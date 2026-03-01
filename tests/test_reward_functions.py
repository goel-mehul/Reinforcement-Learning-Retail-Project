import numpy as np
import pytest
from unittest.mock import MagicMock
from agents.baselines.reward_functions import (
    pure_revenue, profit_margin, market_share, revenue_with_inventory,
    long_term_value, promo_aware_profit, premium_floor, prestige_reward,
    discount_maximization, bulk_volume, compute_reward,
    REWARD_FUNCTIONS, RewardInput
)
from environment.product_catalog import ProductCatalog


@pytest.fixture
def catalog():
    return ProductCatalog(seed=42)


@pytest.fixture
def base_input(catalog):
    """A realistic RewardInput with moderate sales."""
    demand = MagicMock()
    demand.revenue      = {0: 10000.0}
    demand.market_shares = {0: 0.12}
    demand.store_visits  = {0: 150}
    demand.units_sold    = {0: {pid: 10 for pid in range(50)}}

    inv = MagicMock()
    inv.holding_cost     = 50.0
    inv.stockout_penalty = 0.0

    day_info = MagicMock()
    day_info.is_holiday  = False

    prices = {
        p.product_id: p.base_retail_price
        for p in catalog.get_all_products()
    }

    return RewardInput(
        agent_id=0,
        agent_name="walmart",
        demand_result=demand,
        inv_state=inv,
        day_info=day_info,
        current_prices=prices,
        catalog=catalog,
    )


@pytest.fixture
def holiday_input(base_input):
    base_input.day_info.is_holiday = True
    return base_input


@pytest.fixture
def stockout_input(base_input):
    base_input.inv_state.stockout_penalty = 500.0
    return base_input


class TestRewardFunctionRegistry:

    def test_all_ten_registered(self):
        assert len(REWARD_FUNCTIONS) == 10

    def test_compute_reward_dispatches(self, base_input):
        result = compute_reward("pure_revenue", base_input)
        assert result == pure_revenue(base_input)

    def test_compute_reward_fallback(self, base_input):
        result = compute_reward("nonexistent_fn", base_input)
        assert result == pure_revenue(base_input)


class TestPureRevenue:

    def test_equals_revenue(self, base_input):
        assert pure_revenue(base_input) == 10000.0

    def test_ignores_holding_cost(self, base_input):
        base_input.inv_state.holding_cost = 9999.0
        assert pure_revenue(base_input) == 10000.0


class TestProfitMargin:

    def test_less_than_revenue(self, base_input):
        result = profit_margin(base_input)
        assert result < 10000.0

    def test_subtracts_holding_cost(self, base_input):
        r1 = profit_margin(base_input)
        base_input.inv_state.holding_cost = 1000.0
        r2 = profit_margin(base_input)
        assert r1 > r2


class TestMarketShare:

    def test_scaled_correctly(self, base_input):
        result = market_share(base_input)
        assert result == pytest.approx(0.12 * 10000, abs=1)

    def test_higher_share_higher_reward(self, base_input, catalog):
        inp2 = RewardInput(
            agent_id=0, agent_name="walmart",
            demand_result=MagicMock(
                revenue={0: 10000.0},
                market_shares={0: 0.25},
                store_visits={0: 150},
                units_sold={0: {pid: 10 for pid in range(50)}}
            ),
            inv_state=base_input.inv_state,
            day_info=base_input.day_info,
            current_prices=base_input.current_prices,
            catalog=catalog,
        )
        assert market_share(inp2) > market_share(base_input)


class TestRevenueWithInventory:

    def test_penalizes_stockout(self, catalog):
        demand = MagicMock()
        demand.revenue = {0: 10000.0}
        demand.market_shares = {0: 0.12}
        demand.store_visits = {0: 150}
        demand.units_sold = {0: {pid: 10 for pid in range(50)}}
        prices = {p.product_id: p.base_retail_price for p in catalog.get_all_products()}

        inv_normal = MagicMock()
        inv_normal.holding_cost = 50.0
        inv_normal.stockout_penalty = 0.0

        inv_stockout = MagicMock()
        inv_stockout.holding_cost = 50.0
        inv_stockout.stockout_penalty = 500.0

        day = MagicMock()
        day.is_holiday = False

        inp_normal  = RewardInput(0, "walmart", demand, inv_normal,  day, prices, catalog)
        inp_stockout = RewardInput(0, "walmart", demand, inv_stockout, day, prices, catalog)

        assert revenue_with_inventory(inp_normal) > revenue_with_inventory(inp_stockout)

    def test_penalizes_holding(self, base_input):
        r1 = revenue_with_inventory(base_input)
        base_input.inv_state.holding_cost = 500.0
        r2 = revenue_with_inventory(base_input)
        assert r1 > r2


class TestLongTermValue:

    def test_visit_bonus_applied(self, base_input, catalog):
        inp_low = RewardInput(
            agent_id=0, agent_name="walmart",
            demand_result=MagicMock(
                revenue={0: 10000.0},
                market_shares={0: 0.12},
                store_visits={0: 10},
                units_sold={0: {pid: 10 for pid in range(50)}}
            ),
            inv_state=base_input.inv_state,
            day_info=base_input.day_info,
            current_prices=base_input.current_prices,
            catalog=catalog,
        )
        assert long_term_value(base_input) > long_term_value(inp_low)


class TestPromoAwareProfit:

    def test_holiday_bonus_applied(self, catalog):
        demand = MagicMock()
        demand.revenue = {0: 10000.0}
        demand.market_shares = {0: 0.12}
        demand.store_visits = {0: 150}
        demand.units_sold = {0: {pid: 10 for pid in range(50)}}
        prices = {p.product_id: p.base_retail_price for p in catalog.get_all_products()}
        inv = MagicMock()
        inv.holding_cost = 50.0
        inv.stockout_penalty = 0.0

        day_normal  = MagicMock()
        day_normal.is_holiday = False
        day_holiday = MagicMock()
        day_holiday.is_holiday = True

        inp_normal  = RewardInput(0, "walmart", demand, inv, day_normal,  prices, catalog)
        inp_holiday = RewardInput(0, "walmart", demand, inv, day_holiday, prices, catalog)

        assert promo_aware_profit(inp_holiday) > promo_aware_profit(inp_normal)


class TestPremiumFloor:

    def test_penalty_for_low_prices(self, base_input, catalog):
        # price everything at 50% of retail — way below floor
        low_prices = {
            p.product_id: p.base_retail_price * 0.50
            for p in catalog.get_all_products()
        }
        inp_low = RewardInput(
            agent_id=0, agent_name="trader_joes",
            demand_result=base_input.demand_result,
            inv_state=base_input.inv_state,
            day_info=base_input.day_info,
            current_prices=low_prices,
            catalog=catalog,
        )
        assert premium_floor(inp_low) < premium_floor(base_input)


class TestBulkVolume:

    def test_rewards_units_not_revenue(self, base_input, catalog):
        # high units, low revenue
        inp_bulk = RewardInput(
            agent_id=0, agent_name="costco",
            demand_result=MagicMock(
                revenue={0: 100.0},
                market_shares={0: 0.05},
                store_visits={0: 50},
                units_sold={0: {pid: 100 for pid in range(50)}}
            ),
            inv_state=base_input.inv_state,
            day_info=base_input.day_info,
            current_prices=base_input.current_prices,
            catalog=catalog,
        )
        assert bulk_volume(inp_bulk) > pure_revenue(inp_bulk)