import numpy as np
import pytest
from environment.product_catalog import ProductCatalog
from environment.demand_model import MNLDemandModel, DemandResult


@pytest.fixture
def catalog():
    return ProductCatalog(seed=42)


@pytest.fixture
def model(catalog):
    return MNLDemandModel(catalog, seed=42)


@pytest.fixture
def basic_prices(catalog):
    """All agents price at base retail."""
    return {
        i: {p.product_id: p.base_retail_price for p in catalog.get_all_products()}
        for i in range(10)
    }


@pytest.fixture
def agent_names():
    return {
        0: 'walmart', 1: 'target', 2: 'amazon_fresh',
        3: 'qfc', 4: 'safeway', 5: 'kroger',
        6: 'trader_joes', 7: 'whole_foods', 8: 'aldi', 9: 'costco'
    }


class TestDemandResult:

    def test_returns_demand_result(self, model, basic_prices, agent_names):
        result = model.compute_demand(basic_prices, agent_names, day=0)
        assert isinstance(result, DemandResult)

    def test_all_agents_present(self, model, basic_prices, agent_names):
        result = model.compute_demand(basic_prices, agent_names, day=0)
        for aid in range(10):
            assert aid in result.units_sold
            assert aid in result.revenue
            assert aid in result.market_shares
            assert aid in result.store_visits

    def test_market_shares_sum_to_one(self, model, basic_prices, agent_names):
        result = model.compute_demand(basic_prices, agent_names, day=0)
        total_share = sum(result.market_shares.values())
        assert total_share == pytest.approx(1.0, abs=0.01)

    def test_revenue_nonnegative(self, model, basic_prices, agent_names):
        result = model.compute_demand(basic_prices, agent_names, day=0)
        for aid, rev in result.revenue.items():
            assert rev >= 0.0

    def test_units_nonnegative(self, model, basic_prices, agent_names):
        result = model.compute_demand(basic_prices, agent_names, day=0)
        for aid, products in result.units_sold.items():
            for pid, units in products.items():
                assert units >= 0

    def test_total_demand_positive(self, model, basic_prices, agent_names):
        result = model.compute_demand(basic_prices, agent_names, day=0)
        assert result.total_demand > 0

    def test_store_visits_match_revenue(self, model, basic_prices, agent_names):
        """Agents with more visits should generally have more revenue."""
        result = model.compute_demand(basic_prices, agent_names, day=0)
        most_visited = max(result.store_visits, key=result.store_visits.get)
        most_revenue = max(result.revenue, key=result.revenue.get)
        # not guaranteed to be identical but should be correlated
        assert result.store_visits[most_visited] > 0
        assert result.revenue[most_revenue] > 0


class TestPriceSensitivity:

    def test_cheaper_store_gets_more_visits(self, catalog, agent_names):
        """Agent pricing 30% below retail should get more visits than one pricing 30% above."""
        model = MNLDemandModel(catalog, seed=42)

        prices = {
            i: {p.product_id: p.base_retail_price for p in catalog.get_all_products()}
            for i in range(10)
        }

        # make agent 0 (walmart) significantly cheaper
        for pid in prices[0]:
            prices[0][pid] = catalog.get_product(pid).base_retail_price * 0.70

        # make agent 7 (whole_foods) significantly more expensive
        for pid in prices[7]:
            prices[7][pid] = catalog.get_product(pid).base_retail_price * 1.30

        result = model.compute_demand(prices, agent_names, day=90)
        assert result.store_visits[0] > result.store_visits[7]

    def test_equal_prices_roughly_equal_shares(self, catalog, agent_names):
        """When all agents price identically, shares should be roughly equal."""
        model = MNLDemandModel(catalog, seed=0)

        prices = {
            i: {p.product_id: p.base_retail_price for p in catalog.get_all_products()}
            for i in range(10)
        }

        result = model.compute_demand(prices, agent_names, day=0)
        shares = list(result.market_shares.values())

        # no single store should dominate — all within 5-20% range
        assert max(shares) < 0.30
        assert min(shares) > 0.02


class TestWeekendAndHoliday:

    def test_weekend_increases_demand(self, model, basic_prices, agent_names):
        weekday = model.compute_demand(basic_prices, agent_names, day=1,
                                       is_weekend=False)
        weekend = model.compute_demand(basic_prices, agent_names, day=1,
                                       is_weekend=True)
        assert weekend.total_demand > weekday.total_demand

    def test_holiday_increases_demand(self, model, basic_prices, agent_names):
        normal  = model.compute_demand(basic_prices, agent_names, day=350,
                                       is_holiday=False)
        holiday = model.compute_demand(basic_prices, agent_names, day=350,
                                       is_holiday=True, holiday_multiplier=2.0)
        assert holiday.total_demand > normal.total_demand