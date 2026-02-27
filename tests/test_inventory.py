import pytest
from environment.product_catalog import ProductCatalog
from environment.inventory import InventoryManager, InventoryState


@pytest.fixture
def catalog():
    return ProductCatalog(seed=42)


@pytest.fixture
def inv(catalog):
    return InventoryManager(catalog, n_agents=10, seed=42)


@pytest.fixture
def no_sales():
    return {aid: {pid: 0 for pid in range(50)} for aid in range(10)}


@pytest.fixture
def no_incoming():
    return {aid: {} for aid in range(10)}


class TestInventoryInit:

    def test_initial_stock_correct(self, inv):
        for aid in range(10):
            for pid in range(50):
                assert inv.stock[aid][pid] == 500

    def test_initial_pipeline_zero(self, inv):
        for aid in range(10):
            for pid in range(50):
                assert inv.pipeline[aid][pid] == 0

    def test_repr(self, inv):
        assert "agents=10" in repr(inv)
        assert "products=50" in repr(inv)


class TestInventoryStep:

    def test_stock_depletes_after_sales(self, inv, no_incoming):
        units_sold = {aid: {pid: 10 for pid in range(50)} for aid in range(10)}
        inv.step(units_sold, no_incoming)
        for aid in range(10):
            for pid in range(50):
                assert inv.stock[aid][pid] == 490

    def test_stock_cannot_go_negative(self, inv, no_incoming):
        """Selling more than stock should floor at zero."""
        units_sold = {aid: {pid: 9999 for pid in range(50)} for aid in range(10)}
        inv.step(units_sold, no_incoming)
        for aid in range(10):
            for pid in range(50):
                assert inv.stock[aid][pid] == 0

    def test_incoming_stock_received(self, inv, no_sales):
        incoming = {aid: {pid: 100 for pid in range(50)} for aid in range(10)}
        inv.step(no_sales, incoming)
        for aid in range(10):
            for pid in range(50):
                assert inv.stock[aid][pid] == 600

    def test_holding_cost_positive(self, inv, no_sales, no_incoming):
        states = inv.step(no_sales, no_incoming)
        for aid in range(10):
            assert states[aid].holding_cost > 0

    def test_stockout_penalty_when_demand_exceeds_stock(self, inv, no_incoming):
        """Penalty should apply when agent has demand but zero stock."""
        # first sell everything
        units_sold = {aid: {pid: 9999 for pid in range(50)} for aid in range(10)}
        inv.step(units_sold, no_incoming)

        # now try to sell again with zero stock
        units_sold2 = {aid: {pid: 10 for pid in range(50)} for aid in range(10)}
        states = inv.step(units_sold2, no_incoming)

        for aid in range(10):
            assert states[aid].stockout_penalty > 0

    def test_no_stockout_penalty_when_no_demand(self, inv, no_incoming):
        """No penalty if stock is zero but demand is also zero."""
        units_sold = {aid: {pid: 9999 for pid in range(50)} for aid in range(10)}
        inv.step(units_sold, no_incoming)

        zero_demand = {aid: {pid: 0 for pid in range(50)} for aid in range(10)}
        states = inv.step(zero_demand, no_incoming)
        for aid in range(10):
            assert states[aid].stockout_penalty == 0.0

    def test_returns_inventory_state(self, inv, no_sales, no_incoming):
        states = inv.step(no_sales, no_incoming)
        for aid in range(10):
            assert isinstance(states[aid], InventoryState)


class TestReorders:

    def test_reorder_triggered_when_stock_low(self, inv, no_incoming):
        """Reorder should trigger when stock + pipeline <= reorder_point (100)."""
        # sell down to near reorder point
        units_sold = {aid: {pid: 405 for pid in range(50)} for aid in range(10)}
        inv.step(units_sold, no_incoming)

        reorders = inv.place_reorders()
        for aid in range(10):
            for pid in range(50):
                assert reorders[aid][pid] == 400  # reorder_quantity

    def test_no_reorder_when_stock_healthy(self, inv, no_sales, no_incoming):
        """No reorder when stock is well above reorder point."""
        inv.step(no_sales, no_incoming)
        reorders = inv.place_reorders()
        for aid in range(10):
            total_reorders = sum(reorders[aid].values())
            assert total_reorders == 0

    def test_pipeline_updated_after_reorder(self, inv, no_incoming):
        units_sold = {aid: {pid: 405 for pid in range(50)} for aid in range(10)}
        inv.step(units_sold, no_incoming)
        inv.place_reorders()
        for aid in range(10):
            for pid in range(50):
                assert inv.pipeline[aid][pid] == 400


class TestObservationHelpers:

    def test_stock_vector_shape(self, inv):
        vec = inv.get_stock_vector(0)
        assert vec.shape == (50,)

    def test_stock_vector_normalized(self, inv):
        vec = inv.get_stock_vector(0)
        assert vec.min() >= 0.0
        assert vec.max() <= 2.0   # allowing headroom above 1 for restock

    def test_stockout_vector_binary(self, inv, no_incoming):
        units_sold = {aid: {pid: 9999 for pid in range(50)} for aid in range(10)}
        inv.step(units_sold, no_incoming)
        vec = inv.get_stockout_vector(0)
        assert set(vec.tolist()).issubset({0.0, 1.0})
        assert vec.sum() == 50   # all products stocked out

    def test_reset_restores_initial_state(self, inv, no_incoming):
        units_sold = {aid: {pid: 200 for pid in range(50)} for aid in range(10)}
        inv.step(units_sold, no_incoming)
        inv.reset()
        assert inv.stock[0][0] == 500
        assert inv.pipeline[0][0] == 0