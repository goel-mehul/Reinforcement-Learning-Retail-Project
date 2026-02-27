import pytest
from environment.supply_chain import SupplyChain, PendingOrder


@pytest.fixture
def sc():
    return SupplyChain(n_agents=10, seed=42)


@pytest.fixture
def simple_reorders():
    """One product reordered by all 10 agents."""
    return {aid: {0: 400} for aid in range(10)}


class TestSupplyChainInit:

    def test_repr(self, sc):
        assert "agents=10" in repr(sc)
        assert "pending_orders=0" in repr(sc)
        assert "day=0" in repr(sc)

    def test_starts_empty(self, sc):
        assert sc.n_pending_orders() == 0

    def test_reset_clears_state(self, sc, simple_reorders):
        sc.place_orders(simple_reorders)
        sc.reset()
        assert sc.n_pending_orders() == 0
        assert sc.current_day == 0
        assert sc.disruption_log == []


class TestPlaceOrders:

    def test_orders_added_to_pipeline(self, sc, simple_reorders):
        sc.place_orders(simple_reorders)
        assert sc.n_pending_orders() == 10

    def test_zero_quantity_not_added(self, sc):
        reorders = {aid: {0: 0} for aid in range(10)}
        sc.place_orders(reorders)
        assert sc.n_pending_orders() == 0

    def test_arrival_day_within_lead_time(self, sc, simple_reorders):
        sc.place_orders(simple_reorders)
        for order in sc.pending_orders:
            lead = order.arrival_day - order.order_day
            # lead time can be extended by disruption so use generous upper bound
            assert lead >= sc.lead_min
            assert lead <= sc.lead_max * int(sc.disruption_mult) + 1

    def test_order_day_recorded_correctly(self, sc, simple_reorders):
        sc.current_day = 5
        sc.place_orders(simple_reorders)
        for order in sc.pending_orders:
            assert order.order_day == 5


class TestStep:

    def test_no_delivery_before_lead_time(self, sc, simple_reorders):
        sc.place_orders(simple_reorders)
        # day 0 → step to day 1, nothing should arrive yet (min lead = 2)
        incoming = sc.step()
        assert sum(sum(v.values()) for v in incoming.values()) == 0

    def test_orders_delivered_by_max_lead_time(self):
        """All non-disrupted orders should arrive within lead_max days."""
        sc = SupplyChain(n_agents=2, seed=999)
        # use disruption_prob=0 to avoid disruptions for this test
        sc.disruption_prob = 0.0
        reorders = {0: {0: 400}, 1: {0: 400}}
        sc.place_orders(reorders)

        total_received = 0
        for _ in range(sc.lead_max + 1):
            incoming = sc.step()
            total_received += sum(sum(v.values()) for v in incoming.values())

        assert total_received == 800   # 2 agents x 400 units

    def test_day_increments_on_step(self, sc):
        assert sc.current_day == 0
        sc.step()
        assert sc.current_day == 1
        sc.step()
        assert sc.current_day == 2

    def test_delivered_orders_removed_from_pipeline(self):
        sc = SupplyChain(n_agents=1, seed=0)
        sc.disruption_prob = 0.0
        sc.place_orders({0: {0: 400}})
        initial_pending = sc.n_pending_orders()
        assert initial_pending == 1

        # step through until delivered
        for _ in range(sc.lead_max + 1):
            sc.step()

        assert sc.n_pending_orders() == 0

    def test_incoming_format_correct(self, sc, simple_reorders):
        sc.place_orders(simple_reorders)
        for _ in range(sc.lead_max + 1):
            incoming = sc.step()
        # incoming should be dict of dicts
        for aid, products in incoming.items():
            assert isinstance(aid, int)
            assert isinstance(products, dict)


class TestObservationHelpers:

    def test_days_until_arrival_no_order(self, sc):
        assert sc.get_days_until_arrival(0, 0) == -1

    def test_days_until_arrival_with_order(self, sc):
        sc.disruption_prob = 0.0
        sc.place_orders({0: {0: 400}})
        days = sc.get_days_until_arrival(0, 0)
        assert sc.lead_min <= days <= sc.lead_max

    def test_pipeline_vector_shape(self, sc, simple_reorders):
        sc.place_orders(simple_reorders)
        vec = sc.get_pipeline_vector(0, list(range(50)))
        assert vec.shape == (50,)

    def test_pipeline_vector_normalized(self, sc, simple_reorders):
        sc.place_orders(simple_reorders)
        vec = sc.get_pipeline_vector(0, list(range(50)))
        assert vec.min() >= 0.0


class TestDisruptions:

    def test_disruptions_logged(self):
        """With high disruption probability, disruptions should be logged."""
        sc = SupplyChain(n_agents=10, seed=42)
        sc.disruption_prob = 0.5   # 50% chance to trigger disruptions
        reorders = {aid: {pid: 400 for pid in range(50)} for aid in range(10)}
        sc.place_orders(reorders)
        assert len(sc.disruption_log) > 0

    def test_no_disruptions_when_prob_zero(self):
        sc = SupplyChain(n_agents=10, seed=42)
        sc.disruption_prob = 0.0
        reorders = {aid: {pid: 400 for pid in range(50)} for aid in range(10)}
        sc.place_orders(reorders)
        assert len(sc.disruption_log) == 0