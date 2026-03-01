import numpy as np
import pytest
from agents.baselines.baseline_agents import (
    RandomAgent, FixedMarginAgent, AlwaysCheapestAgent
)


@pytest.fixture
def dummy_obs():
    """Fake 266-dim observation with own prices slightly above comp prices."""
    obs = np.ones(266, dtype=np.float32)
    obs[0:50]  = 1.10   # own prices 10% above base
    obs[50:100] = 1.00  # competitor avg at base
    return obs


@pytest.fixture
def cheap_obs():
    """Observation where own prices are already below competitors."""
    obs = np.ones(266, dtype=np.float32)
    obs[0:50]  = 0.90   # own prices below base
    obs[50:100] = 1.00  # competitors at base
    return obs


class TestRandomAgent:

    def test_action_shape(self, dummy_obs):
        agent = RandomAgent(0, "test", 50)
        action = agent.act(dummy_obs)
        assert action.shape == (50,)

    def test_action_values_in_range(self, dummy_obs):
        agent = RandomAgent(0, "test", 50)
        action = agent.act(dummy_obs)
        assert action.min() >= 0
        assert action.max() <= 6

    def test_different_seeds_differ(self, dummy_obs):
        a1 = RandomAgent(0, "test", 50, seed=1)
        a2 = RandomAgent(0, "test", 50, seed=2)
        assert not np.array_equal(a1.act(dummy_obs), a2.act(dummy_obs))

    def test_repr(self):
        agent = RandomAgent(0, "walmart", 50)
        assert "RandomAgent" in repr(agent)
        assert "walmart" in repr(agent)


class TestFixedMarginAgent:

    def test_action_shape(self, dummy_obs):
        agent = FixedMarginAgent(1, "target", 50)
        action = agent.act(dummy_obs)
        assert action.shape == (50,)

    def test_always_holds(self, dummy_obs):
        """FixedMarginAgent should always return action 3 (hold)."""
        agent = FixedMarginAgent(1, "target", 50)
        action = agent.act(dummy_obs)
        assert np.all(action == 3)

    def test_holds_regardless_of_obs(self, cheap_obs):
        agent = FixedMarginAgent(1, "target", 50)
        action = agent.act(cheap_obs)
        assert np.all(action == 3)

    def test_repr(self):
        agent = FixedMarginAgent(1, "target", 50)
        assert "FixedMarginAgent" in repr(agent)


class TestAlwaysCheapestAgent:

    def test_action_shape(self, dummy_obs):
        agent = AlwaysCheapestAgent(0, "walmart", 50)
        action = agent.act(dummy_obs)
        assert action.shape == (50,)

    def test_cuts_when_above_competitors(self, dummy_obs):
        """When own price > comp price, should use action 1 (-10%)."""
        agent = AlwaysCheapestAgent(0, "walmart", 50)
        action = agent.act(dummy_obs)
        assert np.all(action == 1)

    def test_mild_cut_when_already_cheap(self, cheap_obs):
        """When own price <= comp price, should use action 2 (-5%)."""
        agent = AlwaysCheapestAgent(0, "walmart", 50)
        action = agent.act(cheap_obs)
        assert np.all(action == 2)

    def test_never_raises_prices(self, dummy_obs):
        """AlwaysCheapest should never use actions 4, 5, 6 (price increases)."""
        agent = AlwaysCheapestAgent(0, "walmart", 50)
        action = agent.act(dummy_obs)
        assert np.all(action <= 3)

    def test_repr(self):
        agent = AlwaysCheapestAgent(0, "walmart", 50)
        assert "AlwaysCheapestAgent" in repr(agent)