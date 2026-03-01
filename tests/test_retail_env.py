import numpy as np
import pytest
from environment.retail_env import RetailEnv, PRICE_ACTIONS


@pytest.fixture
def env():
    e = RetailEnv()
    e.reset()
    return e


class TestRetailEnvInit:

    def test_correct_number_of_agents(self, env):
        assert len(env.agents) == 10

    def test_all_retailer_names_present(self, env):
        expected = {"walmart", "target", "amazon_fresh", "qfc", "safeway",
                    "kroger", "trader_joes", "whole_foods", "aldi", "costco"}
        assert set(env.agents) == expected

    def test_observation_space_shape(self, env):
        for agent in env.agents:
            assert env.observation_space(agent).shape == (266,)

    def test_action_space_shape(self, env):
        for agent in env.agents:
            assert env.action_space(agent).shape == (50,)

    def test_action_space_n_actions(self, env):
        for agent in env.agents:
            assert all(n == 7 for n in env.action_space(agent).nvec)


class TestRetailEnvReset:

    def test_reset_returns_observations(self):
        env = RetailEnv()
        obs, infos = env.reset()
        assert len(obs) == 10
        for agent, ob in obs.items():
            assert ob.shape == (266,)

    def test_reset_day_is_zero(self):
        env = RetailEnv()
        env.reset()
        assert env.current_day == 0

    def test_reset_agents_active(self):
        env = RetailEnv()
        env.reset()
        assert len(env.agents) == 10

    def test_double_reset_works(self):
        env = RetailEnv()
        env.reset()
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            action = None if (term or trunc) else env.action_space(agent).sample()
            env.step(action)
        env.reset()
        assert env.current_day == 0
        assert len(env.agents) == 10


class TestRetailEnvStep:

    def test_observation_shape_after_step(self, env):
        agent = env.agent_selection
        obs = env.observe(agent)
        assert obs.shape == (266,)

    def test_one_full_day(self):
        env = RetailEnv()
        env.reset()
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            action = None if (term or trunc) else env.action_space(agent).sample()
            env.step(action)
        assert env.current_day == 365  # episode ends after 365 days

    def test_episode_metrics_recorded(self):
        env = RetailEnv()
        env.reset()
        for agent in env.agent_iter():
            obs, reward, term, trunc, info = env.last()
            action = None if (term or trunc) else env.action_space(agent).sample()
            env.step(action)
        assert len(env.episode_metrics) > 0
        assert "revenues" in env.episode_metrics[0]
        assert "market_shares" in env.episode_metrics[0]

    def test_termination_after_episode_length(self):
        env = RetailEnv()
        env.reset()
        # run full episode
        done = False
        step_count = 0
        while env.agents:
            for agent in env.agent_iter():
                obs, reward, term, trunc, info = env.last()
                action = None if (term or trunc) else env.action_space(agent).sample()
                env.step(action)
                step_count += 1
                if step_count > 365 * 10 + 10:
                    done = True
                    break
            if done:
                break
        assert env.current_day >= env.episode_length


class TestPriceActions:

    def test_price_actions_correct_count(self):
        assert len(PRICE_ACTIONS) == 7

    def test_price_actions_include_hold(self):
        assert 1.00 in PRICE_ACTIONS.values()

    def test_price_actions_symmetric(self):
        values = list(PRICE_ACTIONS.values())
        assert min(values) < 1.0
        assert max(values) > 1.0

    def test_price_never_below_cost(self):
        env = RetailEnv()
        env.reset()
        # apply action 0 (most aggressive cut) to all products
        aid = 0
        action = np.zeros(50, dtype=int)  # all -15%
        env._apply_action(aid, action)
        for product in env.catalog.get_all_products():
            pid = product.product_id
            assert env.current_prices[aid][pid] >= product.base_cost * 1.01


class TestObservationContent:

    def test_observation_bounded(self, env):
        for agent in env.agents:
            obs = env.observe(agent)
            assert obs.min() >= 0.0
            assert not np.any(np.isnan(obs))
            assert not np.any(np.isinf(obs))

    def test_agent_onehot_correct(self, env):
        for i, agent in enumerate(env.AGENT_NAMES):
            obs = env.observe(agent)
            onehot = obs[-10:]  # last 10 elements
            assert onehot[i] == 1.0
            assert onehot.sum() == 1.0