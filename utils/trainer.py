import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

from environment.retail_env import RetailEnv
from agents.base_agent import BaseRLAgent
from agents.baselines.baseline_agents import BaseAgent
from utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class EpisodeResult:
    episode:        int
    total_rewards:  Dict[str, float]
    market_shares:  Dict[str, float]
    revenues:       Dict[str, float]
    episode_length: int
    training_metrics: Dict[str, dict] = field(default_factory=dict)


class Trainer:
    """
    Centralized training loop for all 10 agents.

    Handles:
    - Mixed agent types (RL agents + rule-based baselines)
    - Per-agent reward routing
    - Training metrics collection
    - Checkpoint saving
    - Episode logging
    """

    def __init__(
        self,
        env:        RetailEnv,
        agents:     Dict[str, object],   # name -> agent
        checkpoint_dir: Path = Path("checkpoints"),
        save_every:     int  = 10,
    ):
        self.env            = env
        self.agents         = agents
        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_every     = save_every
        self.episode_results: List[EpisodeResult] = []

        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Trainer initialized | "
            f"agents={list(agents.keys())} | "
            f"save_every={save_every}"
        )

    def train(self, n_episodes: int) -> List[EpisodeResult]:
        """Run n_episodes of training."""
        for episode in range(n_episodes):
            result = self._run_episode(episode, training=True)
            self.episode_results.append(result)

            if (episode + 1) % self.save_every == 0:
                self._save_checkpoints(episode)
                self._log_progress(episode, result)

        return self.episode_results

    def evaluate(self, n_episodes: int = 3) -> List[EpisodeResult]:
        """Run evaluation episodes (no learning)."""
        results = []
        for episode in range(n_episodes):
            result = self._run_episode(episode, training=False)
            results.append(result)
        return results

    def _run_episode(self, episode: int, training: bool) -> EpisodeResult:
        obs_dict, _ = self.env.reset()

        # set training mode
        for agent in self.agents.values():
            if hasattr(agent, 'set_training'):
                agent.set_training(training)
            if hasattr(agent, 'on_episode_start'):
                agent.on_episode_start()

        cumulative_rewards = {name: 0.0 for name in self.agents}
        training_metrics   = {name: {} for name in self.agents}
        prev_obs           = {name: obs for name, obs in obs_dict.items()}

        step = 0
        for agent_name in self.env.agent_iter():
            obs, reward, term, trunc, info = self.env.last()
            done = term or trunc

            agent = self.agents[agent_name]

            if done:
                self.env.step(None)
                continue

            # get action
            action = agent.act(obs)
            self.env.step(action)

            # update cumulative reward
            cumulative_rewards[agent_name] += reward

            # learn (only RL agents, only when training)
            if training and reward != 0.0:
                metrics = self._learn_step(
                    agent_name, agent, prev_obs[agent_name],
                    action, reward, obs, done
                )
                if metrics:
                    training_metrics[agent_name] = metrics

            prev_obs[agent_name] = obs
            step += 1

        # collect final episode metrics
        final_metrics = self.env.episode_metrics
        avg_shares  = {
            self.env.AGENT_NAMES[m['market_shares'].keys().__iter__().__next__()]: 0.0
            for m in final_metrics[:1]
        } if final_metrics else {}

        # aggregate market shares and revenues over episode
        market_shares = {}
        revenues      = {}
        if final_metrics:
            for name in self.env.AGENT_NAMES:
                aid = self.env._agent_ids[name]
                shares = [m['market_shares'].get(aid, 0) for m in final_metrics]
                revs   = [m['revenues'].get(aid, 0) for m in final_metrics]
                market_shares[name] = float(np.mean(shares))
                revenues[name]      = float(np.sum(revs))

        result = EpisodeResult(
            episode=episode,
            total_rewards=cumulative_rewards,
            market_shares=market_shares,
            revenues=revenues,
            episode_length=len(final_metrics),
            training_metrics=training_metrics,
        )

        # on_episode_end callbacks
        for name, agent in self.agents.items():
            if hasattr(agent, 'on_episode_end'):
                agent.on_episode_end(cumulative_rewards.get(name, 0.0))

        return result

    def _learn_step(
        self, name, agent, obs, action, reward, next_obs, done
    ):
        """Route learning call to correct agent interface."""
        from agents.dqn.dqn_agent import DQNAgent
        from agents.ppo.ppo_agent import PPOAgent
        from agents.a2c.a2c_agent import A2CAgent
        from agents.qtable.qtable_agent import QTableAgent

        if isinstance(agent, DQNAgent):
            return agent.learn(obs, action, reward, next_obs, done)
        elif isinstance(agent, (PPOAgent, A2CAgent)):
            return agent.learn(reward=reward, done=done, next_obs=next_obs)
        elif isinstance(agent, QTableAgent):
            return agent.learn(reward=reward, done=done, next_obs=next_obs)
        return None

    def _save_checkpoints(self, episode: int):
        for name, agent in self.agents.items():
            if hasattr(agent, 'save'):
                try:
                    agent.save(self.checkpoint_dir / name)
                except Exception as e:
                    logger.warning(f"Failed to save {name}: {e}")

    def _log_progress(self, episode: int, result: EpisodeResult):
        top = sorted(
            result.revenues.items(), key=lambda x: x[1], reverse=True
        )[:3] if result.revenues else []

        logger.info(
            f"Episode {episode+1} | "
            f"top agents: {[(n, f'${r:,.0f}') for n,r in top]}"
        )