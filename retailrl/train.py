"""
RetailRL — Command-line training script
========================================
Usage examples:

  # Full 500-episode training run
  python -m retailrl.train --episodes 500 --seed 42

  # Quick smoke test (5 episodes)
  python -m retailrl.train --episodes 5 --seed 42

  # Ablation: disable OpponentEncoder
  python -m retailrl.train --episodes 500 --seed 42 --no-opponent-encoder

  # Ablation: fix all agents to one algorithm
  python -m retailrl.train --episodes 500 --seed 42 --force-algo DQN

  # Ablation: fix all agents to one reward function
  python -m retailrl.train --episodes 500 --seed 42 --force-reward pure_revenue

  # Custom output directory
  python -m retailrl.train --episodes 500 --seed 42 --out results/run_no_encoder
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Train RetailRL multi-agent pricing simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Core
    p.add_argument("--episodes",  type=int, default=500,
                   help="Number of training episodes (default: 500)")
    p.add_argument("--seed",      type=int, default=42,
                   help="Random seed for reproducibility (default: 42)")
    p.add_argument("--out",       type=str, default="results",
                   help="Output directory for results (default: results/)")
    p.add_argument("--config",    type=str, default="config/agent_config.yaml",
                   help="Path to YAML config (default: config/agent_config.yaml)")

    # Ablation flags
    p.add_argument("--no-opponent-encoder", action="store_true",
                   help="Disable OpponentEncoder (ablation: raw competitor prices only)")
    p.add_argument("--force-algo", type=str, default=None,
                   choices=["DQN", "PPO", "A2C", "QTable"],
                   help="Force all neural agents to use this algorithm (ablation)")
    p.add_argument("--force-reward", type=str, default=None,
                   help="Force all agents to use this reward function (ablation)")

    # Checkpointing
    p.add_argument("--save-every",     type=int, default=50,
                   help="Save checkpoints every N episodes (default: 50)")
    p.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                   help="Directory for agent checkpoints (default: checkpoints/)")

    # Verbosity
    p.add_argument("--quiet", action="store_true",
                   help="Suppress per-episode logging")

    return p.parse_args()


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Agent factory ─────────────────────────────────────────────────────────────

# Default agent configuration — mirrors the notebook setup
DEFAULT_AGENT_CONFIG = {
    "walmart":      {"algo": "DQN",    "reward_fn": "pure_revenue"},
    "target":       {"algo": "DQN",    "reward_fn": "profit_margin"},
    "amazon_fresh": {"algo": "DQN",    "reward_fn": "market_share"},
    "qfc":          {"algo": "QTable", "reward_fn": "revenue_with_inventory"},
    "safeway":      {"algo": "PPO",    "reward_fn": "long_term_value"},
    "kroger":       {"algo": "PPO",    "reward_fn": "promo_aware_profit"},
    "trader_joes":  {"algo": "A2C",    "reward_fn": "premium_floor"},
    "whole_foods":  {"algo": "A2C",    "reward_fn": "prestige_reward"},
    "aldi":         {"algo": "QTable", "reward_fn": "discount_maximization"},
    "costco":       {"algo": "QTable", "reward_fn": "bulk_volume"},
}

def build_agents(env, args, config: dict) -> dict:
    """Construct all agents according to config and ablation flags.

    All agents share the same constructor signature:
        (agent_id, agent_name, obs_size, n_products, reward_fn, config, seed)
    """
    from agents.dqn.dqn_agent       import DQNAgent
    from agents.ppo.ppo_agent       import PPOAgent
    from agents.a2c.a2c_agent       import A2CAgent
    from agents.qtable.qtable_agent import QTableAgent

    obs_size   = env.observation_spaces[env.possible_agents[0]].shape[0]
    n_products = 50   # fixed by environment

    ALGO_MAP = {
        "DQN":    DQNAgent,
        "PPO":    PPOAgent,
        "A2C":    A2CAgent,
        "QTable": QTableAgent,
    }

    # config keys per algorithm (matches agent_config.yaml)
    CONFIG_KEY = {
        "DQN":    "dqn",
        "PPO":    "ppo",
        "A2C":    "a2c",
        "QTable": "qtable",
    }

    agents = {}
    for i, (name, cfg) in enumerate(DEFAULT_AGENT_CONFIG.items()):
        algo      = args.force_algo   if args.force_algo   else cfg["algo"]
        reward_fn = args.force_reward if args.force_reward else cfg["reward_fn"]

        agent_class  = ALGO_MAP[algo]
        agent_config = config.get(CONFIG_KEY[algo], {})

        agent = agent_class(
            agent_id   = i,
            agent_name = name,
            obs_size   = obs_size,
            n_products = n_products,
            reward_fn  = reward_fn,
            config     = agent_config,
            seed       = args.seed + i,   # offset seed per agent for diversity
        )

        agents[name] = agent

    return agents


# ── Results export ────────────────────────────────────────────────────────────

def export_results(results, agents, out_dir: Path, args):
    """Save results in the exact format the dashboard expects."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # training_results.csv
    rows = []
    for r in results:
        for name in r.revenues:
            agent = agents[name]
            rows.append({
                "episode":      r.episode,
                "agent":        name,
                "algorithm":    agent.__class__.__name__,
                "reward_fn":    getattr(agent, "reward_fn", "unknown"),
                "revenue":      r.revenues.get(name, 0),
                "market_share": r.market_shares.get(name, 0),
                "total_reward": r.total_rewards.get(name, 0),
            })
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "training_results.csv", index=False)

    # agent_meta.json
    agent_meta = {
        name: {
            "algorithm": agent.__class__.__name__,
            "reward_fn": getattr(agent, "reward_fn", "unknown"),
        }
        for name, agent in agents.items()
    }
    with open(out_dir / "agent_meta.json", "w") as f:
        json.dump(agent_meta, f, indent=2)

    # episode_summaries.json
    summaries = [
        {
            "episode":      r.episode,
            "revenues":     r.revenues,
            "market_shares": r.market_shares,
            "total_rewards": r.total_rewards,
        }
        for r in results
    ]
    with open(out_dir / "episode_summaries.json", "w") as f:
        json.dump(summaries, f, indent=2)

    # run_config.json — records exactly what was run for reproducibility
    run_config = {
        "episodes":            args.episodes,
        "seed":                args.seed,
        "no_opponent_encoder": args.no_opponent_encoder,
        "force_algo":          args.force_algo,
        "force_reward":        args.force_reward,
        "config":              args.config,
        "timestamp":           time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    with open(out_dir / "run_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    print(f"\nResults saved to {out_dir}/")
    print(f"  training_results.csv     ({len(df)} rows)")
    print(f"  agent_meta.json          ({len(agent_meta)} agents)")
    print(f"  episode_summaries.json   ({len(summaries)} episodes)")
    print(f"  run_config.json          (seed={args.seed}, encoder={not args.no_opponent_encoder})")


# ── Progress reporting ────────────────────────────────────────────────────────

def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        return f"{h}h {m}m"


def print_episode_progress(episode: int, n_episodes: int, result,
                            ep_start: float, run_start: float):
    """Print a single-line progress update after each episode."""
    now        = time.time()
    ep_time    = now - ep_start
    elapsed    = now - run_start
    eps_done   = episode + 1
    eps_left   = n_episodes - eps_done
    eta        = (elapsed / eps_done) * eps_left if eps_done > 0 else 0

    # top revenue agent this episode
    if result.revenues:
        top_name, top_rev = max(result.revenues.items(), key=lambda x: x[1])
        leader = f"{top_name} ${top_rev/1e6:.1f}M"
    else:
        leader = "—"

    pct  = eps_done / n_episodes * 100
    bar_filled = int(pct / 5)
    bar  = "█" * bar_filled + "░" * (20 - bar_filled)

    print(
        f"  [{bar}] {pct:5.1f}%  "
        f"ep {eps_done:>4}/{n_episodes}  "
        f"ep_time {ep_time:.1f}s  "
        f"elapsed {format_time(elapsed)}  "
        f"ETA {format_time(eta)}  "
        f"leader: {leader}",
        flush=True,
    )


def print_summary(results, elapsed: float):
    """Print a clean summary table after training."""
    if not results:
        return

    last50 = [r for r in results if r.episode >= len(results) - 50]
    print("\n" + "─" * 60)
    print(f"  Training complete — {len(results)} episodes in {format_time(elapsed)}")
    print("─" * 60)
    print(f"  {'Agent':<16} {'Avg Revenue':>14} {'Mkt Share':>12} {'Avg Reward':>13}")
    print("─" * 60)

    agent_names = list(results[0].revenues.keys())
    rows = []
    for name in agent_names:
        avg_rev   = np.mean([r.revenues.get(name, 0)      for r in last50])
        avg_share = np.mean([r.market_shares.get(name, 0) for r in last50])
        avg_rwd   = np.mean([r.total_rewards.get(name, 0) for r in last50])
        rows.append((name, avg_rev, avg_share, avg_rwd))

    rows.sort(key=lambda x: x[1], reverse=True)
    for name, rev, share, rwd in rows:
        print(f"  {name:<16} ${rev:>12,.0f}  {share:>10.4f}  {rwd:>12,.0f}")

    print("─" * 60)
    print(f"  (averages over final 50 episodes)\n")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    set_seed(args.seed)

    # Print run configuration
    print("\n" + "═" * 60)
    print("  RetailRL — Multi-Agent Pricing Simulation")
    print("═" * 60)
    print(f"  Episodes:          {args.episodes}")
    print(f"  Seed:              {args.seed}")
    print(f"  Output:            {args.out}/")
    print(f"  OpponentEncoder:   {'disabled (ablation)' if args.no_opponent_encoder else 'enabled'}")
    if args.force_algo:
        print(f"  Force algo:        {args.force_algo} (ablation)")
    if args.force_reward:
        print(f"  Force reward:      {args.force_reward} (ablation)")
    print("═" * 60 + "\n")

    # Load config
    config = {}
    config_path = Path(args.config)
    if config_path.exists():
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    else:
        print(f"  [warn] Config not found at {args.config}, using defaults")

    # Build environment
    from environment.retail_env import RetailEnv
    env = RetailEnv()

    # Build agents
    agents = build_agents(env, args, config)

    # Print agent table
    print(f"  {'Agent':<16} {'Algorithm':<12} {'Reward Function':<28} {'Encoder'}")
    print("  " + "─" * 68)
    for name, agent in agents.items():
        algo      = agent.__class__.__name__.replace("Agent", "")
        reward_fn = getattr(agent, "reward_fn", "—")
        encoder   = "✓" if getattr(agent, "use_opponent_encoder", False) else "—"
        print(f"  {name:<16} {algo:<12} {reward_fn:<28} {encoder}")
    print()

    # Build trainer
    from utils.trainer import Trainer
    trainer = Trainer(
        env=env,
        agents=agents,
        checkpoint_dir=Path(args.checkpoint_dir),
        save_every=args.save_every,
    )

    # Train
    start = time.time()
    print(f"  Starting training...\n")
    try:
        results = []
        for episode in range(args.episodes):
            ep_start = time.time()
            result   = trainer._run_episode(episode, training=True)
            trainer.episode_results.append(result)
            results.append(result)

            if not args.quiet:
                print_episode_progress(episode, args.episodes, result,
                                       ep_start, start)

            if (episode + 1) % args.save_every == 0:
                trainer._save_checkpoints(episode)

    except KeyboardInterrupt:
        print("\n\n  [interrupted] Saving partial results...")
        results = trainer.episode_results

    elapsed = time.time() - start

    # Export and summarize
    export_results(results, agents, Path(args.out), args)
    if not args.quiet:
        print_summary(results, elapsed)


if __name__ == "__main__":
    main()
