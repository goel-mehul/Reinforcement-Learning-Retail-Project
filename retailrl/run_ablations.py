"""
RetailRL — Ablation study runner
==================================
Runs three ablation experiments and produces comparison plots.

Usage:
  python -m retailrl.run_ablations --episodes 500 --seed 42

What it runs:
  1. Baseline:          full model (OpponentEncoder enabled, default reward/algo)
  2. No encoder:        OpponentEncoder disabled, everything else identical
  3. Algo-only effect:  all agents use DQN, keep reward functions varied
  4. Reward-only effect: all agents use DQN, force same reward (pure_revenue)

Outputs (in results/ablations/):
  encoder_ablation.png      — with vs without OpponentEncoder
  algo_comparison.png       — DQN vs PPO vs A2C vs QTable (reward fixed)
  reward_comparison.png     — reward function effect (algo fixed to DQN)
  ablation_summary.csv      — final-50-ep stats for all runs
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "figure.facecolor":  "#13131a",
    "axes.facecolor":    "#1a1a24",
    "axes.edgecolor":    "#2a2a38",
    "axes.labelcolor":   "#a0a0b8",
    "axes.titlecolor":   "#c8c8e0",
    "xtick.color":       "#7a7a98",
    "ytick.color":       "#7a7a98",
    "grid.color":        "#22222e",
    "grid.linewidth":    0.6,
    "text.color":        "#d4d4e0",
    "legend.facecolor":  "#1a1a24",
    "legend.edgecolor":  "#2a2a38",
    "font.family":       "sans-serif",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "figure.titlesize":  15,
})

AGENT_COLORS = {
    "walmart":      "#7eb8d4",
    "target":       "#c47eb0",
    "amazon_fresh": "#d4a76a",
    "qfc":          "#7ec4a0",
    "safeway":      "#a07ec4",
    "kroger":       "#c4907e",
    "trader_joes":  "#7eaac4",
    "whole_foods":  "#90c47e",
    "aldi":         "#c4b87e",
    "costco":       "#b8c47e",
}

ALGO_COLORS = {
    "DQN":    "#7eb8d4",
    "PPO":    "#c4907e",
    "A2C":    "#7ec4a0",
    "QTable": "#c4b87e",
}


def smooth(s, w=20):
    return pd.Series(s).rolling(w, min_periods=1).mean().values


# ── Run a training experiment ─────────────────────────────────────────────────

def run_experiment(label: str, out_dir: str, episodes: int, seed: int,
                   extra_flags: list = None):
    """Call train.py as a subprocess with given flags."""
    cmd = [
        sys.executable, "-m", "retailrl.train",
        "--episodes", str(episodes),
        "--seed",     str(seed),
        "--out",      out_dir,
    ]
    if extra_flags:
        cmd.extend(extra_flags)

    print(f"\n{'═' * 60}")
    print(f"  Experiment: {label}")
    print(f"{'═' * 60}")
    result = subprocess.run(cmd, check=True)
    print(f"\n  ✓ {label} complete")
    return Path(out_dir)


def load_run(out_dir: Path) -> tuple[pd.DataFrame, dict]:
    df   = pd.read_csv(out_dir / "training_results.csv")
    meta = json.loads((out_dir / "agent_meta.json").read_text())
    return df, meta


# ── Plot 1: OpponentEncoder ablation ─────────────────────────────────────────

def plot_encoder_ablation(df_with: pd.DataFrame, df_without: pd.DataFrame,
                          out_path: Path):
    """Compare total market revenue with vs without OpponentEncoder."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Ablation: OpponentEncoder vs No Encoder", fontweight="bold",
                 color="#e0e0f0")

    for ax, (df, label, color) in zip(axes, [
        (df_with,    "With OpponentEncoder",    "#7eb8d4"),
        (df_without, "Without OpponentEncoder", "#c4907e"),
    ]):
        # Plot each agent's revenue
        agents = df["agent"].unique()
        for name in agents:
            revs = df[df["agent"] == name].sort_values("episode")["revenue"].values
            ax.plot(smooth(revs, 20), color=AGENT_COLORS.get(name, "#a0a0b8"),
                    linewidth=1.5, alpha=0.85, label=name)

        ax.set_title(label, color=color)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Revenue ($)")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"${x/1e6:.1f}M"))
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8, ncol=2, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


def plot_encoder_metric_comparison(df_with: pd.DataFrame,
                                   df_without: pd.DataFrame,
                                   out_path: Path):
    """Bar chart: final-50-ep avg revenue, with vs without encoder."""
    agents = sorted(df_with["agent"].unique())

    def final50_avg(df, metric="revenue"):
        n = df["episode"].max()
        return df[df["episode"] >= n - 49].groupby("agent")[metric].mean()

    rev_with    = final50_avg(df_with)
    rev_without = final50_avg(df_without)

    x     = np.arange(len(agents))
    width = 0.38

    fig, ax = plt.subplots(figsize=(12, 5))
    fig.suptitle("Final 50-Episode Revenue: OpponentEncoder Ablation",
                 fontweight="bold", color="#e0e0f0")

    bars1 = ax.bar(x - width/2,
                   [rev_with.get(a, 0) for a in agents],
                   width, label="With encoder", color="#7eb8d4", alpha=0.85)
    bars2 = ax.bar(x + width/2,
                   [rev_without.get(a, 0) for a in agents],
                   width, label="No encoder", color="#c4907e", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=35, ha="right")
    ax.set_ylabel("Avg Revenue — final 50 eps ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.1f}M"))
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend()

    # % difference labels
    for i, name in enumerate(agents):
        w = rev_with.get(name, 0)
        wo = rev_without.get(name, 0)
        if wo > 0:
            pct = (w - wo) / wo * 100
            color = "#7ec4a0" if pct >= 0 else "#c4907e"
            ax.annotate(f"{pct:+.0f}%",
                        xy=(x[i], max(w, wo) + 0.02 * max(rev_with.max(), rev_without.max())),
                        ha="center", fontsize=8, color=color)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot 2: Algorithm effect (reward fixed) ───────────────────────────────────

def plot_algo_comparison(runs: dict, out_path: Path):
    """
    runs = {"DQN": df, "PPO": df, "A2C": df}
    All agents use the same reward function; compare convergence by algo.
    """
    fig, ax = plt.subplots(figsize=(11, 5))
    fig.suptitle(
        "Algorithm Comparison (reward=pure_revenue, all agents)",
        fontweight="bold", color="#e0e0f0"
    )

    for algo, df in runs.items():
        # Average revenue across all agents
        avg_per_ep = df.groupby("episode")["revenue"].mean().sort_index()
        ax.plot(smooth(avg_per_ep.values, 20),
                color=ALGO_COLORS.get(algo, "#a0a0b8"),
                linewidth=2.2, label=algo)

    ax.set_xlabel("Episode")
    ax.set_ylabel("Avg Revenue per Agent ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.1f}M"))
    ax.grid(True, alpha=0.4)
    ax.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot 3: Reward function effect (algo fixed) ───────────────────────────────

def plot_reward_comparison(df_varied: pd.DataFrame, out_path: Path):
    """
    Show revenue vs total_reward for each agent when algo is fixed to DQN.
    Highlights that reward function drives behavioral divergence even with
    identical algorithms.
    """
    n_ep = df_varied["episode"].max()
    final = df_varied[df_varied["episode"] >= n_ep - 49]
    summary = final.groupby(["agent", "reward_fn"])[["revenue", "total_reward"]].mean()
    summary = summary.reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Reward Function Effect on Strategy (all agents use DQN)",
        fontweight="bold", color="#e0e0f0"
    )

    # Scatter: reward vs revenue
    ax = axes[0]
    for _, row in summary.iterrows():
        name = row["agent"]
        ax.scatter(row["revenue"], row["total_reward"],
                   color=AGENT_COLORS.get(name, "#a0a0b8"),
                   s=120, zorder=5, edgecolors="#13131a", linewidths=1.5)
        ax.annotate(f"  {name}", xy=(row["revenue"], row["total_reward"]),
                    fontsize=9, color="#8a8aaa", va="center")

    ax.set_title("High Reward ≠ High Revenue\n(algorithm fixed to DQN)")
    ax.set_xlabel("Avg Revenue — last 50 eps ($)")
    ax.set_ylabel("Avg Total Reward — last 50 eps")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.1f}M"))
    ax.grid(True, alpha=0.4)

    # Bar chart: revenue by reward function
    ax2 = axes[1]
    summary_sorted = summary.sort_values("revenue", ascending=True)
    colors = [AGENT_COLORS.get(n, "#a0a0b8") for n in summary_sorted["agent"]]
    bars = ax2.barh(summary_sorted["reward_fn"], summary_sorted["revenue"],
                    color=colors, alpha=0.85)
    ax2.set_title("Revenue by Reward Function\n(algorithm fixed to DQN)")
    ax2.set_xlabel("Avg Revenue — last 50 eps ($)")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(
        lambda x, _: f"${x/1e6:.1f}M"))
    ax2.grid(True, axis="x", alpha=0.4)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary_table(runs: dict, out_path: Path):
    """Build a CSV comparing final-50-ep stats across all ablation runs."""
    rows = []
    for run_name, df in runs.items():
        n_ep  = df["episode"].max()
        final = df[df["episode"] >= n_ep - 49]
        for name in df["agent"].unique():
            adf = final[final["agent"] == name]
            rows.append({
                "run":          run_name,
                "agent":        name,
                "algorithm":    adf["algorithm"].iloc[0],
                "reward_fn":    adf["reward_fn"].iloc[0],
                "avg_revenue":  adf["revenue"].mean(),
                "avg_mkt_share":adf["market_share"].mean(),
                "avg_reward":   adf["total_reward"].mean(),
            })

    summary = pd.DataFrame(rows)
    summary.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Run RetailRL ablation studies")
    p.add_argument("--episodes",   type=int, default=500)
    p.add_argument("--seed",       type=int, default=42)
    p.add_argument("--out",        type=str, default="results/ablations")
    p.add_argument("--skip-train", action="store_true",
                   help="Skip training, load existing results from --out")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # ── Training runs ─────────────────────────────────────────────────────────

    if not args.skip_train:
        print("\n" + "═" * 60)
        print("  RetailRL Ablation Study")
        print(f"  Episodes per run: {args.episodes}  Seed: {args.seed}")
        print(f"  Total runs: 6  (~6-7 hours on M2 MacBook Air)")
        print("═" * 60)

        import time
        ablation_start = time.time()
        experiments = [
            ("1/6  Baseline (full model)",         str(out / "baseline"),            []),
            ("2/6  No OpponentEncoder",             str(out / "no_encoder"),          ["--no-opponent-encoder"]),
            ("3/6  All DQN, varied rewards",        str(out / "all_dqn_varied_reward"),["--force-algo", "DQN"]),
            ("4/6  All DQN, pure_revenue",          str(out / "all_dqn_same_reward"), ["--force-algo", "DQN", "--force-reward", "pure_revenue"]),
            ("5/6  All PPO, pure_revenue",          str(out / "all_ppo_same_reward"), ["--force-algo", "PPO", "--force-reward", "pure_revenue"]),
            ("6/6  All A2C, pure_revenue",          str(out / "all_a2c_same_reward"), ["--force-algo", "A2C", "--force-reward", "pure_revenue"]),
        ]

        for i, (label, out_dir, flags) in enumerate(experiments):
            run_experiment(label, out_dir, args.episodes, args.seed,
                           extra_flags=flags if flags else None)

            # print overall ETA after each experiment completes
            elapsed   = time.time() - ablation_start
            done      = i + 1
            remaining = len(experiments) - done
            eta       = (elapsed / done) * remaining
            h, m      = int(eta // 3600), int((eta % 3600) // 60)
            print(f"\n  Overall: {done}/{len(experiments)} experiments done  "
                  f"— {h}h {m}m remaining\n")

    # ── Load results ─────────────────────────────────────────────────────────

    print("\n  Loading results...")
    df_baseline,    _ = load_run(out / "baseline")
    df_no_encoder,  _ = load_run(out / "no_encoder")
    df_dqn_varied,  _ = load_run(out / "all_dqn_varied")
    df_dqn_same,    _ = load_run(out / "all_dqn_same")
    df_ppo_same,    _ = load_run(out / "all_ppo_same")
    df_a2c_same,    _ = load_run(out / "all_a2c_same")

    # ── Plots ─────────────────────────────────────────────────────────────────

    print("\n  Generating plots...")

    # Plot 1a: revenue curves side by side
    plot_encoder_ablation(df_baseline, df_no_encoder,
                          out / "encoder_ablation_curves.png")

    # Plot 1b: final-50-ep bar comparison
    plot_encoder_metric_comparison(df_baseline, df_no_encoder,
                                   out / "encoder_ablation_bars.png")

    # Plot 2: algorithm comparison (all same reward)
    plot_algo_comparison(
        {"DQN": df_dqn_same, "PPO": df_ppo_same, "A2C": df_a2c_same},
        out / "algo_comparison.png"
    )

    # Plot 3: reward function effect (algo fixed to DQN)
    plot_reward_comparison(df_dqn_varied, out / "reward_comparison.png")

    # Summary table
    all_runs = {
        "baseline":       df_baseline,
        "no_encoder":     df_no_encoder,
        "all_dqn_varied": df_dqn_varied,
        "all_dqn_same":   df_dqn_same,
        "all_ppo_same":   df_ppo_same,
        "all_a2c_same":   df_a2c_same,
    }
    build_summary_table(all_runs, out / "ablation_summary.csv")

    print("\n" + "═" * 60)
    print(f"  All ablation outputs in: {out}/")
    print("  encoder_ablation_curves.png   — revenue curves with/without encoder")
    print("  encoder_ablation_bars.png     — final-50-ep bars with % difference")
    print("  algo_comparison.png           — DQN vs PPO vs A2C convergence")
    print("  reward_comparison.png         — reward function behavioral divergence")
    print("  ablation_summary.csv          — all stats in one table")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    main()
