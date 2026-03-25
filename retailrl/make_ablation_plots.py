"""
RetailRL — Ablation plots from existing training data
======================================================
Generates 4 clean ablation plots directly from results/training_results.csv.
No retraining needed.

Usage:
    python -m retailrl.make_ablation_plots
    python -m retailrl.make_ablation_plots --data results/training_results.csv --out results/ablations
"""

import argparse
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
    "DQNAgent":    "#7eb8d4",
    "PPOAgent":    "#c4907e",
    "A2CAgent":    "#7ec4a0",
    "QTableAgent": "#c4b87e",
}

ALGO_LABELS = {
    "DQNAgent":    "DQN",
    "PPOAgent":    "PPO",
    "A2CAgent":    "A2C",
    "QTableAgent": "Q-Table",
}

def smooth(s, w=20):
    return pd.Series(s).rolling(w, min_periods=1).mean().values

def fmt_millions(x, _):
    return f"${x/1e6:.1f}M"

def final50(df, metric="revenue"):
    n = df["episode"].max()
    return df[df["episode"] >= n - 49].groupby("agent")[metric].mean()


# ── Plot 1: Algorithm convergence ─────────────────────────────────────────────

def plot_algo_convergence(df, out_path):
    """Revenue over training grouped by algorithm."""
    fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=False)
    fig.suptitle(
        "Algorithm Convergence — Revenue Over 500 Training Episodes",
        fontweight="bold", color="#e0e0f0", y=1.02
    )

    algos = ["DQNAgent", "PPOAgent", "A2CAgent", "QTableAgent"]

    for ax, algo in zip(axes, algos):
        agents_in_algo = df[df["algorithm"] == algo]["agent"].unique()
        for name in agents_in_algo:
            adf  = df[df["agent"] == name].sort_values("episode")
            revs = adf["revenue"].values
            ax.plot(
                smooth(revs, 20),
                color=AGENT_COLORS.get(name, "#a0a0b8"),
                linewidth=1.8, alpha=0.9, label=name
            )

        ax.set_title(ALGO_LABELS[algo],
                     color=ALGO_COLORS[algo], fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Revenue ($)" if algo == "DQNAgent" else "")
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
        ax.grid(True, alpha=0.4)
        ax.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot 2: Algorithm final performance comparison ────────────────────────────

def plot_algo_final_comparison(df, out_path):
    """Bar chart: final 50-ep avg revenue by algorithm."""
    algos  = ["DQNAgent", "PPOAgent", "A2CAgent", "QTableAgent"]
    n_ep   = df["episode"].max()
    final  = df[df["episode"] >= n_ep - 49]

    algo_stats = []
    for algo in algos:
        adf = final[final["algorithm"] == algo]
        algo_stats.append({
            "algo":    ALGO_LABELS[algo],
            "avg_rev": adf["revenue"].mean(),
            "std_rev": adf["revenue"].std(),
        })

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Algorithm Comparison — Final 50 Episodes",
        fontweight="bold", color="#e0e0f0"
    )

    # Left: avg revenue per algorithm
    ax = axes[0]
    labels = [s["algo"] for s in algo_stats]
    values = [s["avg_rev"] for s in algo_stats]
    colors = [ALGO_COLORS[a] for a in algos]
    bars   = ax.bar(labels, values, color=colors, alpha=0.85, width=0.5)
    ax.set_title("Avg Revenue per Agent by Algorithm")
    ax.set_ylabel("Avg Revenue — last 50 eps ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax.grid(True, axis="y", alpha=0.4)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.02,
                f"${val/1e6:.2f}M", ha="center", fontsize=10,
                color="#c8c8e0")

    # Right: rolling std (policy stability) per algorithm
    ax2 = axes[1]
    for algo in algos:
        agents_in = df[df["algorithm"] == algo]["agent"].unique()
        all_r = np.array([
            df[df["agent"] == n].sort_values("episode")["revenue"].values
            for n in agents_in
        ])
        mean_r   = all_r.mean(axis=0)
        roll_std = pd.Series(mean_r).rolling(30, min_periods=30).std().values
        ax2.plot(roll_std, color=ALGO_COLORS[algo],
                 linewidth=2, label=ALGO_LABELS[algo])

    ax2.set_title("Policy Stability — Rolling Std (window=30)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Rolling Std of Revenue")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax2.grid(True, alpha=0.4)
    ax2.legend()

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot 3: Reward function divergence ────────────────────────────────────────

def plot_reward_divergence(df, out_path):
    """Scatter + bar showing reward fn drives strategy, not algorithm."""
    n_ep    = df["episode"].max()
    final   = df[df["episode"] >= n_ep - 49]
    summary = final.groupby("agent")[["revenue", "total_reward", "market_share"]].mean()
    summary = summary.join(df.groupby("agent")["reward_fn"].first())
    summary = summary.join(df.groupby("agent")["algorithm"].first())

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Reward Function Effect — Same Data, Different Objectives",
        fontweight="bold", color="#e0e0f0"
    )

    # Left: reward vs revenue scatter
    ax = axes[0]
    for name, row in summary.iterrows():
        ax.scatter(
            row["revenue"], row["total_reward"],
            color=AGENT_COLORS.get(name, "#a0a0b8"),
            s=140, zorder=5,
            edgecolors="#13131a", linewidths=1.5
        )
        ax.annotate(
            f"  {name}\n  ({ALGO_LABELS.get(row['algorithm'], row['algorithm'])})",
            xy=(row["revenue"], row["total_reward"]),
            fontsize=8, color="#8a8aaa", va="center"
        )

    ax.set_title("High Reward ≠ High Revenue\nSame competitive environment, different reward functions")
    ax.set_xlabel("Avg Revenue — last 50 eps ($)")
    ax.set_ylabel("Avg Total Reward — last 50 eps")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax.grid(True, alpha=0.4)

    # Right: revenue by agent sorted, colored by algo
    ax2 = axes[1]
    summary_sorted = summary.sort_values("revenue", ascending=True)
    bar_colors = [ALGO_COLORS.get(summary_sorted.loc[n, "algorithm"], "#a0a0b8")
                  for n in summary_sorted.index]
    bars = ax2.barh(
        [f"{n}\n({summary_sorted.loc[n,'reward_fn']})"
         for n in summary_sorted.index],
        summary_sorted["revenue"],
        color=bar_colors, alpha=0.85
    )
    ax2.set_title("Final Revenue by Agent\n(bar color = algorithm)")
    ax2.set_xlabel("Avg Revenue — last 50 eps ($)")
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax2.grid(True, axis="x", alpha=0.4)

    # Legend for algorithms
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=ALGO_COLORS[a], label=ALGO_LABELS[a])
        for a in ALGO_COLORS
    ]
    ax2.legend(handles=legend_elements, loc="lower right", fontsize=9)

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ── Plot 4: Competitive pressure ──────────────────────────────────────────────

def plot_competitive_pressure(df, out_path):
    """Early vs late revenue showing competitive pressure over training."""
    agents = df["agent"].unique()
    early  = df[df["episode"] < 50].groupby("agent")["revenue"].mean()
    late   = df[df["episode"] >= df["episode"].max() - 49].groupby("agent")["revenue"].mean()
    pct    = ((late - early) / early.clip(lower=1) * 100).sort_values()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(
        "Competitive Pressure — How Revenue Changed Over Training",
        fontweight="bold", color="#e0e0f0"
    )

    # Left: early vs late grouped bars
    ax   = axes[0]
    x    = np.arange(len(agents))
    names = list(early.index)
    width = 0.38

    ax.bar(x - width/2, [early.get(n, 0) for n in names],
           width, label="Episodes 1–50", color="#5a6a7a", alpha=0.85)
    ax.bar(x + width/2, [late.get(n, 0) for n in names],
           width, label="Episodes 451–500", color="#7eb8d4", alpha=0.85)

    ax.set_title("Revenue: Early vs Late Training")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=35, ha="right")
    ax.set_ylabel("Avg Revenue ($)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(fmt_millions))
    ax.grid(True, axis="y", alpha=0.4)
    ax.legend()

    # Right: % change bar chart
    ax2 = axes[1]
    bar_colors = ["#7ec4a0" if v >= 0 else "#c4907e" for v in pct.values]
    bars = ax2.barh(pct.index, pct.values, color=bar_colors, alpha=0.85)
    ax2.axvline(x=0, color="#3a3a50", linewidth=1)
    ax2.set_title("% Revenue Change: Early → Late\n(9 of 10 agents declined — competitive pressure)")
    ax2.set_xlabel("% Change in Revenue")
    ax2.grid(True, axis="x", alpha=0.4)

    for bar, val in zip(bars, pct.values):
        x_pos = val + (1 if val >= 0 else -1)
        ax2.text(x_pos, bar.get_y() + bar.get_height()/2,
                 f"{val:+.0f}%", va="center", fontsize=9,
                 color="#c8c8e0")

    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  Saved: {out_path}")


# ── Summary table ─────────────────────────────────────────────────────────────

def build_summary(df, out_path):
    n_ep   = df["episode"].max()
    final  = df[df["episode"] >= n_ep - 49]
    early  = df[df["episode"] < 50]

    rows = []
    for name in df["agent"].unique():
        f = final[final["agent"] == name]
        e = early[early["agent"] == name]
        pct = ((f["revenue"].mean() - e["revenue"].mean())
               / max(e["revenue"].mean(), 1) * 100)
        rows.append({
            "agent":          name,
            "algorithm":      f["algorithm"].iloc[0],
            "reward_fn":      f["reward_fn"].iloc[0],
            "avg_revenue":    f"${f['revenue'].mean():,.0f}",
            "avg_mkt_share":  f"{f['market_share'].mean():.4f}",
            "avg_reward":     f"{f['total_reward'].mean():,.0f}",
            "revenue_change": f"{pct:+.1f}%",
        })

    summary = pd.DataFrame(rows).sort_values("avg_revenue", ascending=False)
    summary.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}")
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Generate ablation plots from existing training_results.csv"
    )
    p.add_argument("--data", default="results/training_results.csv",
                   help="Path to training_results.csv")
    p.add_argument("--out",  default="results/ablations",
                   help="Output directory for plots")
    args = p.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n  Loading {args.data}...")
    df = pd.read_csv(args.data)
    print(f"  {len(df)} rows | {df['episode'].max()+1} episodes | "
          f"{df['agent'].nunique()} agents\n")

    print("  Generating plots...")
    plot_algo_convergence(df,         out / "algo_convergence.png")
    plot_algo_final_comparison(df,    out / "algo_comparison.png")
    plot_reward_divergence(df,        out / "reward_divergence.png")
    plot_competitive_pressure(df,     out / "competitive_pressure.png")
    build_summary(df,                 out / "ablation_summary.csv")

    print(f"""
{'═'*55}
  All outputs in: {out}/

  algo_convergence.png     — revenue curves per algorithm
  algo_comparison.png      — final revenue + policy stability
  reward_divergence.png    — reward fn drives strategy
  competitive_pressure.png — early vs late revenue change
  ablation_summary.csv     — full stats table
{'═'*55}
""")


if __name__ == "__main__":
    main()
