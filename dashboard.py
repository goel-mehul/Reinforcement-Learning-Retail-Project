"""
RetailRL — Interactive Training Dashboard
==========================================
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from pathlib import Path
import pickle
import json

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="RetailRL — Competitive Pricing Simulation",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600&family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Serif:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

.stApp { background: #13131a; color: #d4d4e0; }

[data-testid="stSidebar"] {
    background: #0e0e14 !important;
    border-right: 1px solid #2a2a38 !important;
}
[data-testid="stSidebar"] * { color: #d4d4e0 !important; }

.block-container { padding-top: 2rem; padding-bottom: 4rem; }

/* Hero */
.hero-eyebrow {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #6b6b88;
    margin-bottom: 10px;
}
.hero-title {
    font-family: 'IBM Plex Serif', serif;
    font-size: 2.6rem;
    font-weight: 600;
    color: #e8e8f0;
    letter-spacing: -0.5px;
    line-height: 1.1;
    margin-bottom: 10px;
}
.hero-desc {
    font-size: 0.95rem;
    color: #8a8aa0;
    line-height: 1.75;
    max-width: 680px;
}

/* Section headers */
.section-title {
    font-family: 'IBM Plex Serif', serif;
    font-size: 1.15rem;
    font-weight: 600;
    color: #e0e0f0;
    margin: 36px 0 16px 0;
    padding-bottom: 10px;
    border-bottom: 1px solid #2a2a38;
}

/* Metric cards */
.metric-row { display: flex; gap: 12px; margin-bottom: 24px; flex-wrap: wrap; }
.metric-card {
    flex: 1; min-width: 120px;
    background: #1a1a24;
    border: 1px solid #2a2a38;
    border-radius: 5px;
    padding: 16px 20px;
}
.metric-value {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.5rem;
    font-weight: 500;
    color: #c8c8e0;
    line-height: 1;
    margin-bottom: 6px;
}
.metric-label {
    font-size: 0.7rem;
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5a5a78;
}

/* Sidebar */
.sidebar-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.62rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a4a60;
    margin: 20px 0 10px 0;
}
.agent-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 6px 0;
    border-bottom: 1px solid #1e1e2c;
}
.agent-name { font-size: 0.85rem; font-weight: 500; }
.agent-algo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    color: #5a5a78;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #2a2a38;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5a5a78;
    padding: 10px 20px;
    background: transparent;
}
.stTabs [aria-selected="true"] {
    color: #a0a8c0 !important;
    border-bottom: 2px solid #a0a8c0 !important;
    background: transparent !important;
}

/* Slider + select */
.stSlider label, .stSelectbox label {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 1.5px !important;
    text-transform: uppercase !important;
    color: #5a5a78 !important;
}

/* Callout box */
.callout {
    background: #1a1a24;
    border: 1px solid #2a2a38;
    border-radius: 5px;
    padding: 14px 18px;
    font-size: 0.88rem;
    color: #8a8aa8;
    line-height: 1.7;
    margin-bottom: 20px;
}

/* Finding cards */
.finding-card {
    background: #1a1a24;
    border: 1px solid #2a2a38;
    border-left-width: 3px;
    border-radius: 5px;
    padding: 16px 18px;
    margin-bottom: 10px;
}
.finding-agent {
    font-family: 'IBM Plex Serif', serif;
    font-size: 1rem;
    font-weight: 600;
    margin-bottom: 3px;
}
.finding-algo {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.63rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: #5a5a78;
    margin-bottom: 9px;
}
.finding-desc {
    font-size: 0.86rem;
    color: #8a8aa8;
    line-height: 1.65;
}
</style>
""", unsafe_allow_html=True)

# ── Color palette — soft, muted, distinguishable ──────────────────────────────

AGENT_COLORS = {
    'walmart':      '#7eb8d4',
    'target':       '#c47eb0',
    'amazon_fresh': '#d4a76a',
    'qfc':          '#7ec4a0',
    'safeway':      '#a07ec4',
    'kroger':       '#c4907e',
    'trader_joes':  '#7eaac4',
    'whole_foods':  '#90c47e',
    'aldi':         '#c4b87e',
    'costco':       '#b8c47e',
}

ALGO_COLORS = {
    'DQNAgent':    '#7eb8d4',
    'PPOAgent':    '#c4907e',
    'A2CAgent':    '#7ec4a0',
    'QTableAgent': '#c4b87e',
}

ALGO_LABELS = {
    'DQNAgent': 'DQN', 'PPOAgent': 'PPO',
    'A2CAgent': 'A2C', 'QTableAgent': 'Q-Table',
}

def plotly_layout(**overrides):
    base = dict(
        paper_bgcolor='#13131a',
        plot_bgcolor='#1a1a24',
        font=dict(family='IBM Plex Sans', color='#a0a0b8', size=12),
        xaxis=dict(gridcolor='#22222e', linecolor='#2a2a38',
                   zerolinecolor='#2a2a38', tickfont=dict(size=11, color='#7a7a98')),
        yaxis=dict(gridcolor='#22222e', linecolor='#2a2a38',
                   zerolinecolor='#2a2a38', tickfont=dict(size=11, color='#7a7a98')),
        legend=dict(bgcolor='#1a1a24', bordercolor='#2a2a38', borderwidth=1,
                    font=dict(size=11, color='#a0a0b8')),
        margin=dict(l=52, r=24, t=44, b=52),
        hoverlabel=dict(bgcolor='#1e1e2c', bordercolor='#2a2a38',
                        font=dict(family='IBM Plex Sans', size=12, color='#d4d4e0')),
        title_font=dict(family='IBM Plex Serif', size=14, color='#c8c8e0'),
    )
    base.update(overrides)
    return base

# ── Data ──────────────────────────────────────────────────────────────────────

@st.cache_data
def load_data():
    base = Path(__file__).parent / "results"
    with open(base / "training_results.pkl", "rb") as f:
        results = pickle.load(f)
    df = pd.read_csv(base / "training_results.csv")
    with open(base / "agent_meta.json") as f:
        agent_meta = json.load(f)
    return results, df, agent_meta

def smooth(series, window=20):
    return pd.Series(series).rolling(window, min_periods=1).mean().values

try:
    results, df, agent_meta = load_data()
    data_ok = True
except FileNotFoundError:
    data_ok = False

AGENT_NAMES = list(agent_meta.keys()) if data_ok else []

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style='padding: 20px 0 24px 0;'>
        <div style='font-family: IBM Plex Serif, serif; font-size: 1.25rem;
                    font-weight: 600; color: #d4d4e0;'>RetailRL</div>
        <div style='font-family: IBM Plex Mono, monospace; font-size: 0.6rem;
                    letter-spacing: 2px; text-transform: uppercase; color: #4a4a60;
                    margin-top: 4px;'>Multi-Agent Pricing Sim</div>
    </div>""", unsafe_allow_html=True)

    if data_ok:
        n_eps = len(results)
        algos = list(set(v['algorithm'] for v in agent_meta.values()))
        st.markdown(f"""
        <div style='display:flex; flex-direction:column; gap:8px; margin-bottom:20px;'>
            <div class='metric-card'>
                <div class='metric-value'>{n_eps}</div>
                <div class='metric-label'>Episodes</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{len(agent_meta)}</div>
                <div class='metric-label'>Agents</div>
            </div>
            <div class='metric-card'>
                <div class='metric-value'>{len(algos)}</div>
                <div class='metric-label'>Algorithms</div>
            </div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<div class='sidebar-label'>Agent Registry</div>", unsafe_allow_html=True)
        for name, meta in agent_meta.items():
            color = AGENT_COLORS.get(name, '#a0a0b8')
            algo  = meta['algorithm'].replace('Agent', '')
            st.markdown(f"""
            <div class='agent-row'>
                <span class='agent-name' style='color:{color};'>{name}</span>
                <span class='agent-algo'>{algo}</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='margin-top:32px; font-family: IBM Plex Mono, monospace; font-size: 0.58rem;
                color: #3a3a50; letter-spacing: 1px; text-transform: uppercase;
                line-height: 1.9;'>
        NYU · Reinforcement Learning<br>Portfolio Project · 2026
    </div>""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div style='padding-bottom: 28px; border-bottom: 1px solid #2a2a38; margin-bottom: 32px;'>
    <div class='hero-eyebrow'>Multi-Agent Reinforcement Learning</div>
    <div class='hero-title'>Retail Pricing Simulation</div>
    <div class='hero-desc'>
        Ten grocery retailers compete through dynamic pricing across 500 training episodes.
        Each store runs a different RL algorithm — DQN, PPO, A2C, or Q-Table —
        optimizing a distinct objective: revenue, market share, profit margin, or prestige.
    </div>
</div>
""", unsafe_allow_html=True)

if not data_ok:
    st.error("Results files not found. Place training_results.pkl, training_results.csv, and agent_meta.json in the results/ directory.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Training Overview",
    "Algorithm Comparison",
    "Strategy Divergence",
    "Episode Replay",
    "Agent Deep-Dive",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Training Overview
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown("<div class='section-title'>Revenue Over 500 Episodes</div>",
                unsafe_allow_html=True)

    w = st.slider("Smoothing window", 5, 50, 20, key="s1")

    col_chart, col_rank = st.columns([3, 1])

    with col_chart:
        fig = go.Figure()
        for name in AGENT_NAMES:
            revs = df[df['agent'] == name]['revenue'].values
            fig.add_trace(go.Scatter(
                x=list(range(len(revs))),
                y=smooth(revs, w),
                name=name,
                line=dict(color=AGENT_COLORS.get(name, '#a0a0b8'), width=1.8),
                hovertemplate=f"<b>{name}</b><br>Ep %{{x}} · $%{{y:,.0f}}<extra></extra>",
            ))
        fig.update_layout(**plotly_layout(
            height=400, hovermode='x unified',
            title_text="Revenue (smoothed)",
            xaxis_title="Episode", yaxis_title="Revenue ($)",
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col_rank:
        st.markdown("<br><br>", unsafe_allow_html=True)
        final = df[df['episode'] >= 450].groupby('agent')['revenue'].mean().sort_values(ascending=False)
        st.markdown("""
        <div style='font-family: IBM Plex Mono, monospace; font-size: 0.62rem;
                    letter-spacing: 2px; text-transform: uppercase; color: #4a4a60;
                    margin-bottom: 12px;'>Final 50 eps avg</div>""", unsafe_allow_html=True)
        for i, (name, rev) in enumerate(final.items()):
            color = AGENT_COLORS.get(name, '#a0a0b8')
            val_color = '#c8c8e0' if i == 0 else '#7a7a98'
            st.markdown(f"""
            <div style='display:flex; justify-content:space-between; align-items:center;
                        padding:6px 0; border-bottom:1px solid #1e1e2c;'>
                <span style='color:{color}; font-size:0.84rem; font-weight:500;'>{name}</span>
                <span style='color:{val_color}; font-family:IBM Plex Mono;
                             font-size:0.76rem;'>${rev/1e6:.2f}M</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div class='section-title'>Market Share Evolution</div>",
                unsafe_allow_html=True)

    fig2 = go.Figure()
    for name in AGENT_NAMES:
        shares = df[df['agent'] == name]['market_share'].values
        fig2.add_trace(go.Scatter(
            x=list(range(len(shares))),
            y=smooth(shares, w),
            name=name,
            line=dict(color=AGENT_COLORS.get(name, '#a0a0b8'), width=1.8),
            hovertemplate=f"<b>{name}</b><br>Ep %{{x}} · %{{y:.3f}}<extra></extra>",
        ))
    for ep, label in [(100, "ep 100"), (300, "ep 300")]:
        fig2.add_vline(x=ep, line_dash="dot", line_color="#2a2a3c", line_width=1)
        fig2.add_annotation(x=ep, y=1.02, text=label, showarrow=False, yref='paper',
                            font=dict(color='#4a4a60', size=10, family='IBM Plex Mono'))
    fig2.update_layout(**plotly_layout(
        height=360, hovermode='x unified',
        title_text="Market Share (smoothed)",
        xaxis_title="Episode", yaxis_title="Market Share",
    ))
    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("<div class='section-title'>Competitive Pressure — Early vs Late</div>",
                unsafe_allow_html=True)

    early_df  = df[df['episode'] < 50]
    late_df   = df[df['episode'] >= 450]
    early_rev = early_df.groupby('agent')['revenue'].mean()
    late_rev  = late_df.groupby('agent')['revenue'].mean()
    pct_chg   = ((late_rev - early_rev) / early_rev.clip(lower=1)) * 100

    col3, col4 = st.columns(2)
    with col3:
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            name='Episodes 1–50', x=AGENT_NAMES,
            y=[early_rev.get(n, 0) for n in AGENT_NAMES],
            marker_color='#5a6a7a', opacity=0.8,
        ))
        fig3.add_trace(go.Bar(
            name='Episodes 451–500', x=AGENT_NAMES,
            y=[late_rev.get(n, 0) for n in AGENT_NAMES],
            marker_color='#7eb8d4', opacity=0.85,
        ))
        fig3.update_layout(**plotly_layout(
            height=320, barmode='group',
            title_text="Revenue: Early vs Late",
            xaxis_tickangle=-40,
        ))
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        changes    = [pct_chg.get(n, 0) for n in AGENT_NAMES]
        bar_colors = ['#7ec4a0' if c >= 0 else '#c4907e' for c in changes]
        fig4 = go.Figure(go.Bar(
            x=AGENT_NAMES, y=changes,
            marker_color=bar_colors, opacity=0.85,
            text=[f"{c:.0f}%" for c in changes],
            textposition='outside',
            textfont=dict(size=10, color='#7a7a98', family='IBM Plex Mono'),
        ))
        fig4.add_hline(y=0, line_color='#2a2a3c', line_width=1)
        fig4.update_layout(**plotly_layout(
            height=320,
            title_text="% Revenue Change, Early → Late",
            xaxis_tickangle=-40, yaxis_title="Change (%)",
        ))
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Algorithm Comparison
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown("<div class='section-title'>Revenue by Algorithm Group</div>",
                unsafe_allow_html=True)

    w2   = st.slider("Smoothing window", 5, 50, 20, key="s2")
    algos = ['DQNAgent', 'PPOAgent', 'A2CAgent', 'QTableAgent']
    cols  = st.columns(2)

    for i, algo in enumerate(algos):
        with cols[i % 2]:
            algo_agents = [n for n, m in agent_meta.items() if m['algorithm'] == algo]
            fig = go.Figure()
            for name in algo_agents:
                revs = df[df['agent'] == name]['revenue'].values
                fig.add_trace(go.Scatter(
                    x=list(range(len(revs))),
                    y=smooth(revs, w2),
                    name=f"{name} · {agent_meta[name]['reward_fn']}",
                    line=dict(color=AGENT_COLORS.get(name, '#a0a0b8'), width=2),
                    hovertemplate=f"<b>{name}</b><br>$%{{y:,.0f}}<extra></extra>",
                ))
            fig.update_layout(**plotly_layout(
                height=280,
                title_text=ALGO_LABELS[algo],
                title_font=dict(family='IBM Plex Serif', size=14,
                                color=ALGO_COLORS[algo]),
                xaxis_title="Episode", yaxis_title="Revenue ($)",
            ))
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("<div class='section-title'>Convergence Analysis</div>",
                unsafe_allow_html=True)

    col5, col6 = st.columns(2)
    with col5:
        fig_c = go.Figure()
        for algo in algos:
            agents_in = [n for n, m in agent_meta.items() if m['algorithm'] == algo]
            all_r     = np.array([df[df['agent'] == n]['total_reward'].values for n in agents_in])
            mean_r    = all_r.mean(axis=0)
            normed    = (mean_r - mean_r.min()) / (mean_r.max() - mean_r.min() + 1e-8)
            fig_c.add_trace(go.Scatter(
                x=list(range(len(normed))), y=smooth(normed, 20),
                name=ALGO_LABELS[algo],
                line=dict(color=ALGO_COLORS[algo], width=2),
            ))
        fig_c.update_layout(**plotly_layout(
            height=320,
            title_text="Convergence Speed — Normalized Reward",
            xaxis_title="Episode", yaxis_title="Normalized Reward (0–1)",
        ))
        st.plotly_chart(fig_c, use_container_width=True)

    with col6:
        fig_s = go.Figure()
        for algo in algos:
            agents_in = [n for n, m in agent_meta.items() if m['algorithm'] == algo]
            all_r     = np.array([df[df['agent'] == n]['total_reward'].values for n in agents_in])
            mean_r    = all_r.mean(axis=0)
            roll_std  = pd.Series(mean_r).rolling(50, min_periods=50).std().values
            fig_s.add_trace(go.Scatter(
                x=list(range(len(roll_std))), y=roll_std,
                name=ALGO_LABELS[algo],
                line=dict(color=ALGO_COLORS[algo], width=2),
            ))
        fig_s.update_layout(**plotly_layout(
            height=320,
            title_text="Policy Stability — Rolling Std (window=50)",
            xaxis_title="Episode", yaxis_title="Rolling Std",
        ))
        st.plotly_chart(fig_s, use_container_width=True)

    st.markdown("<div class='section-title'>Algorithm Summary — Final 50 Episodes</div>",
                unsafe_allow_html=True)
    final_df = df[df['episode'] >= 450]
    rows = []
    for algo in algos:
        a   = [n for n, m in agent_meta.items() if m['algorithm'] == algo]
        adf = final_df[final_df['agent'].isin(a)]
        rows.append({
            'Algorithm':     ALGO_LABELS[algo],
            'Agents':        ', '.join(a),
            'Avg Revenue':   f"${adf['revenue'].mean():,.0f}",
            'Avg Mkt Share': f"{adf['market_share'].mean():.4f}",
            'Avg Reward':    f"{adf['total_reward'].mean():,.0f}",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Strategy Divergence
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("<div class='section-title'>Reward vs Revenue — Strategy Divergence</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='callout'>
        Different reward functions produce fundamentally different strategies.
        An agent with high reward but low revenue isn't failing —
        it's optimizing exactly what it was designed to optimize.
        This is the central finding of the simulation.
    </div>""", unsafe_allow_html=True)

    final_df = df[df['episode'] >= 450]
    summary  = final_df.groupby('agent')[['total_reward', 'revenue', 'market_share']].mean()

    col7, col8 = st.columns(2)
    with col7:
        fig_d1 = go.Figure()
        for name in AGENT_NAMES:
            x = summary.loc[name, 'revenue']
            y = summary.loc[name, 'total_reward']
            fig_d1.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text', name=name,
                marker=dict(color=AGENT_COLORS.get(name, '#a0a0b8'), size=13,
                            line=dict(color='#13131a', width=2)),
                text=[f"  {name}"],
                textposition='middle right',
                textfont=dict(size=10, family='IBM Plex Sans', color='#8a8aaa'),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Algorithm: {ALGO_LABELS.get(agent_meta[name]['algorithm'], '')}<br>"
                    f"Reward fn: {agent_meta[name]['reward_fn']}<br>"
                    f"Revenue: $%{{x:,.0f}}<br>"
                    f"Reward: %{{y:,.0f}}<extra></extra>"
                ),
            ))
        fig_d1.update_layout(**plotly_layout(
            height=400, showlegend=False,
            title_text="High Reward ≠ High Revenue",
            xaxis_title="Avg Revenue — last 50 episodes ($)",
            yaxis_title="Avg Total Reward — last 50 episodes",
        ))
        st.plotly_chart(fig_d1, use_container_width=True)

    with col8:
        fig_d2 = go.Figure()
        for name in AGENT_NAMES:
            x = summary.loc[name, 'revenue']
            y = summary.loc[name, 'market_share']
            fig_d2.add_trace(go.Scatter(
                x=[x], y=[y], mode='markers+text', name=name,
                marker=dict(color=AGENT_COLORS.get(name, '#a0a0b8'), size=13,
                            line=dict(color='#13131a', width=2)),
                text=[f"  {name}"],
                textposition='middle right',
                textfont=dict(size=10, family='IBM Plex Sans', color='#8a8aaa'),
                hovertemplate=(
                    f"<b>{name}</b><br>"
                    f"Revenue: $%{{x:,.0f}}<br>"
                    f"Market Share: %{{y:.4f}}<extra></extra>"
                ),
            ))
        fig_d2.update_layout(**plotly_layout(
            height=400, showlegend=False,
            title_text="Market Share vs Revenue",
            xaxis_title="Avg Revenue — last 50 episodes ($)",
            yaxis_title="Avg Market Share — last 50 episodes",
        ))
        st.plotly_chart(fig_d2, use_container_width=True)

    st.markdown("<div class='section-title'>Key Findings</div>", unsafe_allow_html=True)

    findings = [
        ("walmart",     "DQN · pure_revenue",
         "Consistent revenue leader across 500 episodes. DQN's experience replay enabled stable value learning, but early dominance eroded by 46% as competitors adapted their pricing strategies."),
        ("costco",      "Q-Table · bulk_volume",
         "Highest total reward despite only mid-tier revenue. The bulk volume objective discovered a resilient market niche — pricing for volume over margin — that proved difficult for competitors to undercut."),
        ("whole_foods", "PPO · prestige_reward",
         "Reward of 41M but revenue of only $300k. The prestige reward penalizes pricing below market average, so the agent learned to intentionally price high and sacrifice volume to protect brand positioning."),
        ("amazon_fresh","A2C · market_share",
         "Lowest revenue but optimizing a completely different objective. The market share reward incentivizes customer capture regardless of margin — a deliberate loss-leader strategy to crowd out competitors."),
    ]

    col_f1, col_f2 = st.columns(2)
    for i, (name, algo_rw, desc) in enumerate(findings):
        color = AGENT_COLORS.get(name, '#a0a0b8')
        with (col_f1 if i % 2 == 0 else col_f2):
            st.markdown(f"""
            <div class='finding-card' style='border-left-color:{color};'>
                <div class='finding-agent' style='color:{color};'>{name}</div>
                <div class='finding-algo'>{algo_rw}</div>
                <div class='finding-desc'>{desc}</div>
            </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Episode Replay
# ══════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("<div class='section-title'>Episode Replay — Day-by-Day Timeline</div>",
                unsafe_allow_html=True)

    st.markdown("""
    <div class='callout'>
        Scrub through any of the 500 training episodes to see how market share
        and revenue evolved day-by-day across the 365-day simulation.
        Episode 0 reflects early exploratory behavior; episode 499 shows trained policies.
    </div>""", unsafe_allow_html=True)

    ep_sel = st.slider("Episode", 0, len(results) - 1, len(results) - 1, key="ep_sel")
    sel    = results[ep_sel]

    top_rev   = max(sel.revenues.items(), key=lambda x: x[1])
    top_share = max(sel.market_shares.items(), key=lambda x: x[1])
    total_rev = sum(sel.revenues.values())

    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card'>
            <div class='metric-value'>{ep_sel}</div>
            <div class='metric-label'>Episode</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>${total_rev/1e6:.1f}M</div>
            <div class='metric-label'>Total Market Revenue</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'
                 style='color:{AGENT_COLORS.get(top_rev[0],"#b0b0c8")};'>{top_rev[0]}</div>
            <div class='metric-label'>Revenue Leader · ${top_rev[1]/1e6:.1f}M</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'
                 style='color:{AGENT_COLORS.get(top_share[0],"#b0b0c8")};'>{top_share[0]}</div>
            <div class='metric-label'>Share Leader · {top_share[1]:.3f}</div>
        </div>
    </div>""", unsafe_allow_html=True)

    metrics    = (sel.episode_metrics
                  if hasattr(sel, 'episode_metrics') and sel.episode_metrics else None)
    id_to_name = {i: n for i, n in enumerate(AGENT_NAMES)}

    if metrics and len(metrics) > 0:
        days      = [m['day'] for m in metrics]
        agent_ids = list(metrics[0]['market_shares'].keys())

        col9, col10 = st.columns(2)
        with col9:
            fig_r1 = go.Figure()
            for aid in agent_ids:
                name   = id_to_name.get(aid, str(aid))
                shares = [m['market_shares'].get(aid, 0) for m in metrics]
                fig_r1.add_trace(go.Scatter(
                    x=days, y=smooth(shares, 14), name=name,
                    line=dict(color=AGENT_COLORS.get(name, '#a0a0b8'), width=1.8),
                    hovertemplate=f"<b>{name}</b><br>Day %{{x}} · %{{y:.3f}}<extra></extra>",
                ))
            fig_r1.update_layout(**plotly_layout(
                height=340, hovermode='x unified',
                title_text=f"Market Share — Episode {ep_sel} (14-day avg)",
                xaxis_title="Day of Year", yaxis_title="Market Share",
            ))
            st.plotly_chart(fig_r1, use_container_width=True)

        with col10:
            fig_r2 = go.Figure()
            for aid in agent_ids:
                name    = id_to_name.get(aid, str(aid))
                revs    = [m['revenues'].get(aid, 0) for m in metrics]
                fig_r2.add_trace(go.Scatter(
                    x=days, y=np.cumsum(revs), name=name,
                    line=dict(color=AGENT_COLORS.get(name, '#a0a0b8'), width=1.8),
                    hovertemplate=f"<b>{name}</b><br>Day %{{x}} · $%{{y:,.0f}}<extra></extra>",
                ))
            fig_r2.update_layout(**plotly_layout(
                height=340, hovermode='x unified',
                title_text=f"Cumulative Revenue — Episode {ep_sel}",
                xaxis_title="Day of Year", yaxis_title="Cumulative Revenue ($)",
            ))
            st.plotly_chart(fig_r2, use_container_width=True)

        total_demand = [m['total_demand'] for m in metrics]
        fig_dem = go.Figure()
        fig_dem.add_trace(go.Scatter(
            x=days, y=total_demand, name='Daily',
            line=dict(color='#2a2a3c', width=1),
            fill='tozeroy', fillcolor='rgba(160,168,192,0.05)',
        ))
        fig_dem.add_trace(go.Scatter(
            x=days, y=smooth(total_demand, 14), name='14-day avg',
            line=dict(color='#a0a8c0', width=2),
        ))
        fig_dem.update_layout(**plotly_layout(
            height=260,
            title_text="Daily Market Demand",
            xaxis_title="Day of Year", yaxis_title="Units Sold",
        ))
        st.plotly_chart(fig_dem, use_container_width=True)

    else:
        fig_bar = go.Figure(go.Bar(
            x=list(sel.revenues.keys()),
            y=list(sel.revenues.values()),
            marker_color=[AGENT_COLORS.get(n, '#a0a0b8') for n in sel.revenues],
            opacity=0.85,
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        ))
        fig_bar.update_layout(**plotly_layout(
            height=340,
            title_text=f"Revenue by Agent — Episode {ep_sel}",
            xaxis_tickangle=-40,
        ))
        st.plotly_chart(fig_bar, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Agent Deep-Dive
# ══════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("<div class='section-title'>Agent Deep-Dive</div>", unsafe_allow_html=True)

    sel_agent = st.selectbox(
        "Select agent",
        options=AGENT_NAMES,
        format_func=lambda x: (
            f"{x}  ·  {ALGO_LABELS.get(agent_meta[x]['algorithm'], agent_meta[x]['algorithm'])}"
            f"  ·  {agent_meta[x]['reward_fn']}"
        ),
    )

    meta  = agent_meta[sel_agent]
    color = AGENT_COLORS.get(sel_agent, '#a0a0b8')
    adf   = df[df['agent'] == sel_agent]

    st.markdown(f"""
    <div style='background:#1a1a24; border:1px solid #2a2a38; border-left:3px solid {color};
                border-radius:5px; padding:18px 22px; margin:12px 0 20px 0;'>
        <div style='font-family: IBM Plex Serif, serif; font-size:1.4rem; font-weight:600;
                    color:{color}; margin-bottom:5px;'>{sel_agent}</div>
        <div style='font-family: IBM Plex Mono, monospace; font-size:0.65rem;
                    letter-spacing:2px; text-transform:uppercase; color:#5a5a78;'>
            {ALGO_LABELS.get(meta["algorithm"], meta["algorithm"])}
            &nbsp;·&nbsp; {meta["reward_fn"]}
        </div>
    </div>""", unsafe_allow_html=True)

    final_a   = adf[adf['episode'] >= 450]
    early_a   = adf[adf['episode'] < 50]
    rev_chg   = ((final_a['revenue'].mean() - early_a['revenue'].mean())
                 / max(early_a['revenue'].mean(), 1) * 100)

    st.markdown(f"""
    <div class='metric-row'>
        <div class='metric-card'>
            <div class='metric-value'>${final_a["revenue"].mean()/1e6:.2f}M</div>
            <div class='metric-label'>Avg Revenue (final 50)</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>{final_a["market_share"].mean():.4f}</div>
            <div class='metric-label'>Avg Market Share</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>{final_a["total_reward"].mean():,.0f}</div>
            <div class='metric-label'>Avg Reward</div>
        </div>
        <div class='metric-card'>
            <div class='metric-value'>{rev_chg:+.1f}%</div>
            <div class='metric-label'>Revenue Change Early→Late</div>
        </div>
    </div>""", unsafe_allow_html=True)

    w3 = st.slider("Smoothing", 5, 50, 20, key="s3")

    col11, col12 = st.columns(2)
    with col11:
        fig_a1 = go.Figure()
        r, g, b = int(color[1:3], 16), int(color[3:5], 16), int(color[5:7], 16)
        fig_a1.add_trace(go.Scatter(
            x=adf['episode'], y=smooth(adf['revenue'].values, w3),
            line=dict(color=color, width=2),
            fill='tozeroy', fillcolor=f'rgba({r},{g},{b},0.06)',
            hovertemplate="Ep %{x}<br>$%{y:,.0f}<extra></extra>",
        ))
        fig_a1.update_layout(**plotly_layout(
            height=300, title_text="Revenue Over Training",
            xaxis_title="Episode", yaxis_title="Revenue ($)",
        ))
        st.plotly_chart(fig_a1, use_container_width=True)

    with col12:
        fig_a2 = go.Figure()
        fig_a2.add_trace(go.Scatter(
            x=adf['episode'], y=smooth(adf['market_share'].values, w3),
            line=dict(color=color, width=2),
            hovertemplate="Ep %{x}<br>%{y:.4f}<extra></extra>",
        ))
        fig_a2.update_layout(**plotly_layout(
            height=300, title_text="Market Share Over Training",
            xaxis_title="Episode", yaxis_title="Market Share",
        ))
        st.plotly_chart(fig_a2, use_container_width=True)

    fig_hist = go.Figure(go.Histogram(
        x=adf['total_reward'], nbinsx=40,
        marker_color=color, opacity=0.7,
    ))
    fig_hist.update_layout(**plotly_layout(
        height=260, title_text="Reward Distribution — All 500 Episodes",
        xaxis_title="Total Episode Reward", yaxis_title="Count",
    ))
    st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("<div class='section-title'>vs. The Field — Final 50 Episodes</div>",
                unsafe_allow_html=True)

    all_sum = df[df['episode'] >= 450].groupby('agent')[['revenue', 'market_share']].mean()
    col13, col14 = st.columns(2)

    with col13:
        fig_v1 = go.Figure(go.Bar(
            x=AGENT_NAMES,
            y=[all_sum.loc[n, 'revenue'] for n in AGENT_NAMES],
            marker_color=[color if n == sel_agent else '#252530' for n in AGENT_NAMES],
            opacity=0.9,
            hovertemplate="%{x}<br>$%{y:,.0f}<extra></extra>",
        ))
        fig_v1.update_layout(**plotly_layout(
            height=300, title_text="Revenue vs All Agents",
            xaxis_tickangle=-40,
        ))
        st.plotly_chart(fig_v1, use_container_width=True)

    with col14:
        fig_v2 = go.Figure(go.Bar(
            x=AGENT_NAMES,
            y=[all_sum.loc[n, 'market_share'] for n in AGENT_NAMES],
            marker_color=[color if n == sel_agent else '#252530' for n in AGENT_NAMES],
            opacity=0.9,
            hovertemplate="%{x}<br>%{y:.4f}<extra></extra>",
        ))
        fig_v2.update_layout(**plotly_layout(
            height=300, title_text="Market Share vs All Agents",
            xaxis_tickangle=-40,
        ))
        st.plotly_chart(fig_v2, use_container_width=True)