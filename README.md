# RetailRL 🛒

> Multi-agent reinforcement learning simulation of competitive retail pricing dynamics

10 heterogeneous RL agents (DQN, PPO, A2C, Q-Table) representing major US retailers compete
in a shared market environment — dynamically pricing 50 products across inventory constraints,
supply chain delays, and promotional cycles.

## Project Status
🚧 Active Development — Phase 1: Foundation & Setup

## Agents
| Retailer | Algorithm | Reward Function |
|---|---|---|
| Walmart | DQN | Pure Revenue |
| Target | DQN | Profit Margin |
| Amazon Fresh | DQN | Market Share |
| QFC | PPO | Revenue + Inventory |
| Safeway | PPO | Long-term Customer Value |
| Kroger | PPO | Promo-Aware Profit |
| Trader Joe's | A2C | Premium Floor |
| Whole Foods | A2C | Prestige Reward |
| Aldi | Q-Table | Discount Maximization |
| Costco | Q-Table | Bulk Volume |

## Stack
- **RL:** stable-baselines3, pettingzoo, gymnasium, PyTorch
- **Simulation:** numpy, pandas, scipy
- **Visualization:** Streamlit, plotly, WandB
- **Dev:** pytest, black, ruff, GitHub Actions

## Setup
```bash
conda create -n retailrl python=3.11 -y
conda activate retailrl
pip install -e ".[dev]"
```

## Project Structure
```
retail-rl/
├── env/          # PettingZoo multi-agent environment
├── agents/       # DQN, PPO, A2C, Q-Table implementations
├── utils/        # Shared utilities
├── data/         # Raw and processed data
├── config/       # YAML configuration files
├── notebooks/    # EDA and analysis
├── tests/        # pytest suite
└── docs/         # Architecture diagrams, writeup
```

## Roadmap
- [ ] Phase 1: Foundation & Setup
- [ ] Phase 2: Core Environment Engine
- [ ] Phase 3: Baseline Agents & Reward Design
- [ ] Phase 4: RL Agent Implementation
- [ ] Phase 5: Multi-Agent Training Pipeline
- [ ] Phase 6: Visualization Dashboard
- [ ] Phase 7: Polish & Portfolio Finish