import torch
import torch.nn as nn


# Observation layout constants (must match retail_env._get_observation)
OWN_PRICES_START  = 0
OWN_PRICES_END    = 50
COMP_PRICES_START = 50
COMP_PRICES_END   = 500   # 9 competitors × 50 products
N_COMPETITORS     = 9
N_PRODUCTS        = 50


class OpponentEncoder(nn.Module):
    """
    Encodes competitor price information into a compact embedding.

    Takes the 9×50 competitor price block from the observation and
    processes each competitor's prices through a shared encoder, then
    aggregates into a single embedding vector.

    This gives the policy a dedicated representation of "what competitors
    are doing" rather than relying on the main backbone to figure it out
    from 450 raw numbers mixed in with everything else.

    Architecture:
        Input:  (batch, 9, 50)  — each competitor's 50 normalized prices
        Per-competitor encoder: 50 → 64 → 32
        Aggregation: mean over 9 competitors → (batch, 32)
        Output encoder: 32 → 64  → (batch, 64)

    The mean aggregation is permutation-invariant (order of competitors
    doesn't matter) and naturally handles the fact that competitor
    identity is encoded in the agent one-hot, not here.
    """

    EMBEDDING_DIM = 64

    def __init__(self):
        super().__init__()

        # shared encoder applied independently to each competitor
        self.competitor_encoder = nn.Sequential(
            nn.Linear(N_PRODUCTS, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # final projection after aggregation
        self.output_encoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: full observation tensor, shape (batch, 716)

        Returns:
            embedding: shape (batch, 64)
        """
        # extract competitor price block → (batch, 450)
        comp_block = obs[:, COMP_PRICES_START:COMP_PRICES_END]

        # reshape to (batch, 9, 50) — one row per competitor
        batch_size = obs.shape[0]
        comp_prices = comp_block.view(batch_size, N_COMPETITORS, N_PRODUCTS)

        # encode each competitor independently
        # reshape to (batch*9, 50) for batched linear, then back
        flat = comp_prices.reshape(batch_size * N_COMPETITORS, N_PRODUCTS)
        encoded = self.competitor_encoder(flat)                   # (batch*9, 32)
        encoded = encoded.view(batch_size, N_COMPETITORS, 32)     # (batch, 9, 32)

        # aggregate across competitors (permutation-invariant)
        aggregated = encoded.mean(dim=1)                          # (batch, 32)

        # final projection
        embedding = self.output_encoder(aggregated)               # (batch, 64)
        return embedding