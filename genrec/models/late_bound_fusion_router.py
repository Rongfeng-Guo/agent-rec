from __future__ import annotations

import torch
from torch import nn


class LateBoundFusionRouter(nn.Module):
    """Lightweight scalar gate over multiple query sources plus route confidence."""

    def __init__(
        self,
        num_features: int,
        num_sources: int,
        hidden_dim: int = 32,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.num_features = int(num_features)
        self.num_sources = int(num_sources)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)

        self.feature_norm = nn.LayerNorm(self.num_features)
        self.gate = nn.Sequential(
            nn.Linear(self.num_features, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.num_sources + 1),
        )

    def gate_weights(self, sample_features: torch.Tensor) -> torch.Tensor:
        logits = self.gate(self.feature_norm(sample_features))
        return torch.softmax(logits, dim=-1)

    def forward(
        self,
        sample_features: torch.Tensor,
        source_scores: torch.Tensor,
        route_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if source_scores.ndim != 3:
            raise ValueError(f"Expected source_scores to be [batch, candidates, sources], got {tuple(source_scores.shape)!r}.")
        if route_scores.ndim != 2:
            raise ValueError(f"Expected route_scores to be [batch, candidates], got {tuple(route_scores.shape)!r}.")
        weights = self.gate_weights(sample_features)
        source_weights = weights[:, : self.num_sources].unsqueeze(1)
        route_weight = weights[:, self.num_sources].unsqueeze(-1)
        logits = (source_scores * source_weights).sum(dim=-1) + (route_scores * route_weight)
        return logits, weights
