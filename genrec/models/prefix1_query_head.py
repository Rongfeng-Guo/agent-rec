from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class Prefix1QueryHead(nn.Module):
    """Query encoder trained for route-bucket item retrieval."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.temperature = float(temperature)
        self.input_norm = nn.LayerNorm(self.embedding_dim)
        self.encoder = nn.Sequential(
            nn.Linear(self.embedding_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.query_head = nn.Linear(self.hidden_dim, self.embedding_dim)

    def pool_history(self, history_embs: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        if history_embs.ndim != 3:
            raise ValueError(f"Expected history_embs to be 3D, got {history_embs.shape!r}")
        if history_mask.ndim != 2:
            raise ValueError(f"Expected history_mask to be 2D, got {history_mask.shape!r}")
        mask = history_mask.float()
        seq_len = history_embs.shape[1]
        positions = torch.arange(seq_len, device=history_embs.device, dtype=history_embs.dtype)
        recency = torch.exp((positions - float(seq_len - 1)) / max(seq_len / 2.0, 1.0))
        recency = recency.unsqueeze(0) * mask
        denom = recency.sum(dim=1, keepdim=True).clamp_min(1e-6)
        return (history_embs * recency.unsqueeze(-1)).sum(dim=1) / denom

    def forward(self, history_embs: torch.Tensor, history_mask: torch.Tensor) -> torch.Tensor:
        pooled = self.pool_history(history_embs, history_mask)
        hidden = self.encoder(self.input_norm(pooled))
        return F.normalize(self.query_head(hidden), dim=-1)
