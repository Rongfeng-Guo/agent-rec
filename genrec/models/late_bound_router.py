from __future__ import annotations

from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class LateBoundRouter(nn.Module):
    """Lightweight hierarchical router for late-bound memory retrieval."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim: int,
        num_prefix1: int,
        num_prefix2: int,
        dropout: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_prefix1 = int(num_prefix1)
        self.num_prefix2 = int(num_prefix2)
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
        self.route1_head = nn.Linear(self.hidden_dim, self.num_prefix1)
        self.route2_head = nn.Linear(self.hidden_dim, self.num_prefix1 * self.num_prefix2)
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
        pooled = (history_embs * recency.unsqueeze(-1)).sum(dim=1) / denom
        return pooled

    def forward(self, history_embs: torch.Tensor, history_mask: torch.Tensor) -> dict:
        pooled = self.pool_history(history_embs, history_mask)
        hidden = self.encoder(self.input_norm(pooled))
        route1_logits = self.route1_head(hidden)
        route2_logits = self.route2_head(hidden).view(-1, self.num_prefix1, self.num_prefix2)
        query_embedding = F.normalize(self.query_head(hidden), dim=-1)
        return {
            "pooled_history": pooled,
            "hidden": hidden,
            "route1_logits": route1_logits,
            "route2_logits": route2_logits,
            "query_embedding": query_embedding,
        }

    def route_log_probs(self, outputs: dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        route1_log_probs = F.log_softmax(outputs["route1_logits"], dim=-1)
        route2_cond_log_probs = F.log_softmax(outputs["route2_logits"], dim=-1)
        joint_log_probs = route1_log_probs.unsqueeze(-1) + route2_cond_log_probs
        return route1_log_probs, route2_cond_log_probs, joint_log_probs

    def contrastive_logits(self, query_embedding: torch.Tensor, target_embedding: torch.Tensor) -> torch.Tensor:
        query_embedding = F.normalize(query_embedding, dim=-1)
        target_embedding = F.normalize(target_embedding, dim=-1)
        return query_embedding @ target_embedding.T / self.temperature
