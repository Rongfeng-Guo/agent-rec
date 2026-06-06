from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from genrec.models.late_bound_router import LateBoundRouter
from genrec.training.router_dataset import RouteVocab, RouterDataset


@dataclass
class TrainerConfig:
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    lambda_contrastive: float = 0.2
    device: str = "cpu"
    seed: int = 42
    training_objective: str = "full"


def evaluate_route_predictions(model: LateBoundRouter, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    route1_top1 = []
    route1_top2 = []
    route1_top4 = []
    route2_top1 = []
    route2_top4 = []
    route2_top8 = []
    with torch.no_grad():
        for batch in loader:
            history_embs = batch["history_embs"].to(device)
            history_mask = batch["history_mask"].to(device)
            route1_idx = batch["route1_idx"].to(device)
            route2_idx = batch["route2_idx"].to(device)
            outputs = model(history_embs, history_mask)
            route1_log_probs, route2_cond_log_probs, joint_log_probs = model.route_log_probs(outputs)
            route1_top1.append((route1_log_probs.argmax(dim=-1) == route1_idx).float().mean().item())
            route1_top2_indices = torch.topk(route1_log_probs, k=min(2, route1_log_probs.shape[-1]), dim=-1).indices
            route1_top4_indices = torch.topk(route1_log_probs, k=min(4, route1_log_probs.shape[-1]), dim=-1).indices
            route1_top2.append((route1_top2_indices == route1_idx.unsqueeze(-1)).any(dim=-1).float().mean().item())
            route1_top4.append((route1_top4_indices == route1_idx.unsqueeze(-1)).any(dim=-1).float().mean().item())
            cond_target = route2_cond_log_probs[torch.arange(len(route1_idx), device=device), route1_idx]
            route2_top1.append((cond_target.argmax(dim=-1) == route2_idx).float().mean().item())
            flat = joint_log_probs.view(len(route1_idx), -1)
            target_flat = route1_idx * joint_log_probs.shape[-1] + route2_idx
            top4 = torch.topk(flat, k=min(4, flat.shape[-1]), dim=-1).indices
            top8 = torch.topk(flat, k=min(8, flat.shape[-1]), dim=-1).indices
            route2_top4.append((top4 == target_flat.unsqueeze(-1)).any(dim=-1).float().mean().item())
            route2_top8.append((top8 == target_flat.unsqueeze(-1)).any(dim=-1).float().mean().item())
    return {
        "prefix1_accuracy": float(np.mean(route1_top1)) if route1_top1 else 0.0,
        "prefix1_top2_accuracy": float(np.mean(route1_top2)) if route1_top2 else 0.0,
        "prefix1_top4_accuracy": float(np.mean(route1_top4)) if route1_top4 else 0.0,
        "prefix2_top1_accuracy": float(np.mean(route2_top1)) if route2_top1 else 0.0,
        "prefix2_top4_accuracy": float(np.mean(route2_top4)) if route2_top4 else 0.0,
        "prefix2_top8_accuracy": float(np.mean(route2_top8)) if route2_top8 else 0.0,
    }


class RouterTrainer:
    def __init__(self, model: LateBoundRouter, config: TrainerConfig) -> None:
        if config.training_objective not in {"full", "prefix1"}:
            raise ValueError(f"Unsupported training objective: {config.training_objective!r}")
        self.model = model
        self.config = config
        self.device = config.device
        self.model.to(self.device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    def _compute_losses(self, batch: Mapping[str, Any]) -> Dict[str, torch.Tensor]:
        history_embs = batch["history_embs"].to(self.device)
        history_mask = batch["history_mask"].to(self.device)
        target_embedding = F.normalize(batch["target_embedding"].to(self.device), dim=-1)
        route1_idx = batch["route1_idx"].to(self.device)
        route2_idx = batch["route2_idx"].to(self.device)

        outputs = self.model(history_embs, history_mask)
        route1_loss = F.cross_entropy(outputs["route1_logits"], route1_idx)
        route2_loss = route1_loss.new_zeros(())
        contrastive_loss = route1_loss.new_zeros(())
        if self.config.training_objective == "full":
            route2_logits = outputs["route2_logits"][torch.arange(len(route1_idx), device=self.device), route1_idx]
            route2_loss = F.cross_entropy(route2_logits, route2_idx)
            contrastive_logits = self.model.contrastive_logits(outputs["query_embedding"], target_embedding)
            contrastive_targets = torch.arange(len(route1_idx), device=self.device)
            contrastive_loss = F.cross_entropy(contrastive_logits, contrastive_targets)
        total = route1_loss + route2_loss + self.config.lambda_contrastive * contrastive_loss
        return {
            "loss": total,
            "route1_loss": route1_loss.detach(),
            "route2_loss": route2_loss.detach(),
            "contrastive_loss": contrastive_loss.detach(),
        }

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None) -> Dict[str, Any]:
        history = []
        best_metric_name = "val_prefix1_accuracy" if self.config.training_objective == "prefix1" else "val_prefix2_top8"
        best = {"epoch": -1, best_metric_name: -1.0}
        for epoch in range(1, self.config.epochs + 1):
            self.model.train()
            losses = []
            for batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                loss_dict = self._compute_losses(batch)
                loss_dict["loss"].backward()
                self.optimizer.step()
                losses.append({key: float(value.item()) for key, value in loss_dict.items()})
            summary = {
                "epoch": epoch,
                "train_loss": float(np.mean([row["loss"] for row in losses])) if losses else 0.0,
                "train_route1_loss": float(np.mean([row["route1_loss"] for row in losses])) if losses else 0.0,
                "train_route2_loss": float(np.mean([row["route2_loss"] for row in losses])) if losses else 0.0,
                "train_contrastive_loss": float(np.mean([row["contrastive_loss"] for row in losses])) if losses else 0.0,
            }
            if val_loader is not None:
                metrics = evaluate_route_predictions(self.model, val_loader, self.device)
                summary.update({f"val_{key}": value for key, value in metrics.items()})
                current_metric = summary[best_metric_name]
                if current_metric > best[best_metric_name]:
                    best = {"epoch": epoch, best_metric_name: current_metric}
            history.append(summary)
        return {"history": history, "best": best}

    def save(self, output_dir: str | Path, route_vocab: RouteVocab, train_result: Mapping[str, Any], extra_metadata: Mapping[str, Any]) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), output_path / "model.pt")
        payload = {
            "model_config": {
                "embedding_dim": self.model.embedding_dim,
                "hidden_dim": self.model.hidden_dim,
                "num_prefix1": self.model.num_prefix1,
                "num_prefix2": self.model.num_prefix2,
                "temperature": self.model.temperature,
            },
            "route_vocab": route_vocab.to_dict(),
            "trainer_config": asdict(self.config),
            "train_result": train_result,
            "extra_metadata": dict(extra_metadata),
        }
        (output_path / "checkpoint_meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
