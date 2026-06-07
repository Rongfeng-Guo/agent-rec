#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import socket
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from genrec.memory.data_adapter import load_item_embeddings
from genrec.models import Prefix1QueryHead
from genrec.training import (
    RouterDataset,
    build_route_vocab,
    build_training_examples,
    default_train_split_name,
    default_validation_split_name,
    filter_training_item_embeddings,
    load_protocol_manifest,
    load_route_mapping,
    protocol_split_examples,
)


@dataclass
class QueryHeadConfig:
    batch_size: int = 256
    epochs: int = 20
    learning_rate: float = 5e-4
    weight_decay: float = 1e-4
    num_hard_negatives: int = 63
    temperature: float = 0.07
    device: str = "cpu"
    seed: int = 42


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


def build_item_tensors(item_embeddings: Mapping[str, np.ndarray], route_mapping: Mapping[str, Sequence[int]], device: str):
    item_ids = [item_id for item_id in item_embeddings if item_id in route_mapping]
    matrix = normalize_matrix(np.stack([np.asarray(item_embeddings[item_id], dtype=np.float32) for item_id in item_ids], axis=0))
    item_to_index = {item_id: idx for idx, item_id in enumerate(item_ids)}
    prefix1_to_indices: dict[int, list[int]] = {}
    for item_id, idx in item_to_index.items():
        prefix1 = int(route_mapping[item_id][0])
        prefix1_to_indices.setdefault(prefix1, []).append(idx)
    item_matrix = torch.from_numpy(matrix).to(device)
    prefix1_tensors = {
        prefix1: torch.tensor(indices, dtype=torch.long, device=device)
        for prefix1, indices in prefix1_to_indices.items()
    }
    return item_ids, item_to_index, item_matrix, prefix1_tensors


def hard_negative_loss(
    query: torch.Tensor,
    target_indices: torch.Tensor,
    route1_idx: torch.Tensor,
    item_matrix: torch.Tensor,
    prefix1_tensors: Mapping[int, torch.Tensor],
    num_hard_negatives: int,
    temperature: float,
) -> torch.Tensor:
    losses = []
    for row_idx in range(query.shape[0]):
        target_index = target_indices[row_idx]
        prefix1 = int(route1_idx[row_idx].item())
        bucket = prefix1_tensors[prefix1]
        scores = item_matrix[bucket] @ query[row_idx]
        not_target = bucket != target_index
        candidate_bucket = bucket[not_target]
        candidate_scores = scores[not_target]
        if candidate_bucket.numel() == 0:
            logits = (item_matrix[target_index].unsqueeze(0) @ query[row_idx].unsqueeze(-1)).T / temperature
            losses.append(F.cross_entropy(logits, torch.zeros(1, dtype=torch.long, device=query.device)))
            continue
        k = min(int(num_hard_negatives), int(candidate_bucket.numel()))
        hard_local = torch.topk(candidate_scores, k=k, dim=0).indices
        hard_indices = candidate_bucket[hard_local]
        candidate_indices = torch.cat([target_index.view(1), hard_indices], dim=0)
        logits = (item_matrix[candidate_indices] @ query[row_idx]) / temperature
        losses.append(F.cross_entropy(logits.unsqueeze(0), torch.zeros(1, dtype=torch.long, device=query.device)))
    return torch.stack(losses).mean()


def evaluate_true_prefix_retrieval(
    model: Prefix1QueryHead,
    loader: DataLoader,
    item_to_index: Mapping[str, int],
    item_matrix: torch.Tensor,
    prefix1_tensors: Mapping[int, torch.Tensor],
    device: str,
    topks: Sequence[int] = (10, 20, 50),
) -> dict[str, float]:
    model.eval()
    hits = {k: [] for k in topks}
    mrr50 = []
    with torch.no_grad():
        for batch in loader:
            query = model(batch["history_embs"].to(device), batch["history_mask"].to(device))
            route1_idx = batch["route1_idx"].to(device)
            for row_idx, target_item_id in enumerate(batch["target_item_id"]):
                target_index = item_to_index[str(target_item_id)]
                bucket = prefix1_tensors[int(route1_idx[row_idx].item())]
                scores = item_matrix[bucket] @ query[row_idx]
                order = torch.argsort(scores, descending=True)
                ranked = bucket[order]
                matches = (ranked == target_index).nonzero(as_tuple=False)
                rank = int(matches[0].item()) + 1 if matches.numel() else None
                for k in topks:
                    hits[k].append(float(rank is not None and rank <= k))
                mrr50.append(0.0 if rank is None or rank > 50 else 1.0 / rank)
    metrics = {f"true_prefix_recall@{k}": float(np.mean(values)) if values else 0.0 for k, values in hits.items()}
    metrics["true_prefix_mrr@50"] = float(np.mean(mrr50)) if mrr50 else 0.0
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--item-embedding-path", required=True)
    parser.add_argument("--item-sid-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--num-hard-negatives", type=int, default=63)
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--protocol-manifest", default=None)
    parser.add_argument("--split", default=None, help="Training split to use from --protocol-manifest.")
    parser.add_argument("--validation-split", default=None, help="Validation split to use from --protocol-manifest.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)

    item_embeddings = load_item_embeddings(args.data_dir, args.item_embedding_path)
    route_mapping = load_route_mapping(args.item_sid_path)
    route_vocab = build_route_vocab(route_mapping)
    examples = build_training_examples(args.data_dir, item_embeddings, route_mapping, max_history=args.max_history)
    protocol_manifest = load_protocol_manifest(args.protocol_manifest) if args.protocol_manifest else None
    training_item_embeddings = filter_training_item_embeddings(item_embeddings, protocol_manifest) if protocol_manifest else item_embeddings
    dataset = RouterDataset(examples, item_embeddings, route_vocab)
    item_ids, item_to_index, item_matrix, prefix1_tensors = build_item_tensors(training_item_embeddings, route_mapping, device)

    if protocol_manifest is not None:
        train_split = args.split or default_train_split_name(protocol_manifest)
        validation_split = args.validation_split or default_validation_split_name(protocol_manifest)
        train_examples = protocol_split_examples(examples, protocol_manifest, train_split)
        val_examples = protocol_split_examples(examples, protocol_manifest, validation_split)
        train_dataset = RouterDataset(train_examples, item_embeddings, route_vocab)
        val_dataset = RouterDataset(val_examples, item_embeddings, route_vocab)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
        train_size = len(train_examples)
        val_size = len(val_examples)
    else:
        val_size = max(1, int(len(dataset) * args.val_ratio))
        train_size = max(1, len(dataset) - val_size)
        if train_size + val_size > len(dataset):
            val_size = len(dataset) - train_size
        generator = torch.Generator().manual_seed(args.seed)
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=generator)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)

    model = Prefix1QueryHead(
        embedding_dim=dataset.embedding_dim,
        hidden_dim=args.hidden_dim,
        temperature=args.temperature,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    config = QueryHeadConfig(
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_hard_negatives=args.num_hard_negatives,
        temperature=args.temperature,
        device=device,
        seed=args.seed,
    )

    best = {"epoch": -1, "val_true_prefix_recall@50": -1.0}
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            history_embs = batch["history_embs"].to(device)
            history_mask = batch["history_mask"].to(device)
            route1_idx = batch["route1_idx"].to(device)
            target_indices = torch.tensor([item_to_index[str(item_id)] for item_id in batch["target_item_id"]], dtype=torch.long, device=device)
            query = model(history_embs, history_mask)
            loss = hard_negative_loss(
                query=query,
                target_indices=target_indices,
                route1_idx=route1_idx,
                item_matrix=item_matrix,
                prefix1_tensors=prefix1_tensors,
                num_hard_negatives=args.num_hard_negatives,
                temperature=args.temperature,
            )
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        metrics = evaluate_true_prefix_retrieval(model, val_loader, item_to_index, item_matrix, prefix1_tensors, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)) if losses else 0.0, **{f"val_{k}": v for k, v in metrics.items()}}
        history.append(row)
        if row["val_true_prefix_recall@50"] > best["val_true_prefix_recall@50"]:
            best = {"epoch": epoch, "val_true_prefix_recall@50": row["val_true_prefix_recall@50"], "val_true_prefix_mrr@50": row["val_true_prefix_mrr@50"]}
            best_state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        print(json.dumps(row, ensure_ascii=False), flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    payload = {
        "model_config": {
            "embedding_dim": model.embedding_dim,
            "hidden_dim": model.hidden_dim,
            "temperature": model.temperature,
        },
        "route_vocab": route_vocab.to_dict(),
        "trainer_config": asdict(config),
        "train_result": {"history": history, "best": best},
        "extra_metadata": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "num_items": len(item_ids),
            "num_examples": len(dataset),
            "num_train_examples": train_size,
            "num_val_examples": val_size,
            "data_dir": str(Path(args.data_dir).resolve()),
            "item_embedding_path": str(Path(args.item_embedding_path).resolve()),
            "item_sid_path": str(Path(args.item_sid_path).resolve()),
            "max_history": args.max_history,
            "objective": "same-prefix hard-negative item retrieval",
            "protocol_manifest": str(Path(args.protocol_manifest).resolve()) if args.protocol_manifest else None,
            "protocol_config_hash": protocol_manifest.get("config_hash") if protocol_manifest else None,
            "protocol_train_split": train_split if protocol_manifest else None,
            "protocol_validation_split": validation_split if protocol_manifest else None,
            "num_training_visible_items": len(training_item_embeddings),
        },
    }
    (output_dir / "checkpoint_meta.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "train_summary.json").write_text(json.dumps({"history": history, "best": best}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(output_dir.resolve()), "device": device, **best}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
