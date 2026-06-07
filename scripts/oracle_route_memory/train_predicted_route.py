#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import json
import random
import socket
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in __import__('sys').path:
    __import__('sys').path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from genrec.models import LateBoundRouter
from genrec.training import (
    RouterDataset,
    RouterTrainer,
    TrainerConfig,
    build_route_vocab,
    build_training_examples,
    default_train_split_name,
    default_validation_split_name,
    load_protocol_manifest,
    load_route_mapping,
    protocol_split_examples,
)
from genrec.memory.data_adapter import load_item_embeddings


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).lower() in {"1", "true", "yes", "y", "t"}


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--item-embedding-path", required=True)
    parser.add_argument("--item-sid-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-contrastive", type=float, default=0.2)
    parser.add_argument("--training-objective", default="prefix1", choices=["full", "prefix1"])
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
    if protocol_manifest is not None:
        train_split = args.split or default_train_split_name(protocol_manifest)
        validation_split = args.validation_split or default_validation_split_name(protocol_manifest)
        train_examples = protocol_split_examples(examples, protocol_manifest, train_split)
        val_examples = protocol_split_examples(examples, protocol_manifest, validation_split)
    else:
        dataset = RouterDataset(examples, item_embeddings, route_vocab)
        val_size = max(1, int(len(dataset) * args.val_ratio))
        train_size = max(1, len(dataset) - val_size)
        if train_size + val_size > len(dataset):
            val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(args.seed),
        )
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=dataset.collate_fn)
        train_examples = None
        val_examples = None

    if protocol_manifest is not None:
        train_dataset = RouterDataset(train_examples, item_embeddings, route_vocab)
        val_dataset = RouterDataset(val_examples, item_embeddings, route_vocab)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)
        train_size = len(train_examples)
        val_size = len(val_examples)
        dataset = RouterDataset(examples, item_embeddings, route_vocab)
    else:
        train_size = len(train_dataset)
        val_size = len(val_dataset)

    model = LateBoundRouter(
        embedding_dim=dataset.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_prefix1=route_vocab.num_prefix1,
        num_prefix2=route_vocab.num_prefix2,
    )
    trainer = RouterTrainer(
        model,
        TrainerConfig(
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            lambda_contrastive=args.lambda_contrastive,
            device=device,
            seed=args.seed,
            training_objective=args.training_objective,
        ),
    )
    result = trainer.fit(train_loader, val_loader)

    output_dir = Path(args.output_dir)
    trainer.save(
        output_dir,
        route_vocab,
        result,
        {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "hostname": socket.gethostname(),
            "num_examples": len(dataset),
            "num_train_examples": train_size,
            "num_val_examples": val_size,
            "data_dir": str(Path(args.data_dir).resolve()),
            "item_embedding_path": str(Path(args.item_embedding_path).resolve()),
            "item_sid_path": str(Path(args.item_sid_path).resolve()),
            "max_history": args.max_history,
            "training_objective": args.training_objective,
            "protocol_manifest": str(Path(args.protocol_manifest).resolve()) if args.protocol_manifest else None,
            "protocol_config_hash": protocol_manifest.get("config_hash") if protocol_manifest else None,
            "protocol_train_split": train_split if protocol_manifest else None,
            "protocol_validation_split": validation_split if protocol_manifest else None,
        },
    )
    (output_dir / "train_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(output_dir.resolve()), "device": device, **result["best"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
