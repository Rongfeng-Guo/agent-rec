#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import socket
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from genrec.memory.data_adapter import load_item_embeddings
from genrec.training import build_training_examples, load_route_mapping
from genrec.training.paper_ready_protocol import (
    align_rows_by_order,
    build_bootstrap_comparison_rows,
    build_candidate_pool_breakdown,
    build_latency_rows,
    build_popularity_baseline_rows,
    build_random_matched_size_bucket_rows,
    build_route_error_breakdown,
    cold_all_metric_map,
    enrich_method,
    filter_eval_smoke_rows,
    filter_selected_policy_rows,
    read_csv_rows,
    read_jsonl,
    render_bootstrap_markdown,
    render_official_comparison_markdown,
    render_readme,
    summarize_selected_policy_rows,
    write_csv_rows,
)

OFFICIAL_FIELDNAMES = [
    "method_key",
    "display_name",
    "category",
    "claim_status",
    "notes",
    "subset",
    "domain",
    "sample_count",
    "Recall@10",
    "Recall@20",
    "Recall@50",
    "NDCG@10",
    "NDCG@20",
    "NDCG@50",
    "MRR",
    "RouteHitRate@1",
    "CandidatePoolHitRate@50",
    "ConditionalRecall@50GivenPoolHit",
    "CandidatePoolLossRate",
    "AverageCandidatePoolSize",
    "LatencyMsPerSample",
]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _git_commit(repo_root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def _resolve_path(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else REPO_ROOT / candidate


def _copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)


def _copy_or_write(path: Path, src: Path | None, *, fallback_text: str | None = None) -> None:
    if src is not None and src.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, path)
        return
    if fallback_text is not None:
        path.write_text(fallback_text, encoding="utf-8")


def _selected_summary_lookup(selected_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    return {(row["subset"], row["domain"]): row for row in summarize_selected_policy_rows(selected_rows)}


def _load_popularity_ranking(args: argparse.Namespace) -> tuple[list[str], list[str]]:
    data_dir = _resolve_path(args.data_dir)
    item_embedding_path = _resolve_path(args.item_embedding_path)
    item_sid_path = _resolve_path(args.item_sid_path)
    item_embeddings = load_item_embeddings(data_dir, item_embedding_path)
    route_mapping = load_route_mapping(item_sid_path)
    examples = build_training_examples(data_dir, item_embeddings, route_mapping, max_history=args.max_history)
    train_target_counts = Counter(str(example.target_item_id) for example in examples)
    popularity_ranking = [
        item_id for item_id, _ in sorted(train_target_counts.items(), key=lambda item_count: (-item_count[1], item_count[0]))
    ]
    return popularity_ranking, sorted(str(item_id) for item_id in item_embeddings.keys())


def _load_eval_rows(eval_smoke_dir: Path, selected_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    eval_rows = read_jsonl(eval_smoke_dir / "per_sample_results.jsonl")
    selected_cold_rows = [row for row in selected_rows if row.get("subset") == "cold"]
    metadata_rows = align_rows_by_order(filter_eval_smoke_rows(eval_rows, mode="metadata"), selected_cold_rows)
    oracle_p1_rows = align_rows_by_order(
        filter_eval_smoke_rows(eval_rows, mode="oracle_route", prefix_len=1),
        selected_cold_rows,
    )
    oracle_p2_rows = align_rows_by_order(
        filter_eval_smoke_rows(eval_rows, mode="oracle_route", prefix_len=2),
        selected_cold_rows,
    )
    return metadata_rows, oracle_p1_rows, oracle_p2_rows


def _extra_summary_rows(summary_csv: Path) -> list[dict[str, Any]]:
    if not summary_csv.exists():
        return []

    sources = [
        (
            "predicted_prefix1_top1_single_query",
            "Predicted Prefix-1 Top-1 Single Query",
            "domain_adaptive",
            "predicted_route_p1",
            "diagnostic",
            "Existing routed single-query reference from validation_query_selector_eval_20260606_234148.",
        ),
        (
            "predicted_prefix1_top4_single_query",
            "Predicted Prefix-1 Top-4 Single Query",
            "domain_adaptive",
            "predicted_route_p1_top4_quota",
            "diagnostic",
            "Best available top-4 routed single-query reference from validation_query_selector_eval_20260606_234148.",
        ),
    ]
    rows = read_csv_rows(summary_csv)
    out: list[dict[str, Any]] = []
    for method_key, display_name, query_source, mode, claim_status, notes in sources:
        for row in rows:
            if row.get("query_source") != query_source or row.get("mode") != mode:
                continue
            out.append(
                {
                    "method_key": method_key,
                    "display_name": display_name,
                    "category": "predicted_route",
                    "claim_status": claim_status,
                    "notes": notes,
                    "subset": row["subset"],
                    "domain": "ALL",
                    "sample_count": int(row["sample_count"]),
                    "Recall@10": float(row["Recall@10"]),
                    "Recall@20": float(row["Recall@20"]),
                    "Recall@50": float(row["Recall@50"]),
                    "NDCG@10": float(row["NDCG@10"]),
                    "NDCG@20": float(row["NDCG@20"]),
                    "NDCG@50": float(row["NDCG@50"]),
                    "MRR": float(row["MRR@50"]),
                    "RouteHitRate@1": None,
                    "CandidatePoolHitRate@50": (
                        float(row["CandidatePoolHitRate"]) if row.get("CandidatePoolHitRate") not in (None, "") else None
                    ),
                    "ConditionalRecall@50GivenPoolHit": None,
                    "CandidatePoolLossRate": (
                        float(row["CandidatePoolLossRate"]) if row.get("CandidatePoolLossRate") not in (None, "") else None
                    ),
                    "AverageCandidatePoolSize": None,
                    "LatencyMsPerSample": float(row["avg_latency_ms"]) if row.get("avg_latency_ms") not in (None, "") else None,
                }
            )
    return out


def _build_comparison_rows(
    args: argparse.Namespace,
    *,
    selected_rows: list[dict[str, Any]],
    eval_smoke_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_rows, oracle_p1_rows, oracle_p2_rows = _load_eval_rows(eval_smoke_dir, selected_rows)
    popularity_ranking, all_item_ids = _load_popularity_ranking(args)
    popularity_rows = build_popularity_baseline_rows(selected_rows, popularity_ranking)
    random_rows = build_random_matched_size_bucket_rows(selected_rows, all_item_ids, seed=args.bootstrap_seed)

    comparison_rows: list[dict[str, Any]] = []
    comparison_rows += enrich_method(
        summarize_selected_policy_rows(popularity_rows),
        method_key="popularity_global",
        display_name="Popularity",
        category="baseline",
        claim_status="claimable",
        notes="Global target-frequency ranking from training examples.",
    )
    comparison_rows += enrich_method(
        summarize_selected_policy_rows(metadata_rows),
        method_key="metadata_global_mean_query",
        display_name="Metadata Global Mean Query",
        category="baseline",
        claim_status="claimable",
        notes="Global no-route metadata retrieval baseline from eval_smoke_20260606_072300.",
    )
    comparison_rows += enrich_method(
        summarize_selected_policy_rows(random_rows),
        method_key="random_matched_size_bucket",
        display_name="Random Matched-Size Bucket",
        category="baseline",
        claim_status="claimable",
        notes="Uniform random candidate pool matched to selected-policy pool size per sample.",
    )
    comparison_rows += enrich_method(
        summarize_selected_policy_rows(selected_rows),
        method_key="predicted_route_validation_selected",
        display_name="Predicted Route Validation-Selected (Explicit Script OldV0)",
        category="predicted_route",
        claim_status="claimable",
        notes="Frozen selector preset rerun with locked fusion_config and route_score_weight=0.0.",
    )
    comparison_rows += enrich_method(
        summarize_selected_policy_rows(oracle_p1_rows),
        method_key="oracle_prefix1_route",
        display_name="Oracle Prefix-1 Route",
        category="upper_bound",
        claim_status="diagnostic",
        notes="Upper bound with target prefix-1 route injected.",
    )
    comparison_rows += enrich_method(
        summarize_selected_policy_rows(oracle_p2_rows),
        method_key="oracle_prefix2_route",
        display_name="Oracle Prefix-2 Route",
        category="upper_bound",
        claim_status="diagnostic",
        notes="Upper bound with target prefix-2 route injected.",
    )
    comparison_rows += _extra_summary_rows(_resolve_path(args.extra_summary_csv))
    comparison_rows.sort(key=lambda row: (row["display_name"], row["subset"], row["domain"]))
    return comparison_rows, metadata_rows, random_rows


def _render_reproduce_script(args: argparse.Namespace) -> str:
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "cd /home/grf/agent-rec",
        "export CUDA_VISIBLE_DEVICES=0",
        args.selector_command.strip(),
        args.locked_eval_command.strip(),
    ]
    bundle_parts = [
        "/home/grf/.conda/envs/gdpo/bin/python scripts/oracle_route_memory/build_protocol_v2_bundle.py",
        f"--output-dir {args.output_dir}",
        f"--protocol-dir {args.protocol_dir}",
        f"--selector-dir {args.selector_dir}",
        f"--locked-eval-dir {args.locked_eval_dir}",
        f"--eval-smoke-dir {args.eval_smoke_dir}",
        f"--progress-report-md {args.progress_report_md}",
        f"--data-dir {args.data_dir}",
        f"--item-embedding-path {args.item_embedding_path}",
        f"--item-sid-path {args.item_sid_path}",
        f"--extra-summary-csv {args.extra_summary_csv}",
        f"--bootstrap-reps {args.bootstrap_reps}",
        f"--bootstrap-seed {args.bootstrap_seed}",
        f"--max-history {args.max_history}",
    ]
    if args.official_comparison_csv:
        bundle_parts.append(f"--official-comparison-csv {args.official_comparison_csv}")
    if args.reference_bundle_dir:
        bundle_parts.append(f"--reference-bundle-dir {args.reference_bundle_dir}")
    lines.append(" ".join(bundle_parts))
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the checked-in paper-ready v2 protocol bundle.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--protocol-dir", required=True)
    parser.add_argument("--selector-dir", required=True)
    parser.add_argument("--locked-eval-dir", required=True)
    parser.add_argument("--eval-smoke-dir", required=True)
    parser.add_argument("--progress-report-md", required=True)
    parser.add_argument("--official-comparison-csv")
    parser.add_argument("--reference-bundle-dir")
    parser.add_argument("--data-dir", default="user_simulator")
    parser.add_argument(
        "--item-embedding-path",
        default="outputs/oracle_route_memory/assets/metadata_embeddings/item_embeddings.npy",
    )
    parser.add_argument(
        "--item-sid-path",
        default="outputs/oracle_route_memory/assets/proxy_routes_b16_d2/item_sid_mapping.json",
    )
    parser.add_argument(
        "--extra-summary-csv",
        default="outputs/oracle_route_memory/validation_query_selector_eval_20260606_234148/summary.csv",
    )
    parser.add_argument("--max-history", type=int, default=10)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=42)
    parser.add_argument("--selector-command", default="# selector command unavailable")
    parser.add_argument("--locked-eval-command", default="# locked eval command unavailable")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    protocol_dir = Path(args.protocol_dir)
    selector_dir = Path(args.selector_dir)
    locked_eval_dir = Path(args.locked_eval_dir)
    eval_smoke_dir = Path(args.eval_smoke_dir)
    reference_bundle_dir = Path(args.reference_bundle_dir) if args.reference_bundle_dir else None

    for name in ("split_manifest.json", "leakage_audit.json", "leakage_audit.md"):
        _copy_if_exists(protocol_dir / name, output_dir / name)

    fusion_config_path = selector_dir / "fusion_config.json"
    _copy_if_exists(fusion_config_path, output_dir / "fusion_config.json")
    if (selector_dir / "method_config.json").exists():
        _copy_if_exists(selector_dir / "method_config.json", output_dir / "method_config.json")
    elif reference_bundle_dir is not None and (reference_bundle_dir / "method_config.json").exists():
        _copy_if_exists(reference_bundle_dir / "method_config.json", output_dir / "method_config.json")
    else:
        _copy_if_exists(fusion_config_path, output_dir / "method_config.json")

    selected_rows = filter_selected_policy_rows(read_jsonl(locked_eval_dir / "results.jsonl"))
    selected_summary_lookup = _selected_summary_lookup(selected_rows)
    selected_cold_all = selected_summary_lookup[("cold", "ALL")]
    claimable_recall50 = float(selected_cold_all["Recall@50"])

    write_csv_rows(output_dir / "route_error_breakdown.csv", build_route_error_breakdown(selected_rows))
    write_csv_rows(output_dir / "candidate_pool_breakdown.csv", build_candidate_pool_breakdown(selected_rows))

    copied_reference_tables = False
    if reference_bundle_dir is not None and (reference_bundle_dir / "official_comparison.csv").exists():
        copied_reference_tables = True
        _copy_if_exists(reference_bundle_dir / "official_comparison.csv", output_dir / "official_comparison.csv")
        _copy_if_exists(reference_bundle_dir / "official_comparison.md", output_dir / "official_comparison.md")
        _copy_if_exists(reference_bundle_dir / "README.md", output_dir / "README.md")
        _copy_if_exists(reference_bundle_dir / "bootstrap_comparison.csv", output_dir / "bootstrap_comparison.csv")
        _copy_if_exists(reference_bundle_dir / "bootstrap_comparison.json", output_dir / "bootstrap_comparison.json")
        _copy_if_exists(reference_bundle_dir / "bootstrap_report.md", output_dir / "bootstrap_report.md")
        _copy_if_exists(reference_bundle_dir / "latency_summary.csv", output_dir / "latency_summary.csv")
    else:
        comparison_rows, metadata_rows, random_rows = _build_comparison_rows(
            args,
            selected_rows=selected_rows,
            eval_smoke_dir=eval_smoke_dir,
        )
        write_csv_rows(output_dir / "official_comparison.csv", comparison_rows, fieldnames=OFFICIAL_FIELDNAMES)
        (output_dir / "official_comparison.md").write_text(
            render_official_comparison_markdown(comparison_rows),
            encoding="utf-8",
        )
        bootstrap_rows = build_bootstrap_comparison_rows(
            selected_rows,
            [
                ("metadata_global_mean_query", "Metadata Global Mean Query", metadata_rows),
                ("random_matched_size_bucket", "Random Matched-Size Bucket", random_rows),
            ],
            k=50,
            reps=args.bootstrap_reps,
            seed=args.bootstrap_seed,
        )
        write_csv_rows(output_dir / "bootstrap_comparison.csv", bootstrap_rows)
        (output_dir / "bootstrap_comparison.json").write_text(
            json.dumps(bootstrap_rows, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (output_dir / "bootstrap_report.md").write_text(render_bootstrap_markdown(bootstrap_rows), encoding="utf-8")
        write_csv_rows(output_dir / "latency_summary.csv", build_latency_rows(comparison_rows))
        (output_dir / "README.md").write_text(render_readme(claimable_recall50), encoding="utf-8")

    _copy_or_write(output_dir / "progress_report.md", Path(args.progress_report_md))
    if not (output_dir / "official_comparison.md").exists():
        (output_dir / "official_comparison.md").write_text("", encoding="utf-8")
    if not (output_dir / "README.md").exists():
        (output_dir / "README.md").write_text(render_readme(claimable_recall50), encoding="utf-8")

    run_metadata = {
        "created_at": _utc_now(),
        "git_commit": _git_commit(REPO_ROOT),
        "protocol_manifest": str((protocol_dir / "split_manifest.json").resolve()),
        "selector_dir": str(selector_dir.resolve()),
        "locked_eval_dir": str(locked_eval_dir.resolve()),
        "eval_smoke_dir": str(eval_smoke_dir.resolve()),
        "reference_bundle_dir": str(reference_bundle_dir.resolve()) if reference_bundle_dir else None,
        "copied_reference_tables": copied_reference_tables,
        "bootstrap_reps": args.bootstrap_reps,
        "bootstrap_seed": args.bootstrap_seed,
        "claimable_cold_recall50": claimable_recall50,
        "hostname": socket.gethostname(),
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    reproduce_path = output_dir / "reproduce.sh"
    reproduce_path.write_text(_render_reproduce_script(args), encoding="utf-8")
    reproduce_path.chmod(0o755)

    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir.resolve()),
                "selected_rows": len(selected_rows),
                "claimable_cold_recall50": claimable_recall50,
                "copied_reference_tables": copied_reference_tables,
                "cold_all_metrics": (
                    cold_all_metric_map(read_csv_rows(output_dir / "official_comparison.csv"))
                    if (output_dir / "official_comparison.csv").exists() and not copied_reference_tables
                    else {"predicted_route_validation_selected": claimable_recall50}
                ),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
