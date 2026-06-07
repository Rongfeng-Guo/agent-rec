#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def row_from_summary(rows: List[Mapping[str, Any]], *, subset: str, mode: str | None = None, query_source: str | None = None) -> Dict[str, Any] | None:
    for row in rows:
        if row.get("subset") != subset:
            continue
        if mode is not None and row.get("mode") != mode:
            continue
        if query_source is not None and row.get("query_source") != query_source:
            continue
        return dict(row)
    return None


def select_best_cold_row(rows: List[Mapping[str, Any]]) -> Dict[str, Any] | None:
    cold_rows = [dict(row) for row in rows if row.get("subset") == "cold"]
    if not cold_rows:
        return None
    cold_rows.sort(key=lambda row: (as_float(row.get("Recall@50")), as_float(row.get("Recall@20")), as_float(row.get("Recall@10"))), reverse=True)
    return cold_rows[0]


def find_method_row(rows: List[Mapping[str, Any]], method_keys: List[str]) -> Dict[str, Any] | None:
    for method_key in method_keys:
        for row in rows:
            if row.get("method_key") == method_key and row.get("subset") == "cold" and row.get("domain") == "ALL":
                return dict(row)
    return None


def make_progress_rows(
    outputs_root: Path,
    *,
    official_comparison_dir: str = "official_comparison_20260607",
    locked_eval_dir: str = "validation_fusion_locked_cold_gpu0",
    seed7_locked_eval_dir: str = "validation_fusion_locked_cold_seed7_policy_gpu0",
    domain_adaptive_eval_dir: str = "fusion_domain_adaptive_eval_20260606_233954",
    prefix1_domain_adaptive_eval_dir: str = "prefix1_domain_adaptive_merge_eval_20260606_232037",
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    oracle_summary = read_csv(outputs_root / "eval_smoke_20260606_072300" / "summary.csv")
    for label, mode, prefix_len in [
        ("Metadata Global", "metadata", ""),
        ("Oracle Route P1", "oracle_route", "1"),
        ("Oracle Route P2", "oracle_route", "2"),
    ]:
        match = None
        for row in oracle_summary:
            if row.get("mode") != mode:
                continue
            if prefix_len and row.get("prefix_len") != prefix_len:
                continue
            if not prefix_len and row.get("prefix_len") not in ("", None):
                continue
            match = dict(row)
            break
        if match is None:
            continue
        rows.append({
            "family": "oracle_baseline",
            "label": label,
            "source_dir": str((outputs_root / "eval_smoke_20260606_072300").resolve()),
            "subset": "cold",
            "selection_status": "fixed_reference" if mode == "metadata" else "oracle_upper_bound",
            "Recall@10": as_float(match.get("Recall@10")),
            "Recall@20": as_float(match.get("Recall@20")),
            "Recall@50": as_float(match.get("Recall@50")),
            "NDCG@50": as_float(match.get("NDCG@50")),
            "MRR@50": as_float(match.get("MRR@50")),
        })

    official_dir = outputs_root / official_comparison_dir
    official_rows = read_csv(official_dir / "comparison.csv")
    for method_keys in (
        ["predicted_route_validation_selected", "predicted_route_validation_selected_seed42"],
        ["predicted_route_diagnostic_fusion_lr_p1t4"],
    ):
        match = find_method_row(official_rows, method_keys)
        if match is None:
            continue
        rows.append({
            "family": "official_comparison",
            "label": match["display_name"],
            "source_dir": str(official_dir.resolve()),
            "subset": "cold",
            "selection_status": match["selection_status"],
            "Recall@10": as_float(match.get("Recall@10")),
            "Recall@20": as_float(match.get("Recall@20")),
            "Recall@50": as_float(match.get("Recall@50")),
            "NDCG@50": as_float(match.get("NDCG@50")),
            "MRR@50": as_float(match.get("MRR@50")),
        })

    locked_dir = outputs_root / locked_eval_dir
    validation_rows = read_csv(locked_dir / "summary.csv")
    selected = row_from_summary(validation_rows, subset="cold", mode="validation_selected", query_source="selected_policy")
    if selected is not None:
        rows.append({
            "family": "validation_selected",
            "label": "Validation Selected Policy",
            "source_dir": str(locked_dir.resolve()),
            "subset": "cold",
            "selection_status": "claimable",
            "Recall@10": as_float(selected.get("Recall@10")),
            "Recall@20": as_float(selected.get("Recall@20")),
            "Recall@50": as_float(selected.get("Recall@50")),
            "NDCG@50": as_float(selected.get("NDCG@50")),
            "MRR@50": as_float(selected.get("MRR@50")),
        })

    diagnostic_candidates = {
        "Domain Adaptive Fusion": outputs_root / domain_adaptive_eval_dir / "summary.csv",
        "Prefix1 Domain Adaptive Merge": outputs_root / prefix1_domain_adaptive_eval_dir / "summary.csv",
        "Validation Selector Seed7 Policy": outputs_root / seed7_locked_eval_dir / "summary.csv",
    }
    for label, path in diagnostic_candidates.items():
        best = select_best_cold_row(read_csv(path))
        if best is None:
            continue
        rows.append({
            "family": "diagnostic",
            "label": label,
            "source_dir": str(path.parent.resolve()),
            "subset": "cold",
            "selection_status": "diagnostic_only",
            "Recall@10": as_float(best.get("Recall@10")),
            "Recall@20": as_float(best.get("Recall@20")),
            "Recall@50": as_float(best.get("Recall@50")),
            "NDCG@50": as_float(best.get("NDCG@50")),
            "MRR@50": as_float(best.get("MRR@50")),
            "mode": best.get("mode", ""),
            "query_source": best.get("query_source", ""),
        })

    baseline = next((row for row in rows if row["label"] == "Metadata Global"), None)
    baseline_r50 = baseline["Recall@50"] if baseline is not None else 0.0
    for row in rows:
        row["DeltaVsMetadata@50"] = row["Recall@50"] - baseline_r50
        row["RatioVsMetadata@50"] = (row["Recall@50"] / baseline_r50) if baseline_r50 > 0 else 0.0
        row["AboveTarget0.029"] = row["Recall@50"] >= 0.029
    return rows


def render_markdown(rows: List[Mapping[str, Any]]) -> str:
    def fmt(value: Any) -> str:
        if isinstance(value, bool):
            return "yes" if value else "no"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    lines = [
        "# Predicted Route Progress Report",
        "",
        "| family | label | selection | Recall@10 | Recall@20 | Recall@50 | NDCG@50 | MRR@50 | delta@50 | ratio@50 | >=0.029 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(rows, key=lambda item: (item["family"], -item["Recall@50"], item["label"])):
        lines.append(
            f"| {row['family']} | {row['label']} | {row['selection_status']} | {fmt(row['Recall@10'])} | {fmt(row['Recall@20'])} | {fmt(row['Recall@50'])} | {fmt(row['NDCG@50'])} | {fmt(row['MRR@50'])} | {fmt(row['DeltaVsMetadata@50'])} | {fmt(row['RatioVsMetadata@50'])} | {fmt(row['AboveTarget0.029'])} |"
        )
    claimable = [row for row in rows if row["selection_status"] == "claimable"]
    diagnostics = [row for row in rows if row["selection_status"] == "diagnostic_only"]
    lines.extend(["", "## Readout", ""])
    if claimable:
        best_claimable = max(claimable, key=lambda row: row["Recall@50"])
        lines.append(f"- Best claimable row: `{best_claimable['label']}` with cold Recall@50 `{best_claimable['Recall@50']:.4f}`.")
    if diagnostics:
        best_diag = max(diagnostics, key=lambda row: row["Recall@50"])
        lines.append(f"- Best diagnostic row: `{best_diag['label']}` with cold Recall@50 `{best_diag['Recall@50']:.4f}` from `{best_diag.get('mode', '')}`.")
    lines.append("- Current target threshold remains `Recall@50 >= 0.0290`.")
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: List[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    extras = sorted({key for row in rows for key in row.keys() if key not in fieldnames})
    fieldnames.extend(extras)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-root", default="outputs/oracle_route_memory")
    parser.add_argument("--official-comparison-dir", default="official_comparison_20260607")
    parser.add_argument("--locked-eval-dir", default="validation_fusion_locked_cold_gpu0")
    parser.add_argument("--seed7-locked-eval-dir", default="validation_fusion_locked_cold_seed7_policy_gpu0")
    parser.add_argument("--domain-adaptive-eval-dir", default="fusion_domain_adaptive_eval_20260606_233954")
    parser.add_argument("--prefix1-domain-adaptive-eval-dir", default="prefix1_domain_adaptive_merge_eval_20260606_232037")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = make_progress_rows(
        outputs_root,
        official_comparison_dir=args.official_comparison_dir,
        locked_eval_dir=args.locked_eval_dir,
        seed7_locked_eval_dir=args.seed7_locked_eval_dir,
        domain_adaptive_eval_dir=args.domain_adaptive_eval_dir,
        prefix1_domain_adaptive_eval_dir=args.prefix1_domain_adaptive_eval_dir,
    )
    (output_dir / "progress.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(output_dir / "progress.csv", rows)
    (output_dir / "report.md").write_text(render_markdown(rows), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(output_dir), "num_rows": len(rows)}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
