#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
import socket
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from scripts.oracle_route_memory.handoff_io import ensure_empty_output_dir
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from handoff_io import ensure_empty_output_dir

from scripts.oracle_route_memory.train_candidate_level_source_ranker import (
    FEATURE_NAMES,
    QUERY_SOURCES,
    PairwiseLinearRanker,
    evaluate_model,
    write_json,
)


NEXT_TARGET = (
    "If this is a validation replay score, validate the domain-routed replay against the locked H5-D "
    "outputs with validate_h5_loaded_model_replay.py. If this is a fresh score, combine locked h100/h300 "
    "outputs with the locked domain route and feed the result into the fresh confirmation report; do not "
    "retrain, retune, or change the locked model files after fresh labels are available."
)



class LockedRankerUnpickler(pickle.Unpickler):
    def find_class(self, module: str, name: str) -> Any:
        if module == "__main__" and name == "PairwiseLinearRanker":
            return PairwiseLinearRanker
        return super().find_class(module, name)


def load_model_bundle(model_path: Path) -> dict[str, Any]:
    with model_path.open("rb") as handle:
        bundle = LockedRankerUnpickler(handle).load()
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError(f"Expected a model bundle with a 'model' key: {model_path}")
    feature_names = tuple(bundle.get("feature_names", ()))
    query_sources = tuple(bundle.get("query_sources", ()))
    if feature_names != FEATURE_NAMES:
        raise ValueError("Loaded model feature_names do not match current FEATURE_NAMES")
    if query_sources != QUERY_SOURCES:
        raise ValueError("Loaded model query_sources do not match current QUERY_SOURCES")
    return bundle


def format_optional_float(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def render_report(summary: Mapping[str, Any], topk: int) -> str:
    metric = summary["eval_metric"]
    return "\n".join(
        [
            "# H5 Loaded Candidate-Level Source Ranker Score",
            "",
            f"- Model path: `{summary['model_path']}`",
            f"- Eval rows: `{summary['eval_rows_path']}`",
            f"- Outputs: `{summary['files']['scored_outputs']}`",
            "",
            "## Evaluation",
            "",
            f"- Recall@{topk}: `{metric[f'Recall@{topk}']:.6f}`",
            f"- CandidatePoolHitRate: `{metric['CandidatePoolHitRate']:.6f}`",
            f"- ConditionalRecall@50GivenPoolHit: `{metric['ConditionalRecall@50GivenPoolHit']:.6f}`",
            f"- AvgCandidatePoolMatchRank: `{format_optional_float(metric['AvgCandidatePoolMatchRank'])}`",
            f"- AvgPoolHitRankMissMatchRank: `{format_optional_float(metric['AvgPoolHitRankMissMatchRank'])}`",
            f"- OracleSourceHit@{topk}: `{metric[f'OracleSourceHit@{topk}Rate']:.6f}`",
            f"- AvgOracleSourceMatchRank: `{format_optional_float(metric['AvgOracleSourceMatchRank'])}`",
            "",
            "## Next Target",
            "",
            str(summary["next_target"]),
            "",
        ]
    )


def score_loaded_ranker(
    *,
    model_path: Path,
    eval_rows_path: Path,
    output_dir: Path,
    topk: int = 50,
    output_filename: str = "scored_outputs.json",
) -> dict[str, Any]:
    ensure_empty_output_dir(output_dir)
    bundle = load_model_bundle(model_path)
    metric, outputs = evaluate_model(bundle["model"], eval_rows_path, topk=topk)
    outputs_path = output_dir / output_filename
    write_json(outputs_path, outputs)
    summary = {
        "name": "H5LoadedCandidateLevelSourceRankerScore",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "hostname": socket.gethostname(),
        "model_path": str(model_path),
        "eval_rows_path": str(eval_rows_path),
        "feature_names": list(FEATURE_NAMES),
        "query_sources": list(QUERY_SOURCES),
        "eval_metric": metric,
        "files": {
            "scored_outputs": str(outputs_path),
        },
        "next_target": NEXT_TARGET,
    }
    write_json(output_dir / "summary.json", summary)
    (output_dir / "report.md").write_text(render_report(summary, topk), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidate-level rows with a locked H5 ranker model.pkl.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--eval-rows", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--output-filename", default="scored_outputs.json")
    args = parser.parse_args()

    summary = score_loaded_ranker(
        model_path=Path(args.model),
        eval_rows_path=Path(args.eval_rows),
        output_dir=Path(args.output_dir),
        topk=int(args.topk),
        output_filename=str(args.output_filename),
    )
    metric = summary["eval_metric"]
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(Path(args.output_dir).resolve()),
                f"Recall@{args.topk}": metric[f"Recall@{args.topk}"],
                "CandidatePoolHitRate": metric["CandidatePoolHitRate"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
