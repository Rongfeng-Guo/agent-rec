#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.oracle_route_memory.train_candidate_level_source_ranker import summarize_eval_rows

CLAIM_BOUNDARY = (
    "Validation-only locked manifest check. This validates the locked H5-D validation-selected outputs; "
    "it is not a fresh blind-confirmation result."
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_close(name: str, actual: float, expected: float, tolerance: float) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f"{name} mismatch: actual={actual!r} expected={expected!r}")


def hit_count(rows: list[Mapping[str, Any]], topk: int) -> int:
    return sum(1 for row in rows if row.get("match_rank") is not None and int(row["match_rank"]) <= topk)


def component_model_checks(manifest: Mapping[str, Any], repo_root: Path) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    for name, component in sorted(manifest.get("component_rankers", {}).items()):
        output_dir = repo_root / str(component["output_dir"])
        model_path = output_dir / "model.pkl"
        checks.append(
            {
                "name": str(name),
                "output_dir": str(output_dir),
                "model_path": str(model_path),
                "exists": model_path.exists(),
            }
        )
    return checks


def validate_manifest(manifest_path: Path, *, repo_root: Path, topk: int = 50, tolerance: float = 1e-9) -> dict[str, Any]:
    manifest = read_json(manifest_path)
    selected = manifest["selected_outputs"]
    output_path = repo_root / selected["cold_like_outputs"]
    if not output_path.exists():
        raise FileNotFoundError(output_path)
    for section_name in ("candidate_export", "selected_outputs"):
        for key, value in manifest.get(section_name, {}).items():
            if isinstance(value, str) and (key.endswith("rows") or key in {"output_dir", "summary", "analyzer"}):
                path = repo_root / value
                if not path.exists():
                    raise FileNotFoundError(path)
    for component in manifest.get("component_rankers", {}).values():
        for key in ("output_dir", "cold_like_outputs"):
            path = repo_root / component[key]
            if not path.exists():
                raise FileNotFoundError(path)

    rows = read_json(output_path)
    if not isinstance(rows, list):
        raise ValueError(f"Expected selected output rows to be a list: {output_path}")
    metric = summarize_eval_rows(rows, topk=topk)
    expected = manifest["validation_metrics"]
    assert int(expected["sample_count"]) == len(rows)
    assert int(expected["hits_at_50"]) == hit_count(rows, topk)
    assert_close("Recall@50", metric[f"Recall@{topk}"], expected["Recall@50"], tolerance)
    assert_close("CandidatePoolHitRate", metric["CandidatePoolHitRate"], expected["CandidatePoolHitRate"], tolerance)
    assert_close(
        "ConditionalRecall@50GivenPoolHit",
        metric["ConditionalRecall@50GivenPoolHit"],
        expected["ConditionalRecall@50GivenPoolHit"],
        tolerance,
    )

    domain_results: dict[str, dict[str, Any]] = {}
    for domain, expected_domain in expected.get("domain_metrics", {}).items():
        domain_rows = [row for row in rows if str(row.get("domain")) == str(domain)]
        actual_hits = hit_count(domain_rows, topk)
        if actual_hits != int(expected_domain["hits_at_50"]):
            raise ValueError(f"{domain} hits mismatch: actual={actual_hits} expected={expected_domain['hits_at_50']}")
        assert_close(
            f"{domain} Recall@50",
            actual_hits / len(domain_rows) if domain_rows else 0.0,
            expected_domain["Recall@50"],
            tolerance,
        )
        domain_results[domain] = {
            "sample_count": len(domain_rows),
            "hits_at_50": actual_hits,
            "Recall@50": actual_hits / len(domain_rows) if domain_rows else 0.0,
        }
    return {
        "status": "ok",
        "claim_boundary": CLAIM_BOUNDARY,
        "manifest": str(manifest_path),
        "selected_output": str(output_path),
        "metric": {
            "sample_count": len(rows),
            "hits_at_50": hit_count(rows, topk),
            f"Recall@{topk}": metric[f"Recall@{topk}"],
            "CandidatePoolHitRate": metric["CandidatePoolHitRate"],
            "ConditionalRecall@50GivenPoolHit": metric["ConditionalRecall@50GivenPoolHit"],
        },
        "domain_results": domain_results,
        "component_model_checks": component_model_checks(manifest, repo_root),
        "next_target": (
            "If status is ok, keep this manifest locked for validation-only reporting. Before any fresh claim, "
            "run readiness checks, score a clearly fresh/unconsumed split with locked model.pkl files, and keep "
            "fresh metrics separate from this validation result."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a locked H5 policy manifest against its output rows.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()
    result = validate_manifest(Path(args.manifest), repo_root=Path(args.repo_root), topk=int(args.topk))
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
