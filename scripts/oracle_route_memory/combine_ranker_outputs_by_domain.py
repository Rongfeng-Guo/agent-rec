#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.oracle_route_memory.train_candidate_level_source_ranker import summarize_eval_rows

try:
    from scripts.oracle_route_memory.handoff_io import ensure_empty_output_dir
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from handoff_io import ensure_empty_output_dir


NEXT_TARGET = (
    "If these are validation outputs, validate the combined rows against the locked H5-D manifest before "
    "using them as selected evidence. If these are fresh outputs, feed them directly into the fresh "
    "confirmation report without retraining, retuning, or changing the locked domain route."
)


def read_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return [dict(row) for row in payload]


def parse_named_path(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"Expected NAME=PATH, got {value!r}")
    name, path = value.split("=", 1)
    if not name:
        raise ValueError(f"Missing source name in {value!r}")
    return name, Path(path)


def parse_domain_source(value: str) -> tuple[str, str]:
    if "=" not in value:
        raise ValueError(f"Expected DOMAIN=SOURCE_NAME, got {value!r}")
    domain, source = value.split("=", 1)
    if not domain or not source:
        raise ValueError(f"Expected DOMAIN=SOURCE_NAME, got {value!r}")
    return domain, source


def format_optional_float(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def rows_by_sample(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        sample_id = str(row["sample_id"])
        if sample_id in indexed:
            raise ValueError(f"Duplicate sample_id {sample_id!r}")
        indexed[sample_id] = row
    return indexed


def combine_rows_by_domain(
    sources: Mapping[str, Sequence[Mapping[str, Any]]],
    domain_sources: Mapping[str, str],
    *,
    default_source: str,
) -> list[dict[str, Any]]:
    if default_source not in sources:
        raise ValueError(f"default_source {default_source!r} is not in sources")
    indexed_sources = {name: rows_by_sample(rows) for name, rows in sources.items()}
    default_rows = list(sources[default_source])
    combined: list[dict[str, Any]] = []
    for default_row in default_rows:
        sample_id = str(default_row["sample_id"])
        domain = str(default_row.get("domain", ""))
        source_name = domain_sources.get(domain, default_source)
        if source_name not in indexed_sources:
            raise ValueError(f"Domain {domain!r} mapped to unknown source {source_name!r}")
        if sample_id not in indexed_sources[source_name]:
            raise ValueError(f"sample_id {sample_id!r} is missing from source {source_name!r}")
        selected = dict(indexed_sources[source_name][sample_id])
        selected["policy_name"] = f"domain_routed_{source_name}"
        selected["selected_source"] = source_name
        combined.append(selected)
    return combined


def render_report(summary: Mapping[str, Any], domain_sources: Mapping[str, str], files: Mapping[str, str], topk: int) -> str:
    lines = [
        "# Domain-Routed Ranker Outputs",
        "",
        "## Routing",
        "",
    ]
    for domain, source in sorted(domain_sources.items()):
        lines.append(f"- `{domain}` -> `{source}`")
    lines.extend(
        [
            "",
            "## Sources",
            "",
        ]
    )
    for name, path in sorted(files.items()):
        lines.append(f"- `{name}`: `{path}`")
    metric = summary["metric"]
    lines.extend(
        [
            "",
            "## Evaluation",
            "",
            f"- Recall@{topk}: `{metric[f'Recall@{topk}']:.6f}`",
            f"- CandidatePoolHitRate: `{metric['CandidatePoolHitRate']:.6f}`",
            f"- ConditionalRecall@50GivenPoolHit: `{metric['ConditionalRecall@50GivenPoolHit']:.6f}`",
            f"- AvgCandidatePoolMatchRank: `{format_optional_float(metric['AvgCandidatePoolMatchRank'])}`",
            f"- AvgPoolHitRankMissMatchRank: `{format_optional_float(metric['AvgPoolHitRankMissMatchRank'])}`",
            "",
            "## Next Target",
            "",
            str(summary["next_target"]),
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Combine analyzer-compatible ranker outputs by domain.")
    parser.add_argument("--source", action="append", required=True, help="Named source as NAME=PATH")
    parser.add_argument("--domain-source", action="append", required=True, help="Domain route as DOMAIN=SOURCE_NAME")
    parser.add_argument("--default-source", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    named_paths = dict(parse_named_path(value) for value in args.source)
    domain_sources = dict(parse_domain_source(value) for value in args.domain_source)
    sources = {name: read_rows(path) for name, path in named_paths.items()}
    combined = combine_rows_by_domain(sources, domain_sources, default_source=str(args.default_source))
    output_dir = Path(args.output_dir)
    ensure_empty_output_dir(output_dir)
    rows_path = output_dir / "cold_like_outputs.json"
    rows_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    metric = summarize_eval_rows(combined, topk=int(args.topk))
    summary = {
        "name": "DomainRoutedRankerOutputs",
        "domain_sources": domain_sources,
        "default_source": str(args.default_source),
        "source_files": {name: str(path) for name, path in named_paths.items()},
        "metric": metric,
        "files": {
            "cold_like_outputs": str(rows_path),
        },
        "next_target": NEXT_TARGET,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "report.md").write_text(render_report(summary, domain_sources, summary["source_files"], int(args.topk)), encoding="utf-8")
    print(json.dumps({"status": "ok", "output_dir": str(output_dir.resolve()), f"Recall@{args.topk}": metric[f"Recall@{args.topk}"]}, indent=2))


if __name__ == "__main__":
    main()
