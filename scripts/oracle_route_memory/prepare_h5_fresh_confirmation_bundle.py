#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.oracle_route_memory.validate_locked_policy_manifest import validate_manifest

try:
    from scripts.oracle_route_memory.handoff_io import (
        ensure_empty_output_dir,
        repo_relative_required as repo_relative,
        resolve_path_under_repo_root as resolve_repo_path,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script execution fallback
    from handoff_io import (
        ensure_empty_output_dir,
        repo_relative_required as repo_relative,
        resolve_path_under_repo_root as resolve_repo_path,
    )


DEFAULT_ARTIFACTS = (
    Path("experiments/h5-candidate-level-source-reranker/README.md"),
    Path("experiments/h5-candidate-level-source-reranker/handoff_index.json"),
    Path("experiments/h5-candidate-level-source-reranker/repro_commands.md"),
    Path("experiments/h5-candidate-level-source-reranker/validation_comparison.md"),
    Path("experiments/h5-candidate-level-source-reranker/fresh_confirmation_checklist.md"),
    Path("experiments/h5-candidate-level-source-reranker/fresh_confirmation_bundle.md"),
    Path("to_human/h5_paper_facing_validation_summary_20260608.md"),
    Path("to_human/h5_fresh_confirmation_handoff_summary_20260608.md"),
    Path("research-state.yaml"),
    Path("research-log.md"),
)

CLAIM_GUARDRAILS = (
    "Validation-only preparation bundle; not a fresh blind-confirmation result.",
    "This script validates existing H5-D validation outputs and copies documentation artifacts only.",
    "Do not read or reinterpret consumed protocol-v3 blind-confirmation labels for H5-D.",
    "Before any blind-style claim, record a clearly fresh/unconsumed split manifest path and hash.",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def copy_repo_artifact(path: Path, *, repo_root: Path, bundle_dir: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"Expected a file artifact, got: {path}")
    relative = repo_relative(path, repo_root)
    copied = bundle_dir / "artifacts" / relative
    copied.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, copied)
    source_hash = sha256_file(path)
    copied_hash = sha256_file(copied)
    if source_hash != copied_hash:
        raise ValueError(f"Hash mismatch after copying {path}")
    return {
        "source_path": relative.as_posix(),
        "bundle_path": copied.relative_to(bundle_dir).as_posix(),
        "sha256": source_hash,
        "size_bytes": path.stat().st_size,
    }


def dedupe_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[str] = set()
    result: list[Path] = []
    for path in paths:
        key = path.as_posix()
        if key not in seen:
            seen.add(key)
            result.append(path)
    return result


def render_readme(bundle_manifest: Mapping[str, Any]) -> str:
    metric = bundle_manifest["validation"]["metric"]
    artifact_lines = [
        f"- `{artifact['bundle_path']}` from `{artifact['source_path']}`"
        for artifact in bundle_manifest["artifacts"]
    ]
    guardrail_lines = [f"- {line}" for line in bundle_manifest["guardrails"]]
    return "\n".join(
        [
            "# H5-D Fresh Confirmation Prep Bundle",
            "",
            f"Created UTC: `{bundle_manifest['created_at_utc']}`",
            "",
            "## Claim Boundary",
            "",
            str(bundle_manifest["claim_boundary"]),
            "",
            "## Guardrails",
            "",
            *guardrail_lines,
            "",
            "## Locked Validation Check",
            "",
            f"- sample_count: `{metric['sample_count']}`",
            f"- hits_at_50: `{metric['hits_at_50']}`",
            f"- Recall@50: `{metric['Recall@50']:.6f}`",
            f"- CandidatePoolHitRate: `{metric['CandidatePoolHitRate']:.6f}`",
            "",
            "## Bundle Summary",
            "",
            f"- artifact_count: `{bundle_manifest.get('artifact_count', len(bundle_manifest['artifacts']))}`",
            f"- validator_status: `{bundle_manifest['validation']['status']}`",
            "",
            "## Copied Artifacts",
            "",
            *artifact_lines,
            "",
            "## Next Target",
            "",
            str(bundle_manifest["next_target"]),
            "",
        ]
    )


def build_bundle(
    manifest_path: Path,
    *,
    repo_root: Path,
    output_dir: Path,
    artifacts: Sequence[Path] | None = None,
    topk: int = 50,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    manifest_path = resolve_repo_path(repo_root, manifest_path)
    output_dir = resolve_repo_path(repo_root, output_dir)
    repo_relative(output_dir, repo_root)
    ensure_empty_output_dir(output_dir)

    validation = validate_manifest(manifest_path, repo_root=repo_root, topk=topk)
    write_json(output_dir / "validator_output.json", validation)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    artifact_paths = [manifest_path]
    for artifact in artifacts if artifacts is not None else DEFAULT_ARTIFACTS:
        artifact_paths.append(resolve_repo_path(repo_root, artifact))
    copied_artifacts = [copy_repo_artifact(path, repo_root=repo_root, bundle_dir=output_dir) for path in dedupe_paths(artifact_paths)]

    bundle_manifest = {
        "name": "h5_fresh_confirmation_prep_bundle",
        "created_at_utc": utc_now(),
        "repo_root": str(repo_root),
        "locked_policy_name": manifest.get("name"),
        "locked_manifest": repo_relative(manifest_path, repo_root).as_posix(),
        "claim_boundary": manifest.get(
            "claim_boundary",
            "Validation-only. Not a model-performance claim until locked before a fresh blind confirmation.",
        ),
        "guardrails": list(CLAIM_GUARDRAILS),
        "validation": validation,
        "artifact_count": len(copied_artifacts),
        "artifacts": copied_artifacts,
        "files": {
            "bundle_manifest": "bundle_manifest.json",
            "validator_output": "validator_output.json",
            "readme": "README.md",
        },
        "next_target": (
            "Use this bundle as the immutable H5-D input package for a future fresh confirmation run; "
            "record the fresh split manifest path and hash before scoring any fresh labels."
        ),
    }
    write_json(output_dir / "bundle_manifest.json", bundle_manifest)
    (output_dir / "README.md").write_text(render_readme(bundle_manifest), encoding="utf-8")
    return bundle_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a validation-only H5-D fresh-confirmation bundle.")
    parser.add_argument("--manifest", required=True, help="Locked H5-D manifest path, relative to repo root or absolute.")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--artifact", action="append", default=None, help="Additional or replacement artifact path. Repeatable.")
    parser.add_argument("--topk", type=int, default=50)
    args = parser.parse_args()

    artifact_paths = [Path(value) for value in args.artifact] if args.artifact is not None else None
    repo_root = Path(args.repo_root).resolve()
    output_dir = resolve_repo_path(repo_root, args.output_dir)
    result = build_bundle(
        Path(args.manifest),
        repo_root=repo_root,
        output_dir=output_dir,
        artifacts=artifact_paths,
        topk=int(args.topk),
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "output_dir": str(output_dir),
                "bundle_manifest": str(output_dir / "bundle_manifest.json"),
                "Recall@50": result["validation"]["metric"]["Recall@50"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
