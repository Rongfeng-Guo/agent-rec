from __future__ import annotations

from pathlib import Path

import pytest

from scripts.oracle_route_memory.handoff_io import (
    ensure_empty_output_dir,
    repo_relative_or_absolute,
    repo_relative_required,
    resolve_output_dir,
    resolve_path_under_repo_root,
    resolve_repo_path,
)


def test_ensure_empty_output_dir_creates_missing_and_accepts_empty(tmp_path) -> None:
    output_dir = tmp_path / "new" / "out"

    assert ensure_empty_output_dir(output_dir) == output_dir
    assert output_dir.is_dir()
    assert ensure_empty_output_dir(output_dir) == output_dir


def test_ensure_empty_output_dir_refuses_non_empty_directory(tmp_path) -> None:
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-empty output directory"):
        ensure_empty_output_dir(output_dir)

    assert (output_dir / "old.txt").read_text(encoding="utf-8") == "keep"


def test_ensure_empty_output_dir_refuses_file_path(tmp_path) -> None:
    output_path = tmp_path / "out.txt"
    output_path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(FileExistsError, match="non-directory output path"):
        ensure_empty_output_dir(output_path)


def test_resolve_output_dir_uses_repo_root_only_for_relative_paths(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    absolute = tmp_path / "absolute"

    assert resolve_output_dir("outputs/run", repo_root) == repo_root / "outputs" / "run"
    assert resolve_output_dir(absolute, repo_root) == absolute
    assert resolve_output_dir("outputs/run") == Path("outputs/run")


def test_resolve_repo_path_handles_optional_relative_and_absolute_paths(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    absolute = tmp_path / "config.json"

    assert resolve_repo_path("configs/config.json", repo_root) == repo_root / "configs" / "config.json"
    assert resolve_repo_path(absolute, repo_root) == absolute
    assert resolve_repo_path("configs/config.json") == Path("configs/config.json")
    assert resolve_repo_path(None, repo_root) is None
    assert resolve_repo_path("", repo_root) is None


def test_resolve_path_under_repo_root_resolves_relative_and_absolute_paths(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    relative = "outputs/gate"
    absolute = tmp_path / "absolute"

    assert resolve_path_under_repo_root(repo_root, relative) == (repo_root / relative).resolve()
    assert resolve_path_under_repo_root(repo_root, absolute) == absolute.resolve()


def test_repo_relative_or_absolute_returns_relative_or_absolute_string(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    inside = repo_root / "docs" / "file.md"
    outside = tmp_path / "outside.md"
    inside.parent.mkdir(parents=True)
    inside.write_text("inside", encoding="utf-8")
    outside.write_text("outside", encoding="utf-8")

    assert repo_relative_or_absolute(inside, repo_root) == "docs/file.md"
    assert repo_relative_or_absolute(outside, repo_root) == str(outside.resolve())


def test_repo_relative_required_rejects_paths_outside_repo_root(tmp_path) -> None:
    repo_root = tmp_path / "repo"
    inside = repo_root / "docs" / "file.md"
    outside = tmp_path / "outside.md"
    inside.parent.mkdir(parents=True)
    inside.write_text("inside", encoding="utf-8")
    outside.write_text("outside", encoding="utf-8")

    assert repo_relative_required(inside, repo_root) == Path("docs/file.md")
    with pytest.raises(ValueError, match="outside repo_root"):
        repo_relative_required(outside, repo_root)
