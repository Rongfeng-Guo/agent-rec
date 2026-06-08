from __future__ import annotations

from pathlib import Path


class OutputPathError(FileExistsError, ValueError):
    pass


class OutputDirectoryNotEmptyError(FileExistsError, ValueError):
    pass


def ensure_empty_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.exists():
        if not path.is_dir():
            raise OutputPathError(f"Refusing to write into non-directory output path: {path}")
        if any(path.iterdir()):
            raise OutputDirectoryNotEmptyError(
                "Refusing to write into non-empty output directory; "
                f"output directory already exists and is not empty: {path}"
            )
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_output_dir(output_dir: str | Path, repo_root: str | Path | None = None) -> Path:
    path = Path(output_dir)
    if path.is_absolute() or repo_root in (None, ""):
        return path
    return Path(repo_root) / path


def resolve_repo_path(path_value: str | Path | None, repo_root: str | Path | None = None) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(path_value)
    if path.is_absolute() or repo_root in (None, ""):
        return path
    return Path(repo_root) / path


def resolve_path_under_repo_root(repo_root: str | Path, value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(repo_root) / path
    return path.resolve()


def repo_relative_or_absolute(path: str | Path, repo_root: str | Path) -> str:
    resolved = Path(path).resolve()
    root = Path(repo_root).resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError:
        return str(resolved)


def repo_relative_required(path: str | Path, repo_root: str | Path) -> Path:
    resolved = Path(path).resolve()
    root = Path(repo_root).resolve()
    try:
        return resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"Refusing to bundle a path outside repo_root: {path}") from exc
