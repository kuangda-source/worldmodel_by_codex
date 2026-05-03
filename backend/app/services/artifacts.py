from __future__ import annotations

from pathlib import Path

from ..config import ARTIFACT_ROOT


def artifact_url(path: Path) -> str:
    rel = path.resolve().relative_to(ARTIFACT_ROOT.resolve()).as_posix()
    return f"/artifacts/{rel}"


def artifact_dir(*parts: str) -> Path:
    path = ARTIFACT_ROOT.joinpath(*parts)
    path.mkdir(parents=True, exist_ok=True)
    return path
