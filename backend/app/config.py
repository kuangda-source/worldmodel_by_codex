from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.getenv("ORWM_DATA_ROOT", REPO_ROOT / "data" / "demo")).resolve()
ARTIFACT_ROOT = Path(os.getenv("ORWM_ARTIFACT_ROOT", REPO_ROOT / "artifacts")).resolve()
DB_PATH = Path(os.getenv("ORWM_DB_PATH", ARTIFACT_ROOT / "orwm.sqlite3")).resolve()


def ensure_runtime_dirs() -> None:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
