from __future__ import annotations

import sys
from pathlib import Path

from ..config import DATA_ROOT, REPO_ROOT
from ..schemas import RUGDImportRequest, RUGDImportResponse, TartanDriveImportRequest, TartanDriveImportResponse
from .datasets import scan_dataset

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.rugd_adapter import import_rugd_sequence  # noqa: E402
from ml.tartandrive_adapter import import_tartandrive_sequence  # noqa: E402


def import_rugd_dataset(payload: RUGDImportRequest) -> RUGDImportResponse:
    result = import_rugd_sequence(
        Path(payload.root_path).resolve(),
        DATA_ROOT,
        sequence_id=payload.sequence_id,
        max_samples=payload.max_samples,
        overwrite=payload.overwrite,
    )
    return RUGDImportResponse(**result, dataset=scan_dataset())


def import_tartandrive_dataset(payload: TartanDriveImportRequest) -> TartanDriveImportResponse:
    result = import_tartandrive_sequence(
        Path(payload.root_path).resolve(),
        DATA_ROOT,
        sequence_id=payload.sequence_id,
        max_samples=payload.max_samples,
        overwrite=payload.overwrite,
    )
    return TartanDriveImportResponse(**result, dataset=scan_dataset())
