from __future__ import annotations

import sys
from uuid import uuid4

from ..config import REPO_ROOT
from ..provenance import mock
from ..schemas import ReconstructionRequest, ReconstructionResponse
from .artifacts import artifact_dir, artifact_url
from .datasets import load_metadata
from .runs import record_run

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.reconstruction import reconstruct_mock_bev  # noqa: E402


def run_reconstruction(payload: ReconstructionRequest) -> ReconstructionResponse:
    metadata = load_metadata(payload.sequence_id)
    run_id = f"recon_{uuid4().hex[:8]}"
    out_dir = artifact_dir("reconstruction", run_id)
    metrics = reconstruct_mock_bev(out_dir, seed=payload.seed, terrain=metadata.terrain)
    assets = {
        "occupancy": artifact_url(out_dir / "occupancy.png"),
        "traversability": artifact_url(out_dir / "traversability.png"),
        "risk": artifact_url(out_dir / "risk.png"),
        "heightmap": artifact_url(out_dir / "heightmap.png"),
    }
    response = ReconstructionResponse(
        run_id=run_id,
        sequence_id=payload.sequence_id,
        method=payload.method,
        assets=assets,
        metrics=metrics,
        provenance=mock("mock BEV reconstruction", method=payload.method, required_next="LiDAR/depth projection adapter"),
    )
    record_run(
        run_id=run_id,
        kind="reconstruction",
        name=f"Reconstruction {payload.sequence_id}",
        sequence_id=payload.sequence_id,
        provenance=response.provenance,
        metrics=response.metrics,
        artifacts=response.assets,
        config=payload.model_dump(),
    )
    return response
