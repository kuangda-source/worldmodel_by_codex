from __future__ import annotations

import json
import sys
from uuid import uuid4

from fastapi import HTTPException

from ..config import ARTIFACT_ROOT, REPO_ROOT
from ..provenance import trajectory_source
from ..schemas import TrajectoryPredictRequest, TrajectoryPredictResponse, TrajectoryTrainRequest, TrajectoryTrainResponse
from .artifacts import artifact_dir, artifact_url
from .datasets import sequence_root
from .runs import record_run

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.trajectory_prediction import predict_tiny_traj_gru, train_tiny_traj_gru  # noqa: E402


def _latest_model_path() -> tuple[str, object] | None:
    models_root = ARTIFACT_ROOT / "trajectory" / "models"
    if not models_root.exists():
        return None
    candidates = sorted((path for path in models_root.iterdir() if (path / "model.pt").exists()), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None
    latest = candidates[-1]
    return latest.name, latest / "model.pt"


def train_trajectory(payload: TrajectoryTrainRequest) -> TrajectoryTrainResponse:
    run_id = f"traj_{uuid4().hex[:8]}"
    out_dir = artifact_dir("trajectory", "models", run_id)
    metadata = train_tiny_traj_gru(
        out_dir,
        sequence_root(payload.sequence_id),
        history=payload.history,
        horizon=payload.horizon,
        epochs=payload.epochs,
        hidden_dim=payload.hidden_dim,
        seed=payload.seed,
        augment=payload.augment,
    )
    response = TrajectoryTrainResponse(
        run_id=run_id,
        model=str(metadata["model"]),
        backend=str(metadata["backend"]),
        checkpoint_url=artifact_url(out_dir / "model.pt"),
        history=int(metadata["history"]),
        horizon=int(metadata["horizon"]),
        sample_count=int(metadata["samples"]),
        training_curve=metadata["training_curve"],
        provenance=trajectory_source(payload.sequence_id),
    )
    record_run(
        run_id=run_id,
        kind="trajectory_train",
        name=f"Trajectory model {payload.model}",
        sequence_id=payload.sequence_id,
        provenance=response.provenance,
        metrics={"curve": response.training_curve, "samples": response.sample_count},
        artifacts={"checkpoint": response.checkpoint_url},
        config=payload.model_dump(),
    )
    return response


def predict_trajectory(payload: TrajectoryPredictRequest) -> TrajectoryPredictResponse:
    if payload.model_id:
        model_id = payload.model_id
        model_path = ARTIFACT_ROOT / "trajectory" / "models" / model_id / "model.pt"
    else:
        latest = _latest_model_path()
        if latest is None:
            train_response = train_trajectory(TrajectoryTrainRequest(sequence_id=payload.sequence_id, epochs=40, augment=64))
            model_id = train_response.run_id
            model_path = ARTIFACT_ROOT / "trajectory" / "models" / model_id / "model.pt"
        else:
            model_id, model_path = latest
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Trajectory model not found: {payload.model_id}")

    prediction_id = f"traj_pred_{uuid4().hex[:8]}"
    out_dir = artifact_dir("trajectory", "predictions", prediction_id)
    result = predict_tiny_traj_gru(
        out_dir,
        sequence_root(payload.sequence_id),
        model_path,
        history=payload.history,
        horizon=payload.horizon,
        frame_index=payload.frame_index,
    )
    response = TrajectoryPredictResponse(
        prediction_id=prediction_id,
        model_id=model_id,
        assets={"trajectory": artifact_url(out_dir / "trajectory.png"), "data": artifact_url(out_dir / "trajectory.json")},
        observed=result["observed"],
        predicted=result["predicted"],
        ground_truth=result["ground_truth"],
        metrics=result["metrics"],
        provenance=trajectory_source(payload.sequence_id),
    )
    record_run(
        run_id=prediction_id,
        kind="trajectory_predict",
        name=f"Trajectory prediction {payload.sequence_id}",
        sequence_id=payload.sequence_id,
        provenance=response.provenance,
        metrics=response.metrics,
        artifacts=response.assets,
        config=payload.model_dump(),
    )
    return response
