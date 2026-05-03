from __future__ import annotations

import sys
from uuid import uuid4

from ..config import REPO_ROOT
from ..provenance import mock, synthetic
from ..schemas import WorldModelPredictRequest, WorldModelPredictResponse, WorldModelTrainRequest, WorldModelTrainResponse
from .artifacts import artifact_dir, artifact_url
from .runs import record_run

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.world_model import predict_future_bev, train_toy_world_model  # noqa: E402


def train_world_model(payload: WorldModelTrainRequest) -> WorldModelTrainResponse:
    run_id = f"wm_{uuid4().hex[:8]}"
    out_dir = artifact_dir("world_model", run_id)
    result = train_toy_world_model(out_dir, epochs=payload.epochs, seed=payload.seed, model_name=payload.model)
    response = WorldModelTrainResponse(
        run_id=run_id,
        model=payload.model,
        checkpoint_url=artifact_url(out_dir / "checkpoint.json"),
        metrics=result["metrics"],
        backend=str(result["backend"]),
        provenance=synthetic("toy BEV world-model training", model=payload.model, data="synthetic BEV sequences"),
    )
    record_run(
        run_id=run_id,
        kind="world_model_train",
        name=f"World model {payload.model}",
        sequence_id=payload.sequence_id,
        provenance=response.provenance,
        metrics={"curve": response.metrics, "epochs": payload.epochs},
        artifacts={"checkpoint": response.checkpoint_url},
        config=payload.model_dump(),
    )
    return response


def predict_world_model(payload: WorldModelPredictRequest) -> WorldModelPredictResponse:
    prediction_id = f"pred_{uuid4().hex[:8]}"
    out_dir = artifact_dir("predictions", prediction_id)
    steer = float(payload.action.get("steer", 0.0))
    throttle = float(payload.action.get("throttle", 0.4))
    result = predict_future_bev(out_dir, seed=payload.seed, horizon=payload.horizon, steer=steer, throttle=throttle)
    assets = {
        "occupancy": artifact_url(out_dir / "occupancy.png"),
        "traversability": artifact_url(out_dir / "traversability.png"),
        "risk": artifact_url(out_dir / "risk.png"),
        "heightmap": artifact_url(out_dir / "heightmap.png"),
        "future_trajectory": artifact_url(out_dir / "future_trajectory.png"),
    }
    response = WorldModelPredictResponse(
        prediction_id=prediction_id,
        assets=assets,
        ego_motion=result["ego_motion"],
        uncertainty=float(result["uncertainty"]),
        provenance=mock("rule-based future BEV prediction", predictor="ml.world_model.predict_future_bev"),
    )
    record_run(
        run_id=prediction_id,
        kind="world_model_predict",
        name=f"World model prediction {payload.sequence_id}",
        sequence_id=payload.sequence_id,
        provenance=response.provenance,
        metrics={"uncertainty": response.uncertainty, "horizon": payload.horizon},
        artifacts=response.assets,
        config=payload.model_dump(),
    )
    return response
