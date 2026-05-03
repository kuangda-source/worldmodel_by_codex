from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException

from ..config import ARTIFACT_ROOT, DATA_ROOT, REPO_ROOT
from ..provenance import terrain_source
from ..schemas import TraversabilityPredictRequest, TraversabilityPredictResponse, TraversabilityTrainRequest, TraversabilityTrainResponse
from .artifacts import artifact_dir, artifact_url
from .datasets import asset_url, get_sequence, sequence_root
from .runs import record_run

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.lightweight_segmentation import predict_tiny_mlp, train_tiny_mlp_model  # noqa: E402
from ml.traversability_model import predict_traversability, train_color_prototype_model  # noqa: E402


def _latest_model_path() -> tuple[str, Path] | None:
    models_root = ARTIFACT_ROOT / "traversability" / "models"
    if not models_root.exists():
        return None
    candidates = sorted((path for path in models_root.iterdir() if (path / "model.json").exists()), key=lambda path: path.stat().st_mtime)
    if not candidates:
        return None
    latest = candidates[-1]
    return latest.name, latest / "model.json"


def train_traversability(payload: TraversabilityTrainRequest) -> TraversabilityTrainResponse:
    run_id = f"trav_{uuid4().hex[:8]}"
    out_dir = artifact_dir("traversability", "models", run_id)
    dataset_root = Path(payload.dataset_root).resolve() if payload.dataset_root else sequence_root(payload.sequence_id)
    if payload.trainer == "color-prototype":
        model = train_color_prototype_model(out_dir, dataset_root, max_samples=payload.max_samples)
        backend = "pillow"
    else:
        model = train_tiny_mlp_model(
            out_dir,
            dataset_root,
            max_samples=payload.max_samples,
            max_pixels=payload.max_pixels,
            epochs=payload.epochs,
            learning_rate=payload.learning_rate,
        )
        backend = str(model.get("backend", "numpy"))

    metrics = {
        "sample_count": float(model["samples"]),
        "prototype_groups": float(len(model.get("prototypes", {}))),
        "labeled_pixels": float(sum(int(value) for value in model["class_counts"].values())),
    }
    response = TraversabilityTrainResponse(
        run_id=run_id,
        model=str(model["model"]),
        source_format=payload.source_format,
        backend=backend,
        model_url=artifact_url(out_dir / "model.json"),
        sample_count=int(model["samples"]),
        prototypes=model.get("prototypes", {}),
        groups=model["groups"] if isinstance(model["groups"], dict) else {"groups": model["groups"]},
        metrics=metrics,
        training_curve=model.get("training_curve", []),
        provenance=terrain_source(payload.sequence_id, payload.source_format),
    )
    record_run(
        run_id=run_id,
        kind="traversability_train",
        name=f"Terrain model {payload.trainer}",
        sequence_id=payload.sequence_id,
        provenance=response.provenance,
        metrics={**response.metrics, "curve": response.training_curve},
        artifacts={"model": response.model_url},
        config=payload.model_dump(),
    )
    return response


def predict_sequence_traversability(payload: TraversabilityPredictRequest) -> TraversabilityPredictResponse:
    if payload.model_id:
        model_path = ARTIFACT_ROOT / "traversability" / "models" / payload.model_id / "model.json"
        model_id = payload.model_id
    else:
        latest = _latest_model_path()
        if latest is None:
            train_payload = TraversabilityTrainRequest(sequence_id=payload.sequence_id, source_format="orwm-demo", max_samples=32)
            train_response = train_traversability(train_payload)
            model_id = train_response.run_id
            model_path = ARTIFACT_ROOT / "traversability" / "models" / model_id / "model.json"
        else:
            model_id, model_path = latest

    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Traversability model not found: {payload.model_id}")

    sequence = get_sequence(payload.sequence_id)
    if not sequence.frames:
        raise HTTPException(status_code=404, detail=f"No frames found for sequence: {payload.sequence_id}")
    frame_index = min(payload.frame_index, len(sequence.frames) - 1)
    frame_path = sorted((sequence_root(payload.sequence_id) / "images").glob("*.png"))[frame_index]

    prediction_id = f"trav_pred_{uuid4().hex[:8]}"
    out_dir = artifact_dir("traversability", "predictions", prediction_id)
    model_header = __import__("json").loads(model_path.read_text(encoding="utf-8"))
    if model_header.get("model") == "tiny-mlp-traversability":
        result = predict_tiny_mlp(out_dir, model_path, frame_path)
    else:
        result = predict_traversability(out_dir, model_path, frame_path)
    assets = {
        "semantic": artifact_url(out_dir / "semantic.png"),
        "traversability": artifact_url(out_dir / "traversability.png"),
        "risk": artifact_url(out_dir / "risk.png"),
        "overlay": artifact_url(out_dir / "overlay.png"),
    }
    response = TraversabilityPredictResponse(
        prediction_id=prediction_id,
        model_id=model_id,
        frame_url=asset_url(frame_path, DATA_ROOT),
        assets=assets,
        metrics=result["metrics"],
        counts=result["counts"],
        provenance=terrain_source(payload.sequence_id),
    )
    record_run(
        run_id=prediction_id,
        kind="traversability_predict",
        name=f"Terrain prediction {payload.sequence_id}",
        sequence_id=payload.sequence_id,
        provenance=response.provenance,
        metrics=response.metrics,
        artifacts={**response.assets, "frame": response.frame_url},
        config=payload.model_dump(),
    )
    return response
