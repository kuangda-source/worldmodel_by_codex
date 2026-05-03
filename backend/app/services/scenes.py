from __future__ import annotations

import sys
from uuid import uuid4

from ..config import REPO_ROOT
from ..provenance import synthetic
from ..schemas import SceneGenerateRequest, SceneGenerateResponse
from .artifacts import artifact_dir, artifact_url
from .runs import record_run

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.scene_generation import generate_scene_assets  # noqa: E402


def generate_scene(payload: SceneGenerateRequest) -> SceneGenerateResponse:
    scene_id = f"scene_{uuid4().hex[:8]}"
    out_dir = artifact_dir("scenes", scene_id)
    generate_scene_assets(
        out_dir,
        seed=payload.seed,
        terrain=payload.terrain,
        weather=payload.weather,
        task=payload.task,
        prompt=payload.prompt,
        obstacle_density=payload.obstacle_density,
        slope=payload.slope,
    )
    assets = {
        "front_view": artifact_url(out_dir / "front_view.png"),
        "occupancy": artifact_url(out_dir / "occupancy.png"),
        "traversability": artifact_url(out_dir / "traversability.png"),
        "risk": artifact_url(out_dir / "risk.png"),
        "heightmap": artifact_url(out_dir / "heightmap.png"),
    }
    metadata = __import__("json").loads((out_dir / "metadata.json").read_text(encoding="utf-8"))
    response = SceneGenerateResponse(
        scene_id=scene_id,
        seed=payload.seed,
        terrain=payload.terrain,
        weather=payload.weather,
        task=payload.task,
        prompt=payload.prompt,
        assets=assets,
        metrics=metadata["metrics"],
        provenance=synthetic("procedural scene generation", generator="ml.scene_generation.generate_scene_assets"),
    )
    record_run(
        run_id=scene_id,
        kind="scene_generation",
        name=f"{payload.terrain} {payload.task}",
        provenance=response.provenance,
        metrics=response.metrics,
        artifacts=response.assets,
        config=payload.model_dump(),
    )
    return response
