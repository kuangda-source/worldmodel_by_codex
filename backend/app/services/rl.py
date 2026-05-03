from __future__ import annotations

import json
import sys
from uuid import uuid4

from fastapi import HTTPException

from ..config import ARTIFACT_ROOT, REPO_ROOT
from ..provenance import toy_env
from ..schemas import ReplayResponse, RlTrainRequest, RlTrainResponse
from .artifacts import artifact_dir, artifact_url
from .runs import record_run

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.rl_env import train_fallback_policy  # noqa: E402


def train_rl(payload: RlTrainRequest) -> RlTrainResponse:
    run_id = f"rl_{uuid4().hex[:8]}"
    out_dir = artifact_dir("rl", run_id)
    result = train_fallback_policy(out_dir, episodes=payload.episodes, seed=payload.seed, algorithm=payload.algorithm)
    response = RlTrainResponse(
        run_id=run_id,
        algorithm=payload.algorithm,
        metrics=result["metrics"],
        replay_url=artifact_url(out_dir / "replay.json"),
        training_curve=result["curve"],
        provenance=toy_env("toy off-road policy training", algorithm=payload.algorithm),
    )
    record_run(
        run_id=run_id,
        kind="rl_train",
        name=f"RL policy {payload.algorithm}",
        sequence_id=payload.scene_id,
        provenance=response.provenance,
        metrics={**response.metrics, "curve": response.training_curve},
        artifacts={"replay": response.replay_url},
        config=payload.model_dump(),
    )
    return response


def get_replay(run_id: str) -> ReplayResponse:
    path = ARTIFACT_ROOT / "rl" / run_id / "replay.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Replay not found: {run_id}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return ReplayResponse(
        run_id=run_id,
        states=payload["states"],
        events=payload["events"],
        metrics=payload["metrics"],
        provenance=toy_env("toy off-road policy replay", algorithm=str(payload.get("algorithm", "ppo-fallback"))),
    )
