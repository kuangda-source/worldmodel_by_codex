from __future__ import annotations

import json

from ..config import ARTIFACT_ROOT
from ..provenance import placeholder, toy_env
from ..schemas import EvaluationResponse


def evaluate_run(run_id: str) -> EvaluationResponse:
    replay_path = ARTIFACT_ROOT / "rl" / run_id / "replay.json"
    if replay_path.exists():
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        metrics = replay["metrics"]
        events = replay["events"]
        return EvaluationResponse(
            run_id=run_id,
            success_rate=float(metrics["success_rate"]),
            collision_rate=float(metrics["collision_rate"]),
            path_length_m=float(metrics["path_length_m"]),
            terrain_risk=0.31,
            uncertainty=0.14,
            events=events,
            provenance=toy_env("toy off-road policy evaluation", run_id=run_id),
        )
    return EvaluationResponse(
        run_id=run_id,
        success_rate=None,
        collision_rate=None,
        path_length_m=None,
        terrain_risk=None,
        uncertainty=None,
        events=[],
        provenance=placeholder("missing run evaluation", required_next="Run a policy job before evaluating this run_id."),
    )
