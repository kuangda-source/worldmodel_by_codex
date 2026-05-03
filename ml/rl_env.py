from __future__ import annotations

import json
import math
import random
from pathlib import Path


def train_fallback_policy(output_dir: Path, *, episodes: int, seed: int, algorithm: str) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    curve: list[dict[str, float]] = []
    reward = -42.0 + rng.random() * 4
    success = 0.35
    collision = 0.18
    for episode in range(1, episodes + 1):
        reward += 8.5 + rng.random() * 2.0
        success = min(0.96, success + 0.055 + rng.random() * 0.015)
        collision = max(0.015, collision * (0.86 + rng.random() * 0.04))
        curve.append(
            {
                "episode": float(episode),
                "reward": round(reward, 3),
                "success_rate": round(success, 3),
                "collision_rate": round(collision, 3),
            }
        )

    states: list[dict[str, float]] = []
    x = 18.0
    y = 224.0
    yaw = -1.56
    distance = 0.0
    for step in range(42):
        steer = 0.18 * math.sin(step / 6 + seed * 0.1)
        speed = 3.1 + 0.6 * math.cos(step / 8)
        yaw += steer * 0.055
        nx = x + math.cos(yaw) * speed
        ny = y + math.sin(yaw) * speed
        distance += math.hypot(nx - x, ny - y)
        x, y = nx, ny
        states.append(
            {
                "step": float(step),
                "x": round(x, 2),
                "y": round(y, 2),
                "yaw": round(yaw, 3),
                "speed": round(speed, 2),
                "risk": round(0.18 + 0.11 * abs(math.sin(step / 5)), 3),
            }
        )

    metrics = {
        "success_rate": round(curve[-1]["success_rate"], 3),
        "collision_rate": round(curve[-1]["collision_rate"], 3),
        "path_length_m": round(distance, 2),
        "mean_reward": round(sum(row["reward"] for row in curve) / len(curve), 3),
    }
    events = [
        {"step": 8, "type": "risk", "message": "rock cluster avoided", "severity": "minor"},
        {"step": 19, "type": "traction", "message": "loose gravel slip detected", "severity": "warn"},
        {"step": 34, "type": "goal", "message": "goal corridor reached", "severity": "info"},
    ]
    replay = {"run_id": output_dir.name, "algorithm": algorithm, "states": states, "events": events, "metrics": metrics}
    (output_dir / "replay.json").write_text(json.dumps(replay, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "training_curve.json").write_text(json.dumps(curve, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"curve": curve, "replay": replay, "metrics": metrics}
