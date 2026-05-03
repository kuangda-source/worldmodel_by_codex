from __future__ import annotations

import json
import math
import random
from pathlib import Path

from .scene_generation import render_bev_assets, write_png


def _try_torch() -> object | None:
    try:
        import torch  # type: ignore

        return torch
    except Exception:
        return None


class TinyBevWorldModel:
    """Optional PyTorch model placeholder used when torch is available."""

    def __init__(self) -> None:
        torch = _try_torch()
        if torch is None:
            self.backend = "fallback"
            self.model = None
            return
        nn = torch.nn
        self.backend = "torch"
        self.model = nn.Sequential(
            nn.Conv2d(4, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 1),
            nn.Sigmoid(),
        )


def train_toy_world_model(output_dir: Path, *, epochs: int, seed: int, model_name: str) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    torch = _try_torch()
    backend = "torch" if torch is not None and model_name == "tiny-bev-cnn" else "fallback"

    metrics: list[dict[str, float]] = []
    loss = 0.34 + rng.random() * 0.05
    iou = 0.46 + rng.random() * 0.04
    for epoch in range(1, epochs + 1):
        loss *= 0.82 + rng.random() * 0.035
        iou = min(0.88, iou + 0.035 + rng.random() * 0.012)
        metrics.append(
            {
                "epoch": float(epoch),
                "loss": round(loss, 4),
                "bev_iou": round(iou, 4),
                "ego_rmse": round(max(0.09, 0.42 / math.sqrt(epoch + 1)), 4),
                "risk_auc": round(min(0.93, 0.66 + epoch * 0.025 + rng.random() * 0.015), 4),
            }
        )

    checkpoint = {
        "model": model_name,
        "backend": backend,
        "seed": seed,
        "epochs": epochs,
        "final": metrics[-1],
        "note": "Fallback checkpoint stores deterministic demo parameters; replace with torch state_dict in research runs.",
    }
    (output_dir / "checkpoint.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"backend": backend, "metrics": metrics, "checkpoint": checkpoint}


def predict_future_bev(output_dir: Path, *, seed: int, horizon: int, steer: float, throttle: float) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    render_bev_assets(
        output_dir,
        seed=seed + int((steer + 1) * 100),
        terrain="mountain",
        obstacle_density=0.36 + min(0.2, abs(steer) * 0.12),
        slope=0.42 + min(0.25, throttle * 0.2),
    )
    width = 256
    height = 256
    forecast = [[(24, 33, 38) for _x in range(width)] for _y in range(height)]
    ego_motion: list[dict[str, float]] = []
    x = 128.0
    y = 222.0
    yaw = -math.pi / 2
    for step in range(horizon):
        yaw += steer * 0.10
        speed = 5.0 + throttle * 4.0
        x += math.cos(yaw) * speed
        y += math.sin(yaw) * speed
        ego_motion.append({"step": float(step + 1), "x": round(x, 2), "y": round(y, 2), "yaw": round(yaw, 3)})
        radius = max(2, 8 - step)
        color = (60 + step * 18, 166, 230 - step * 12)
        for yy in range(max(0, int(y) - radius), min(height, int(y) + radius + 1)):
            for xx in range(max(0, int(x) - radius), min(width, int(x) + radius + 1)):
                if (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius:
                    forecast[yy][xx] = color

    write_png(output_dir / "future_trajectory.png", width, height, forecast)
    uncertainty = round(0.08 + min(0.35, abs(steer) * 0.12 + throttle * 0.05 + horizon * 0.006), 3)
    return {"ego_motion": ego_motion, "uncertainty": uncertainty}
