from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


POSE_FIELDS = ["x", "y", "yaw", "speed", "pitch", "roll"]


def _try_torch() -> Any | None:
    try:
        import torch

        return torch
    except Exception:
        return None


def load_pose_rows(sequence_root: Path) -> list[dict[str, float]]:
    path = sequence_root / "poses.csv"
    if not path.exists():
        raise FileNotFoundError(f"poses.csv not found under {sequence_root}")
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append({name: float(row[name]) for name in ["timestamp", *POSE_FIELDS]})
    if len(rows) < 2:
        raise ValueError(f"Need at least two pose rows in {path}")
    return rows


def _synthetic_pose_sequence(length: int, seed: int) -> list[dict[str, float]]:
    rng = random.Random(seed)
    x = rng.uniform(-1.0, 1.0)
    y = rng.uniform(-1.0, 1.0)
    yaw = rng.uniform(-0.25, 0.25)
    speed = rng.uniform(3.5, 8.0)
    curvature = rng.uniform(-0.09, 0.09)
    rows: list[dict[str, float]] = []
    for step in range(length):
        yaw += curvature + 0.018 * math.sin(step / 5 + seed)
        speed = max(1.2, speed + rng.uniform(-0.05, 0.05))
        x += math.cos(yaw) * speed * 0.22
        y += math.sin(yaw) * speed * 0.22
        rows.append(
            {
                "timestamp": step * 0.1,
                "x": x,
                "y": y,
                "yaw": yaw,
                "speed": speed,
                "pitch": 0.04 * math.sin(step / 8),
                "roll": 0.06 * math.sin(step / 6 + 0.5),
            }
        )
    return rows


def _pose_features(rows: list[dict[str, float]]) -> list[list[float]]:
    return [[row[name] for name in POSE_FIELDS] for row in rows]


def _make_windows(rows: list[dict[str, float]], history: int, horizon: int) -> list[tuple[list[list[float]], list[list[float]]]]:
    features = _pose_features(rows)
    windows: list[tuple[list[list[float]], list[list[float]]]] = []
    for start in range(0, len(features) - history - horizon + 1):
        observed = features[start : start + history]
        future = features[start + history : start + history + horizon]
        anchor_x, anchor_y = observed[-1][0], observed[-1][1]
        windows.append(
            (
                [[value for value in row] for row in observed],
                [[row[0] - anchor_x, row[1] - anchor_y, row[2], row[3], row[4], row[5]] for row in future],
            )
        )
    return windows


def _collect_training_windows(sequence_root: Path, history: int, horizon: int, seed: int, augment: int) -> list[tuple[list[list[float]], list[list[float]]]]:
    rows = load_pose_rows(sequence_root)
    windows = _make_windows(rows, history, horizon)
    synthetic_length = max(history + horizon + 8, 32)
    for index in range(augment):
        windows.extend(_make_windows(_synthetic_pose_sequence(synthetic_length, seed + index * 13), history, horizon))
    if not windows:
        raise ValueError(f"No trajectory windows available: need at least {history + horizon} poses")
    return windows


def _normalize_windows(windows: list[tuple[list[list[float]], list[list[float]]]]) -> tuple[Any, Any, dict[str, list[float]]]:
    import numpy as np

    x = np.asarray([item[0] for item in windows], dtype=np.float32)
    y = np.asarray([item[1] for item in windows], dtype=np.float32)
    x_mean = x.reshape(-1, x.shape[-1]).mean(axis=0)
    x_std = x.reshape(-1, x.shape[-1]).std(axis=0) + 1e-4
    y_mean = y.reshape(-1, y.shape[-1]).mean(axis=0)
    y_std = y.reshape(-1, y.shape[-1]).std(axis=0) + 1e-4
    return (x - x_mean) / x_std, (y - y_mean) / y_std, {
        "x_mean": x_mean.round(6).tolist(),
        "x_std": x_std.round(6).tolist(),
        "y_mean": y_mean.round(6).tolist(),
        "y_std": y_std.round(6).tolist(),
    }


def _build_model(torch: Any, input_dim: int, hidden_dim: int, horizon: int, output_dim: int) -> Any:
    nn = torch.nn

    class TinyTrajGRU(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.GRU(input_dim, hidden_dim, batch_first=True)
            self.head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, horizon * output_dim))

        def forward(self, observed: Any) -> Any:
            _encoded, hidden = self.encoder(observed)
            raw = self.head(hidden[-1])
            return raw.reshape(observed.shape[0], horizon, output_dim)

    return TinyTrajGRU()


def train_tiny_traj_gru(
    output_dir: Path,
    sequence_root: Path,
    *,
    history: int = 6,
    horizon: int = 8,
    epochs: int = 60,
    hidden_dim: int = 48,
    seed: int = 23,
    augment: int = 96,
) -> dict[str, Any]:
    torch = _try_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for tiny-traj-gru. Use the orwm311 conda environment.")
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    windows = _collect_training_windows(sequence_root, history, horizon, seed, augment)
    x_norm, y_norm, norm = _normalize_windows(windows)
    x_tensor = torch.tensor(x_norm, dtype=torch.float32)
    y_tensor = torch.tensor(y_norm, dtype=torch.float32)
    model = _build_model(torch, len(POSE_FIELDS), hidden_dim, horizon, len(POSE_FIELDS))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.012)
    loss_fn = torch.nn.SmoothL1Loss()
    curve: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        prediction = model(x_tensor)
        loss = loss_fn(prediction, y_tensor)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            denorm = prediction * torch.tensor(norm["y_std"]) + torch.tensor(norm["y_mean"])
            target = y_tensor * torch.tensor(norm["y_std"]) + torch.tensor(norm["y_mean"])
            ade = torch.linalg.norm(denorm[..., :2] - target[..., :2], dim=-1).mean()
            fde = torch.linalg.norm(denorm[:, -1, :2] - target[:, -1, :2], dim=-1).mean()
        curve.append({"epoch": float(epoch), "loss": round(float(loss.item()), 6), "ade": round(float(ade.item()), 4), "fde": round(float(fde.item()), 4)})

    checkpoint_path = output_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "norm": norm, "history": history, "horizon": horizon, "hidden_dim": hidden_dim, "fields": POSE_FIELDS}, checkpoint_path)
    metadata = {
        "model": "tiny-traj-gru",
        "backend": "torch",
        "version": "0.1",
        "history": history,
        "horizon": horizon,
        "hidden_dim": hidden_dim,
        "samples": len(windows),
        "checkpoint": "model.pt",
        "training_curve": curve,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def _draw_trajectory(output_dir: Path, observed: list[dict[str, float]], predicted: list[dict[str, float]], ground_truth: list[dict[str, float]]) -> None:
    width, height = 520, 360
    image = Image.new("RGB", (width, height), (237, 241, 235))
    draw = ImageDraw.Draw(image)
    points = [(row["x"], row["y"]) for row in [*observed, *predicted, *ground_truth]]
    min_x = min(x for x, _y in points)
    max_x = max(x for x, _y in points)
    min_y = min(y for _x, y in points)
    max_y = max(y for _x, y in points)
    pad_x = max(2.0, (max_x - min_x) * 0.18)
    pad_y = max(2.0, (max_y - min_y) * 0.45)

    def project(point: tuple[float, float]) -> tuple[int, int]:
        x, y = point
        px = int((x - min_x + pad_x) / max(1e-6, max_x - min_x + 2 * pad_x) * width)
        py = int(height - (y - min_y + pad_y) / max(1e-6, max_y - min_y + 2 * pad_y) * height)
        return px, py

    for gx in range(0, width, 40):
        draw.line([(gx, 0), (gx, height)], fill=(218, 225, 216))
    for gy in range(0, height, 40):
        draw.line([(0, gy), (width, gy)], fill=(218, 225, 216))

    def draw_path(rows: list[dict[str, float]], color: tuple[int, int, int], width_px: int) -> None:
        path = [project((row["x"], row["y"])) for row in rows]
        if len(path) > 1:
            draw.line(path, fill=color, width=width_px, joint="curve")
        for px, py in path:
            draw.ellipse((px - 4, py - 4, px + 4, py + 4), fill=color)

    draw_path(observed, (34, 88, 150), 5)
    draw_path(ground_truth, (37, 128, 78), 4)
    draw_path(predicted, (214, 141, 38), 5)
    image.save(output_dir / "trajectory.png")


def predict_tiny_traj_gru(
    output_dir: Path,
    sequence_root: Path,
    model_path: Path,
    *,
    history: int | None = None,
    horizon: int | None = None,
    frame_index: int | None = None,
) -> dict[str, Any]:
    torch = _try_torch()
    if torch is None:
        raise RuntimeError("PyTorch is required for tiny-traj-gru. Use the orwm311 conda environment.")
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    history = int(history or checkpoint["history"])
    horizon = int(horizon or checkpoint["horizon"])
    rows = load_pose_rows(sequence_root)
    if len(rows) < history:
        raise ValueError(f"Need at least {history} poses for prediction")
    start = 0 if frame_index is None else max(0, min(frame_index, len(rows) - history))
    observed_rows = rows[start : start + history]
    ground_truth_rows = rows[start + history : start + history + horizon]
    anchor_x, anchor_y = observed_rows[-1]["x"], observed_rows[-1]["y"]
    observed = [[row[name] for name in POSE_FIELDS] for row in observed_rows]
    norm = checkpoint["norm"]
    x_norm = torch.tensor([[(value - norm["x_mean"][idx]) / norm["x_std"][idx] for idx, value in enumerate(row)] for row in observed], dtype=torch.float32).unsqueeze(0)
    model = _build_model(torch, len(POSE_FIELDS), int(checkpoint["hidden_dim"]), horizon, len(POSE_FIELDS))
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    with torch.no_grad():
        predicted_norm = model(x_norm)[0]
        predicted = predicted_norm * torch.tensor(norm["y_std"]) + torch.tensor(norm["y_mean"])

    predicted_rows: list[dict[str, float]] = []
    for step, row in enumerate(predicted.tolist(), start=1):
        predicted_rows.append(
            {
                "step": float(step),
                "x": round(anchor_x + row[0], 3),
                "y": round(anchor_y + row[1], 3),
                "yaw": round(row[2], 4),
                "speed": round(row[3], 3),
                "pitch": round(row[4], 4),
                "roll": round(row[5], 4),
            }
        )
    observed_out = [{"step": float(index), **{name: round(row[name], 4) for name in ["x", "y", "yaw", "speed"]}} for index, row in enumerate(observed_rows)]
    gt_out = [{"step": float(index + 1), **{name: round(row[name], 4) for name in ["x", "y", "yaw", "speed"]}} for index, row in enumerate(ground_truth_rows)]
    if gt_out:
        ade = sum(math.hypot(predicted_rows[i]["x"] - gt_out[i]["x"], predicted_rows[i]["y"] - gt_out[i]["y"]) for i in range(min(len(predicted_rows), len(gt_out)))) / min(len(predicted_rows), len(gt_out))
        fde = math.hypot(predicted_rows[min(len(predicted_rows), len(gt_out)) - 1]["x"] - gt_out[min(len(predicted_rows), len(gt_out)) - 1]["x"], predicted_rows[min(len(predicted_rows), len(gt_out)) - 1]["y"] - gt_out[min(len(predicted_rows), len(gt_out)) - 1]["y"])
    else:
        ade = 0.0
        fde = 0.0

    _draw_trajectory(output_dir, observed_out, predicted_rows, gt_out)
    payload = {"observed": observed_out, "predicted": predicted_rows, "ground_truth": gt_out, "metrics": {"ade": round(ade, 4), "fde": round(fde, 4), "horizon": float(horizon)}}
    (output_dir / "trajectory.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
