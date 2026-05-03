from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .traversability_model import GROUP_COLORS, GROUP_RISK, group_for_mask_color, discover_image_label_pairs

GROUPS = ["traversable", "caution", "obstacle", "ignore"]
GROUP_TO_ID = {name: index for index, name in enumerate(GROUPS)}


def _features_from_image(image: Image.Image) -> np.ndarray:
    arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    height, width, _channels = arr.shape
    yy, xx = np.meshgrid(
        np.linspace(0.0, 1.0, height, dtype=np.float32),
        np.linspace(0.0, 1.0, width, dtype=np.float32),
        indexing="ij",
    )
    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]
    exg = np.clip(2 * g - r - b, -1.0, 1.0)
    brightness = (r + g + b) / 3.0
    features = np.stack([r, g, b, xx, yy, exg, brightness], axis=-1)
    return features.reshape(-1, features.shape[-1])


def _targets_from_mask(mask: Image.Image) -> np.ndarray:
    rgb = np.asarray(mask.convert("RGB"), dtype=np.uint8)
    flat = rgb.reshape(-1, 3)
    targets = np.empty(flat.shape[0], dtype=np.int64)
    cache: dict[tuple[int, int, int], int] = {}
    for index, color in enumerate(flat):
        key = (int(color[0]), int(color[1]), int(color[2]))
        group_id = cache.get(key)
        if group_id is None:
            group_id = GROUP_TO_ID[group_for_mask_color(key)]
            cache[key] = group_id
        targets[index] = group_id
    return targets


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.maximum(exp.sum(axis=1, keepdims=True), 1e-8)


def _forward(features: np.ndarray, weights: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    hidden_linear = features @ weights["w1"] + weights["b1"]
    hidden = np.tanh(hidden_linear)
    logits = hidden @ weights["w2"] + weights["b2"]
    probs = _softmax(logits)
    return hidden, logits, probs


def _load_training_arrays(pairs: list[tuple[Path, Path]], *, max_pixels: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    per_image = max(128, max_pixels // max(1, len(pairs)))
    for image_path, label_path in pairs:
        image = Image.open(image_path).convert("RGB").resize((160, 90), Image.Resampling.BILINEAR)
        label = Image.open(label_path).convert("RGB").resize((160, 90), Image.Resampling.NEAREST)
        features = _features_from_image(image)
        targets = _targets_from_mask(label)
        buckets = [np.flatnonzero(targets == class_id) for class_id in range(len(GROUPS))]
        per_group = max(16, per_image // len(GROUPS))
        selected: list[np.ndarray] = []
        for bucket in buckets:
            if len(bucket) == 0:
                continue
            take = min(per_group, len(bucket))
            selected.append(rng.choice(bucket, size=take, replace=False))
        if selected:
            indices = np.concatenate(selected)
            if len(indices) > per_image:
                indices = rng.choice(indices, size=per_image, replace=False)
        else:
            take = min(per_image, len(targets))
            indices = rng.choice(len(targets), size=take, replace=False)
        xs.append(features[indices])
        ys.append(targets[indices])
    return np.concatenate(xs, axis=0), np.concatenate(ys, axis=0)


def train_tiny_mlp_model(
    output_dir: Path,
    dataset_root: Path,
    *,
    max_samples: int = 32,
    max_pixels: int = 24000,
    epochs: int = 12,
    learning_rate: float = 0.08,
    hidden_dim: int = 18,
    seed: int = 17,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = discover_image_label_pairs(dataset_root, max_samples=max_samples)
    if not pairs:
        raise FileNotFoundError(f"No image/semantic-label pairs found under {dataset_root}")

    features, targets = _load_training_arrays(pairs, max_pixels=max_pixels, seed=seed)
    rng = np.random.default_rng(seed)
    input_dim = features.shape[1]
    class_count = len(GROUPS)
    weights = {
        "w1": rng.normal(0.0, 0.22, size=(input_dim, hidden_dim)).astype(np.float32),
        "b1": np.zeros((hidden_dim,), dtype=np.float32),
        "w2": rng.normal(0.0, 0.18, size=(hidden_dim, class_count)).astype(np.float32),
        "b2": np.zeros((class_count,), dtype=np.float32),
    }

    y_onehot = np.eye(class_count, dtype=np.float32)[targets]
    curve: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        hidden, _logits, probs = _forward(features, weights)
        loss = float(-np.log(np.maximum(probs[np.arange(len(targets)), targets], 1e-8)).mean())
        pred = probs.argmax(axis=1)
        acc = float((pred == targets).mean())

        grad_logits = (probs - y_onehot) / len(targets)
        grad_w2 = hidden.T @ grad_logits
        grad_b2 = grad_logits.sum(axis=0)
        grad_hidden = (grad_logits @ weights["w2"].T) * (1.0 - hidden**2)
        grad_w1 = features.T @ grad_hidden
        grad_b1 = grad_hidden.sum(axis=0)

        weights["w1"] -= learning_rate * grad_w1.astype(np.float32)
        weights["b1"] -= learning_rate * grad_b1.astype(np.float32)
        weights["w2"] -= learning_rate * grad_w2.astype(np.float32)
        weights["b2"] -= learning_rate * grad_b2.astype(np.float32)
        curve.append({"epoch": float(epoch), "loss": round(loss, 5), "pixel_acc": round(acc, 5)})

    class_counts: dict[str, int] = defaultdict(int)
    for target in targets:
        class_counts[GROUPS[int(target)]] += 1

    model = {
        "model": "tiny-mlp-traversability",
        "version": "0.3",
        "backend": "numpy",
        "dataset_root": str(dataset_root),
        "samples": len(pairs),
        "max_pixels": int(len(targets)),
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "groups": GROUPS,
        "weights": {name: value.round(6).tolist() for name, value in weights.items()},
        "class_counts": dict(class_counts),
        "training_curve": curve,
    }
    (output_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    return model


def _weights_from_model(model: dict[str, Any]) -> dict[str, np.ndarray]:
    raw = model["weights"]
    return {name: np.asarray(value, dtype=np.float32) for name, value in raw.items()}


def _risk_color(risk: float) -> tuple[int, int, int]:
    risk = max(0.0, min(1.0, risk))
    return (int(50 + risk * 190), int(180 - risk * 105), int(74 - risk * 34))


def predict_tiny_mlp(output_dir: Path, model_path: Path, image_path: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = json.loads(model_path.read_text(encoding="utf-8"))
    weights = _weights_from_model(model)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    features = _features_from_image(image)

    class_ids = np.empty((features.shape[0],), dtype=np.int64)
    confidence = np.empty((features.shape[0],), dtype=np.float32)
    chunk_size = 65536
    for start in range(0, features.shape[0], chunk_size):
        end = min(start + chunk_size, features.shape[0])
        _hidden, _logits, probs = _forward(features[start:end], weights)
        class_ids[start:end] = probs.argmax(axis=1)
        confidence[start:end] = probs.max(axis=1)

    src = np.asarray(image, dtype=np.uint8)
    semantic = np.zeros_like(src)
    traversability = np.zeros_like(src)
    risk_img = np.zeros_like(src)
    overlay = src.copy()
    counts: dict[str, int] = defaultdict(int)

    class_grid = class_ids.reshape(height, width)
    confidence_grid = confidence.reshape(height, width)
    for group_id, group in enumerate(GROUPS):
        mask = class_grid == group_id
        counts[group] = int(mask.sum())
        color = np.asarray(GROUP_COLORS[group], dtype=np.uint8)
        semantic[mask] = color
        score = 1.0 - GROUP_RISK[group]
        traversability[mask] = np.asarray((int(220 - score * 130), int(80 + score * 145), 76), dtype=np.uint8)
        risk_img[mask] = np.asarray(_risk_color(GROUP_RISK[group]), dtype=np.uint8)
        overlay[mask] = (src[mask].astype(np.float32) * 0.62 + color.astype(np.float32) * 0.38).astype(np.uint8)

    Image.fromarray(semantic).save(output_dir / "semantic.png")
    Image.fromarray(traversability).save(output_dir / "traversability.png")
    Image.fromarray(risk_img).save(output_dir / "risk.png")
    Image.fromarray(overlay).save(output_dir / "overlay.png")

    total = max(1, width * height)
    risk_score = sum(counts[group] * GROUP_RISK[group] for group in GROUPS) / total
    return {
        "metrics": {
            "traversable_ratio": round(counts["traversable"] / total, 4),
            "caution_ratio": round(counts["caution"] / total, 4),
            "obstacle_ratio": round(counts["obstacle"] / total, 4),
            "risk_score": round(risk_score, 4),
            "confidence": round(float(confidence_grid.mean()), 4),
        },
        "counts": dict(counts),
    }
