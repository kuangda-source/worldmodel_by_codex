from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image

RGB = tuple[int, int, int]

RUGD_PALETTE: dict[str, RGB] = {
    "void": (0, 0, 0),
    "dirt": (108, 64, 20),
    "sand": (255, 229, 204),
    "grass": (0, 102, 0),
    "tree": (0, 255, 0),
    "pole": (0, 153, 153),
    "water": (0, 128, 255),
    "sky": (0, 0, 255),
    "vehicle": (255, 255, 0),
    "container": (255, 0, 127),
    "asphalt": (64, 64, 64),
    "gravel": (255, 128, 0),
    "building": (255, 0, 0),
    "mulch": (153, 76, 0),
    "rock-bed": (102, 102, 0),
    "log": (102, 0, 0),
    "bicycle": (0, 255, 128),
    "person": (204, 153, 255),
    "fence": (102, 0, 204),
    "bush": (255, 153, 204),
    "sign": (0, 102, 102),
    "rock": (153, 153, 153),
    "bridge": (102, 102, 255),
    "concrete": (153, 153, 255),
    "picnic-table": (102, 51, 0),
}

CLASS_TO_GROUP = {
    "dirt": "traversable",
    "sand": "traversable",
    "grass": "traversable",
    "asphalt": "traversable",
    "gravel": "traversable",
    "mulch": "caution",
    "concrete": "traversable",
    "water": "caution",
    "bush": "caution",
    "rock-bed": "obstacle",
    "rock": "obstacle",
    "tree": "obstacle",
    "pole": "obstacle",
    "vehicle": "obstacle",
    "container": "obstacle",
    "building": "obstacle",
    "log": "obstacle",
    "bicycle": "obstacle",
    "person": "obstacle",
    "fence": "obstacle",
    "sign": "obstacle",
    "bridge": "caution",
    "picnic-table": "obstacle",
    "sky": "ignore",
    "void": "ignore",
}

GROUP_COLORS: dict[str, RGB] = {
    "traversable": (42, 156, 87),
    "caution": (226, 171, 69),
    "obstacle": (204, 69, 58),
    "ignore": (75, 91, 104),
}

GROUP_RISK = {
    "traversable": 0.18,
    "caution": 0.58,
    "obstacle": 0.92,
    "ignore": 0.45,
}

DEFAULT_PROTOTYPES: dict[str, list[float]] = {
    "traversable": [105.0, 118.0, 78.0],
    "caution": [133.0, 116.0, 87.0],
    "obstacle": [92.0, 92.0, 83.0],
    "ignore": [178.0, 203.0, 216.0],
}


def _distance_sq(a: RGB, b: RGB | Iterable[float]) -> float:
    br, bg, bb = b
    return (a[0] - br) ** 2 + (a[1] - bg) ** 2 + (a[2] - bb) ** 2


def nearest_rugd_label(rgb: RGB) -> str:
    return min(RUGD_PALETTE, key=lambda name: _distance_sq(rgb, RUGD_PALETTE[name]))


def group_for_mask_color(rgb: RGB) -> str:
    return CLASS_TO_GROUP.get(nearest_rugd_label(rgb), "ignore")


def _find_demo_pairs(sequence_root: Path) -> list[tuple[Path, Path]]:
    images = sorted((sequence_root / "images").glob("*.png"))
    labels = sorted((sequence_root / "labels").glob("semantic_*.png"))
    if images and labels:
        pairs: list[tuple[Path, Path]] = []
        by_index = {label.stem.split("_")[-1]: label for label in labels}
        for image in images:
            index = image.stem.split("_")[-1]
            label = by_index.get(index)
            if label:
                pairs.append((image, label))
        return pairs
    return []


def discover_image_label_pairs(root: Path, max_samples: int = 64) -> list[tuple[Path, Path]]:
    if (root / "images").exists() and (root / "labels").exists():
        pairs = _find_demo_pairs(root)
        if pairs:
            return pairs[:max_samples]

    rugd_frames = root / "RUGD_frames-with-annotations"
    rugd_annotations = root / "RUGD_annotations"
    if rugd_frames.exists() and rugd_annotations.exists():
        pairs = []
        for label in sorted(rugd_annotations.rglob("*.png")):
            rel = label.relative_to(rugd_annotations)
            image = rugd_frames / rel
            if image.exists():
                pairs.append((image, label))
            if len(pairs) >= max_samples:
                return pairs
        if pairs:
            return pairs[:max_samples]

    sequence_root = root / "sequences"
    if sequence_root.exists():
        pairs: list[tuple[Path, Path]] = []
        for seq in sorted(path for path in sequence_root.iterdir() if path.is_dir()):
            pairs.extend(_find_demo_pairs(seq))
            if len(pairs) >= max_samples:
                return pairs[:max_samples]
        if pairs:
            return pairs[:max_samples]

    label_candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        and any(token in path.parent.name.lower() for token in ("label", "mask", "annotation"))
    ]
    pairs = []
    image_dirs = [path for path in root.rglob("*") if path.is_dir() and any(token in path.name.lower() for token in ("image", "frame", "rgb"))]
    for label in sorted(label_candidates):
        stem_tokens = [label.stem, label.stem.replace("semantic_", ""), label.stem.replace("_label", "")]
        match = None
        for image_dir in image_dirs:
            for token in stem_tokens:
                for suffix in (".png", ".jpg", ".jpeg"):
                    candidate = image_dir / f"{token}{suffix}"
                    if candidate.exists():
                        match = candidate
                        break
                if match:
                    break
            if match:
                break
        if match:
            pairs.append((match, label))
        if len(pairs) >= max_samples:
            break
    return pairs


def train_color_prototype_model(output_dir: Path, dataset_root: Path, *, max_samples: int = 64) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = discover_image_label_pairs(dataset_root, max_samples=max_samples)
    if not pairs:
        raise FileNotFoundError(f"No image/semantic-label pairs found under {dataset_root}")

    sums: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0])
    counts: dict[str, int] = defaultdict(int)
    class_counts: dict[str, int] = defaultdict(int)

    for image_path, label_path in pairs:
        image = Image.open(image_path).convert("RGB").resize((160, 90), Image.Resampling.BILINEAR)
        label = Image.open(label_path).convert("RGB").resize((160, 90), Image.Resampling.NEAREST)
        image_pixels = image.load()
        label_pixels = label.load()
        for y in range(image.height):
            for x in range(image.width):
                group = group_for_mask_color(label_pixels[x, y])
                class_counts[group] += 1
                if group == "ignore":
                    continue
                rgb = image_pixels[x, y]
                sums[group][0] += rgb[0]
                sums[group][1] += rgb[1]
                sums[group][2] += rgb[2]
                counts[group] += 1

    prototypes: dict[str, list[float]] = {}
    for group in ("traversable", "caution", "obstacle", "ignore"):
        if counts[group] > 0:
            prototypes[group] = [round(channel / counts[group], 3) for channel in sums[group]]
        else:
            prototypes[group] = DEFAULT_PROTOTYPES[group]

    model = {
        "model": "color-prototype-traversability",
        "version": "0.2",
        "dataset_root": str(dataset_root),
        "samples": len(pairs),
        "prototypes": prototypes,
        "class_counts": dict(class_counts),
        "groups": {
            "traversable": ["dirt", "sand", "grass", "asphalt", "gravel", "concrete"],
            "caution": ["mulch", "water", "bush", "bridge"],
            "obstacle": ["rock", "rock-bed", "tree", "pole", "vehicle", "container", "building", "log", "person", "fence"],
            "ignore": ["sky", "void"],
        },
    }
    (output_dir / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2), encoding="utf-8")
    return model


def load_model(model_path: Path) -> dict[str, object]:
    return json.loads(model_path.read_text(encoding="utf-8"))


def _classify_rgb(rgb: RGB, prototypes: dict[str, list[float]]) -> str:
    candidates = [group for group in ("traversable", "caution", "obstacle") if group in prototypes]
    return min(candidates, key=lambda group: _distance_sq(rgb, prototypes[group]))


def _risk_color(risk: float) -> RGB:
    risk = max(0.0, min(1.0, risk))
    return (int(50 + risk * 190), int(180 - risk * 105), int(74 - risk * 34))


def predict_traversability(output_dir: Path, model_path: Path, image_path: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    model = load_model(model_path)
    prototypes = model["prototypes"]
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    semantic = Image.new("RGB", (width, height))
    traversability = Image.new("RGB", (width, height))
    risk = Image.new("RGB", (width, height))
    overlay = image.copy()

    src = image.load()
    semantic_px = semantic.load()
    trav_px = traversability.load()
    risk_px = risk.load()
    overlay_px = overlay.load()
    counts: dict[str, int] = defaultdict(int)

    for y in range(height):
        for x in range(width):
            rgb = src[x, y]
            sky_bias = y < height * 0.28 and rgb[2] > rgb[1] and rgb[2] > rgb[0]
            group = "ignore" if sky_bias else _classify_rgb(rgb, prototypes)
            counts[group] += 1
            group_color = GROUP_COLORS[group]
            semantic_px[x, y] = group_color
            score = 1.0 - GROUP_RISK[group]
            trav_px[x, y] = (int(220 - score * 130), int(80 + score * 145), 76)
            risk_px[x, y] = _risk_color(GROUP_RISK[group])
            overlay_px[x, y] = tuple(int(src[x, y][i] * 0.62 + group_color[i] * 0.38) for i in range(3))  # type: ignore[index]

    semantic.save(output_dir / "semantic.png")
    traversability.save(output_dir / "traversability.png")
    risk.save(output_dir / "risk.png")
    overlay.save(output_dir / "overlay.png")

    total = max(1, width * height)
    traversable_ratio = counts["traversable"] / total
    obstacle_ratio = counts["obstacle"] / total
    caution_ratio = counts["caution"] / total
    risk_score = (
        counts["traversable"] * GROUP_RISK["traversable"]
        + counts["caution"] * GROUP_RISK["caution"]
        + counts["obstacle"] * GROUP_RISK["obstacle"]
        + counts["ignore"] * GROUP_RISK["ignore"]
    ) / total
    return {
        "metrics": {
            "traversable_ratio": round(traversable_ratio, 4),
            "caution_ratio": round(caution_ratio, 4),
            "obstacle_ratio": round(obstacle_ratio, 4),
            "risk_score": round(risk_score, 4),
            "confidence": round(1.0 - min(0.72, math.sqrt(obstacle_ratio + caution_ratio) * 0.38), 4),
        },
        "counts": dict(counts),
    }
