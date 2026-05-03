from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw

from .reconstruction import reconstruct_mock_bev


STATE_CANDIDATES = ("poses.csv", "pose.csv", "states.csv", "state.csv", "odometry.csv", "trajectory.csv")
ACTION_CANDIDATES = ("actions.csv", "action.csv", "controls.csv", "control.csv", "commands.csv")
IMAGE_DIR_CANDIDATES = ("images", "image", "rgb", "camera", "frames", "front")


def _find_first(root: Path, names: Iterable[str]) -> Path | None:
    lower_names = {name.lower() for name in names}
    for path in root.rglob("*"):
        if path.is_file() and path.name.lower() in lower_names:
            return path
    return None


def _find_image_files(root: Path) -> list[Path]:
    for name in IMAGE_DIR_CANDIDATES:
        candidates = [path for path in root.rglob(name) if path.is_dir()]
        for folder in candidates:
            images = sorted(
                [
                    *folder.glob("*.png"),
                    *folder.glob("*.jpg"),
                    *folder.glob("*.jpeg"),
                ]
            )
            if images:
                return images
    return sorted([*root.glob("*.png"), *root.glob("*.jpg"), *root.glob("*.jpeg")])


def _pick(row: dict[str, str], names: tuple[str, ...], default: float = 0.0) -> float:
    lower = {key.lower(): value for key, value in row.items()}
    for name in names:
        value = lower.get(name.lower())
        if value not in (None, ""):
            return float(value)
    return default


def _read_state_rows(path: Path, max_samples: int) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            timestamp = _pick(row, ("timestamp", "time", "t", "sec"), index * 0.1)
            x = _pick(row, ("x", "pos_x", "position_x", "odom_x", "state_x"))
            y = _pick(row, ("y", "pos_y", "position_y", "odom_y", "state_y"))
            z = _pick(row, ("z", "pos_z", "position_z", "odom_z"), 0.0)
            yaw = _pick(row, ("yaw", "heading", "psi", "theta"), 0.0)
            pitch = _pick(row, ("pitch",), 0.0)
            roll = _pick(row, ("roll",), 0.0)
            speed = _pick(row, ("speed", "v", "vel", "velocity", "vx"), 0.0)
            rows.append({"timestamp": timestamp, "x": x, "y": y, "z": z, "yaw": yaw, "pitch": pitch, "roll": roll, "speed": speed})
            if len(rows) >= max_samples:
                break
    if len(rows) < 2:
        raise ValueError(f"Need at least two state rows in {path}")
    if all(abs(row["speed"]) < 1e-6 for row in rows):
        for index in range(1, len(rows)):
            dt = max(1e-3, rows[index]["timestamp"] - rows[index - 1]["timestamp"])
            rows[index]["speed"] = math.hypot(rows[index]["x"] - rows[index - 1]["x"], rows[index]["y"] - rows[index - 1]["y"]) / dt
        rows[0]["speed"] = rows[1]["speed"]
    return rows


def _read_action_rows(path: Path | None, state_rows: list[dict[str, float]]) -> list[dict[str, float]]:
    if path is None:
        return [{"timestamp": row["timestamp"], "steer": 0.0, "throttle": min(1.0, max(0.0, row["speed"] / 10.0)), "brake": 0.0} for row in state_rows]
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader):
            rows.append(
                {
                    "timestamp": _pick(row, ("timestamp", "time", "t", "sec"), state_rows[min(index, len(state_rows) - 1)]["timestamp"]),
                    "steer": _pick(row, ("steer", "steering", "steering_angle", "cmd_steer"), 0.0),
                    "throttle": _pick(row, ("throttle", "gas", "accel", "acceleration", "cmd_throttle"), 0.0),
                    "brake": _pick(row, ("brake", "cmd_brake"), 0.0),
                }
            )
            if len(rows) >= len(state_rows):
                break
    if len(rows) < len(state_rows):
        rows.extend({"timestamp": row["timestamp"], "steer": 0.0, "throttle": 0.0, "brake": 0.0} for row in state_rows[len(rows) :])
    return rows[: len(state_rows)]


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, float]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _copy_images(images: list[Path], sequence_dir: Path, count: int) -> list[dict[str, str]]:
    manifest: list[dict[str, str]] = []
    image_dir = sequence_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    if images:
        for index, src in enumerate(images[:count]):
            dst = image_dir / f"frame_{index:04d}.png"
            if src.suffix.lower() == ".png":
                shutil.copy2(src, dst)
            else:
                Image.open(src).convert("RGB").save(dst)
            manifest.append({"frame": f"frame_{index:04d}", "source_image": str(src), "image": str(dst.relative_to(sequence_dir))})
    else:
        for index in range(count):
            dst = image_dir / f"frame_{index:04d}.png"
            image = Image.new("RGB", (640, 360), (225, 232, 222))
            draw = ImageDraw.Draw(image)
            draw.rectangle((0, 230, 640, 360), fill=(122, 116, 86))
            draw.line((80, 360, 300, 210, 560, 360), fill=(213, 198, 151), width=18)
            draw.text((18, 18), f"TartanDrive-style frame {index:04d}", fill=(34, 50, 42))
            image.save(dst)
            manifest.append({"frame": f"frame_{index:04d}", "source_image": "", "image": str(dst.relative_to(sequence_dir)), "generated": "preview"})
    return manifest


def import_tartandrive_sequence(
    source_root: Path,
    target_data_root: Path,
    *,
    sequence_id: str = "tartandrive_mini",
    max_samples: int = 64,
    overwrite: bool = True,
) -> dict[str, object]:
    state_path = _find_first(source_root, STATE_CANDIDATES)
    if state_path is None:
        raise FileNotFoundError(f"No TartanDrive-style state/pose CSV found under {source_root}")
    action_path = _find_first(source_root, ACTION_CANDIDATES)
    state_rows = _read_state_rows(state_path, max_samples=max_samples)
    action_rows = _read_action_rows(action_path, state_rows)
    sequence_dir = target_data_root / "sequences" / sequence_id
    if sequence_dir.exists() and overwrite:
        shutil.rmtree(sequence_dir)
    for folder in ("images", "labels", "lidar", "occupancy"):
        (sequence_dir / folder).mkdir(parents=True, exist_ok=True)

    images = _find_image_files(source_root)
    manifest = _copy_images(images, sequence_dir, len(state_rows))
    for index, item in enumerate(manifest):
        item["source_state"] = str(state_path)
        if action_path:
            item["source_action"] = str(action_path)
        item["pose"] = f"poses.csv:{index + 2}"
        item["action"] = f"actions.csv:{index + 2}"

    poses = [
        {
            "timestamp": row["timestamp"],
            "x": row["x"],
            "y": row["y"],
            "z": row["z"],
            "yaw": row["yaw"],
            "pitch": row["pitch"],
            "roll": row["roll"],
            "speed": row["speed"],
        }
        for row in state_rows
    ]
    _write_csv(sequence_dir / "poses.csv", ["timestamp", "x", "y", "z", "yaw", "pitch", "roll", "speed"], poses)
    _write_csv(sequence_dir / "actions.csv", ["timestamp", "steer", "throttle", "brake"], action_rows)
    reconstruct_mock_bev(sequence_dir / "occupancy", seed=71, terrain="offroad")
    (sequence_dir / "calibration.json").write_text(
        json.dumps(
            {
                "camera": {"width": 0, "height": 0, "fx": None, "fy": None, "cx": None, "cy": None},
                "note": "TartanDrive-style import; calibration is preserved only if a future adapter maps source calibration files.",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    metadata = {
        "scene_id": sequence_id,
        "terrain": "offroad",
        "weather": "unknown",
        "time_of_day": "unknown",
        "difficulty": "real-trajectory-data",
        "tags": ["tartandrive", "public-dataset", "trajectory", "actions", "off-road"],
        "has_lidar": False,
        "has_occupancy": True,
        "vehicle_id": "ugv_default",
    }
    source_card = {
        "sequence_id": sequence_id,
        "dataset_name": "TartanDrive-style off-road trajectory/action import",
        "source_type": "public",
        "license": "Check the original TartanDrive dataset license/terms before redistribution.",
        "citation": "TartanDrive: A Large-Scale Dataset for Learning Off-Road Dynamics Models. Cite the original dataset/paper/site.",
        "homepage": "https://theairlab.org/tartandrive-dataset/",
        "importer": "ml.tartandrive_adapter.import_tartandrive_sequence",
        "importer_version": "orwm-tartandrive-importer-v0.1",
        "source_root": str(source_root),
        "target_root": str(sequence_dir),
        "manifest_path": str(sequence_dir / "manifest.json"),
        "sensors": ["front_rgb" if images else "generated_front_preview", "pose", "actions", "occupancy_or_bev"],
        "tags": metadata["tags"],
        "known_limitations": [
            "Importer currently supports CSV/flat mini subsets, not the full TartanDrive pickle/ROS tooling.",
            "Calibration is placeholder unless a future adapter maps source calibration files.",
            "Occupancy maps are generated by OR-WM mock BEV reconstruction.",
        ],
        "recommended_next": [
            "Use this sequence for real trajectory prediction smoke tests.",
            "Add full TartanDrive pickle conversion if training on the complete dataset.",
        ],
    }
    (sequence_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (sequence_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (sequence_dir / "source_card.json").write_text(json.dumps(source_card, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "sequence_id": sequence_id,
        "source_root": str(source_root),
        "target_root": str(sequence_dir),
        "imported_frames": len(state_rows),
        "manifest": str(sequence_dir / "manifest.json"),
    }
