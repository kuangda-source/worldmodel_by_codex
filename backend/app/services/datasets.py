from __future__ import annotations

import json
import os
from pathlib import Path

from ..config import DATA_ROOT
from ..schemas import DatasetSourceCard, DatasetSummary, QualityItem, SequenceDetail, SequenceMetadata, SequenceQuality


IMPORTER_VERSION = "orwm-importers-v0.1"


def sequence_root(sequence_id: str, root: Path = DATA_ROOT) -> Path:
    return root / "sequences" / sequence_id


def _metadata_path(sequence_id: str, root: Path = DATA_ROOT) -> Path:
    return sequence_root(sequence_id, root) / "metadata.json"


def load_metadata(sequence_id: str, root: Path = DATA_ROOT) -> SequenceMetadata:
    path = _metadata_path(sequence_id, root)
    if not path.exists():
        raise FileNotFoundError(f"Sequence metadata not found: {sequence_id}")
    return SequenceMetadata(**json.loads(path.read_text(encoding="utf-8")))


def list_sequence_ids(root: Path = DATA_ROOT) -> list[str]:
    seq_root = root / "sequences"
    if not seq_root.exists():
        return []
    return sorted(path.name for path in seq_root.iterdir() if path.is_dir() and (path / "metadata.json").exists())


def scan_dataset(root: Path = DATA_ROOT, dataset_id: str = "demo", name: str = "OR-WM Demo Dataset") -> DatasetSummary:
    sequence_ids = list_sequence_ids(root)
    terrains: set[str] = set()
    for seq_id in sequence_ids:
        try:
            terrains.add(load_metadata(seq_id, root).terrain)
        except (FileNotFoundError, ValueError):
            continue
    return DatasetSummary(
        id=dataset_id,
        name=name,
        root=str(root),
        sequence_count=len(sequence_ids),
        terrain_types=sorted(terrains),
        sequences=sequence_ids,
    )


def asset_url(path: Path, root: Path = DATA_ROOT) -> str:
    rel = path.resolve().relative_to(root.resolve()).as_posix()
    return f"/assets/{rel}"


def get_sequence(sequence_id: str, root: Path = DATA_ROOT) -> SequenceDetail:
    seq = sequence_root(sequence_id, root)
    metadata = load_metadata(sequence_id, root)
    image_paths = sorted((seq / "images").glob("*.png"))
    occupancy_paths = sorted((seq / "occupancy").glob("*.png"))
    label_paths = sorted((seq / "labels").glob("*.png"))
    return SequenceDetail(
        id=sequence_id,
        metadata=metadata,
        frames=[asset_url(path, root) for path in image_paths],
        occupancy=[asset_url(path, root) for path in occupancy_paths],
        labels=[asset_url(path, root) for path in label_paths],
        poses_url=asset_url(seq / "poses.csv", root),
        calibration_url=asset_url(seq / "calibration.json", root),
    )


def _count_files(path: Path, patterns: tuple[str, ...] = ("*",)) -> int:
    if not path.exists():
        return 0
    count = 0
    for pattern in patterns:
        count += sum(1 for item in path.glob(pattern) if item.is_file())
    return count


def _item(
    key: str,
    label: str,
    status: str,
    value: str,
    required_for: list[str] | None = None,
    notes: list[str] | None = None,
) -> QualityItem:
    return QualityItem(
        key=key,
        label=label,
        status=status,  # type: ignore[arg-type]
        value=value,
        required_for=required_for or [],
        notes=notes or [],
    )


def sequence_quality(sequence_id: str, root: Path = DATA_ROOT) -> SequenceQuality:
    seq = sequence_root(sequence_id, root)
    metadata = load_metadata(sequence_id, root)
    image_count = _count_files(seq / "images", ("*.png", "*.jpg", "*.jpeg"))
    label_count = _count_files(seq / "labels", ("*.png", "*.jpg", "*.jpeg"))
    occupancy_count = _count_files(seq / "occupancy", ("*.png", "*.jpg", "*.jpeg"))
    lidar_count = _count_files(seq / "lidar")
    depth_count = _count_files(seq / "depth", ("*.png", "*.npy", "*.npz"))
    calibration_exists = (seq / "calibration.json").exists()
    poses_exists = (seq / "poses.csv").exists()
    actions_exists = (seq / "actions.csv").exists()

    items = [
        _item("images", "RGB frames", "ok" if image_count > 0 else "missing", str(image_count), ["all"], [] if image_count else ["No camera frames found."]),
        _item(
            "labels",
            "semantic masks",
            "ok" if image_count > 0 and label_count == image_count else ("missing" if label_count == 0 else "warning"),
            f"{label_count}/{image_count}",
            ["terrain"],
            [] if label_count == image_count and image_count > 0 else ["Terrain training expects one mask per frame."],
        ),
        _item(
            "occupancy",
            "occupancy/BEV maps",
            "ok" if occupancy_count > 0 else "missing",
            str(occupancy_count),
            ["world_model", "planner"],
            [] if occupancy_count else ["BEV world-model paths will fall back to mock/procedural maps."],
        ),
        _item(
            "calibration",
            "calibration",
            "ok" if calibration_exists else "missing",
            "present" if calibration_exists else "missing",
            ["reconstruction"],
            [] if calibration_exists else ["Camera/LiDAR projection requires calibration."],
        ),
        _item(
            "poses",
            "poses",
            "placeholder" if poses_exists and sequence_id.lower().startswith("rugd") else ("ok" if poses_exists else "missing"),
            "placeholder" if poses_exists and sequence_id.lower().startswith("rugd") else ("present" if poses_exists else "missing"),
            ["trajectory", "world_model"],
            ["RUGD import creates placeholder straight-line poses; use TartanDrive/RELLIS for real motion."] if poses_exists and sequence_id.lower().startswith("rugd") else ([] if poses_exists else ["Trajectory prediction needs real pose history and future pose."]),
        ),
        _item(
            "actions",
            "control actions",
            "ok" if actions_exists else "missing",
            "present" if actions_exists else "missing",
            ["action_conditioned_world_model", "rl"],
            [] if actions_exists else ["No steering/throttle/brake stream yet; action-conditioned models stay blocked."],
        ),
        _item(
            "lidar",
            "LiDAR",
            "ok" if lidar_count > 0 else "missing",
            str(lidar_count),
            ["reconstruction"],
            [] if lidar_count else ["Real BEV reconstruction needs LiDAR or depth."],
        ),
        _item(
            "depth",
            "depth",
            "ok" if depth_count > 0 else "missing",
            str(depth_count),
            ["reconstruction"],
            [] if depth_count else ["Depth can be an alternative path for BEV/heightmap reconstruction."],
        ),
        _item(
            "vehicle",
            "vehicle id",
            "ok" if metadata.vehicle_id else "missing",
            metadata.vehicle_id or "missing",
            ["simulation", "planner"],
            [] if metadata.vehicle_id else ["Vehicle configuration link is absent."],
        ),
    ]

    critical_missing = any(item.status == "missing" and item.key in {"images", "calibration", "poses"} for item in items)
    has_placeholder = any(item.status == "placeholder" for item in items)
    has_warning = any(item.status == "warning" for item in items)
    has_missing = any(item.status == "missing" for item in items)
    if has_placeholder:
        overall = "placeholder"
    elif critical_missing:
        overall = "missing"
    elif has_warning or has_missing:
        overall = "warning"
    else:
        overall = "ok"
    summary = f"{image_count} frames, {label_count} masks, {occupancy_count} BEV maps, {lidar_count} LiDAR files"
    return SequenceQuality(sequence_id=sequence_id, overall_status=overall, summary=summary, items=items)  # type: ignore[arg-type]


def dataset_quality(root: Path = DATA_ROOT) -> list[SequenceQuality]:
    return [sequence_quality(sequence_id, root=root) for sequence_id in list_sequence_ids(root)]


def _manifest_source_root(manifest_path: Path) -> str | None:
    if not manifest_path.exists():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    source_paths = [
        entry.get("source_image")
        for entry in manifest
        if isinstance(entry, dict) and isinstance(entry.get("source_image"), str)
    ]
    if not source_paths:
        return None
    try:
        return os.path.commonpath(source_paths)
    except ValueError:
        return str(Path(source_paths[0]).parent)


def sequence_source_card(sequence_id: str, root: Path = DATA_ROOT) -> DatasetSourceCard:
    seq = sequence_root(sequence_id, root)
    source_card_path = seq / "source_card.json"
    if source_card_path.exists():
        try:
            return DatasetSourceCard(**json.loads(source_card_path.read_text(encoding="utf-8")))
        except (json.JSONDecodeError, ValueError):
            pass
    metadata = load_metadata(sequence_id, root)
    tags = metadata.tags
    manifest_path = seq / "manifest.json"
    source_root = _manifest_source_root(manifest_path)
    quality = sequence_quality(sequence_id, root)

    sensors: list[str] = ["front_rgb"]
    if any(item.key == "labels" and item.status in {"ok", "warning"} for item in quality.items):
        sensors.append("semantic_mask")
    if metadata.has_occupancy:
        sensors.append("occupancy_or_bev")
    if metadata.has_lidar:
        sensors.append("lidar")
    if any(item.key == "poses" and item.status == "placeholder" for item in quality.items):
        sensors.append("placeholder_pose")
    elif any(item.key == "poses" and item.status == "ok" for item in quality.items):
        sensors.append("pose")
    if any(item.key == "actions" and item.status == "ok" for item in quality.items):
        sensors.append("actions")

    if "rugd" in tags or sequence_id.lower().startswith("rugd"):
        return DatasetSourceCard(
            sequence_id=sequence_id,
            dataset_name="RUGD-style public semantic dataset import",
            source_type="public",
            license="Check the original RUGD dataset license/terms before redistribution.",
            citation="RUGD: Robot Unstructured Ground Driving dataset. Cite the original RUGD dataset/paper/site.",
            homepage="https://rugd.vision/",
            importer="ml.rugd_adapter.import_rugd_sequence",
            importer_version=IMPORTER_VERSION,
            source_root=source_root,
            target_root=str(seq),
            manifest_path=str(manifest_path) if manifest_path.exists() else None,
            sensors=sensors,
            tags=tags,
            known_limitations=[
                "No LiDAR/depth/action stream in the current RUGD-style import.",
                "poses.csv is a placeholder straight-line pose stream.",
                "calibration.json is unknown/placeholder.",
                "occupancy maps are generated by OR-WM mock BEV reconstruction.",
            ],
            recommended_next=[
                "Use this source for terrain perception and label remapping.",
                "Use TartanDrive-style data for real trajectory prediction.",
                "Use RELLIS-3D-style data for LiDAR BEV reconstruction.",
            ],
        )

    if "synthetic" in tags or sequence_id == "seq_0001":
        return DatasetSourceCard(
            sequence_id=sequence_id,
            dataset_name="OR-WM built-in synthetic demo sequence",
            source_type="synthetic",
            license="Local demo asset generated by OR-WM Studio; not a public benchmark.",
            citation="No external citation; generated by OR-WM procedural demo scripts.",
            homepage=None,
            importer="ml.scene_generation.generate_demo_sequence",
            importer_version=IMPORTER_VERSION,
            source_root=None,
            target_root=str(seq),
            manifest_path=None,
            sensors=sensors,
            tags=tags,
            known_limitations=[
                "Synthetic/demo data only.",
                "Vehicle state, actions, LiDAR, and battery telemetry are not real.",
                "Metrics from this sequence should be reported as synthetic or mock unless replaced.",
            ],
            recommended_next=[
                "Use this sequence for UI smoke tests.",
                "Replace with public dataset imports for model evaluation.",
            ],
        )

    return DatasetSourceCard(
        sequence_id=sequence_id,
        dataset_name=f"Custom sequence {sequence_id}",
        source_type="custom" if manifest_path.exists() else "unknown",
        license="Unknown; fill this in before publishing or sharing results.",
        citation="Unknown; add dataset citation before reporting experiments.",
        homepage=None,
        importer="unknown",
        importer_version=IMPORTER_VERSION,
        source_root=source_root,
        target_root=str(seq),
        manifest_path=str(manifest_path) if manifest_path.exists() else None,
        sensors=sensors,
        tags=tags,
        known_limitations=[
            "Dataset source card was inferred because no explicit source_card.json exists.",
            "Verify license, citation, calibration, pose/action semantics, and sensor coverage.",
        ],
        recommended_next=[
            "Add source_card.json or extend the importer to emit explicit source metadata.",
        ],
    )


def dataset_source_cards(root: Path = DATA_ROOT) -> list[DatasetSourceCard]:
    return [sequence_source_card(sequence_id, root=root) for sequence_id in list_sequence_ids(root)]
