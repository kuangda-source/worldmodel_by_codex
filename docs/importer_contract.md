# OR-WM Importer Contract v0.1

Every dataset adapter should write one sequence under:

```text
data/demo/sequences/{sequence_id}/
```

## Required Files

```text
metadata.json
manifest.json
source_card.json
images/
poses.csv
calibration.json
```

`images/` should contain RGB frames named consistently, for example `frame_0000.png`.

## Optional Sensor Folders

```text
labels/       semantic or traversability masks
occupancy/    BEV, occupancy, risk, heightmap artifacts
lidar/        point clouds
depth/        depth maps or arrays
actions.csv   steering, throttle, brake, or command stream
```

Missing optional streams are allowed, but they must be made explicit through `source_card.json` and will appear in the dataset quality panel.

## metadata.json

```json
{
  "scene_id": "sequence_id",
  "terrain": "gravel",
  "weather": "sunny",
  "time_of_day": "day",
  "difficulty": "real-data",
  "tags": ["public-dataset"],
  "has_lidar": false,
  "has_occupancy": true,
  "vehicle_id": "ugv_default"
}
```

## manifest.json

`manifest.json` maps OR-WM files back to original dataset files.

```json
[
  {
    "frame": "frame_0000",
    "source_image": "D:/datasets/RUGD/...",
    "source_label": "D:/datasets/RUGD/...",
    "image": "images/frame_0000.png",
    "label": "labels/semantic_0000.png"
  }
]
```

For LiDAR, depth, poses, or actions, add keys such as `source_lidar`, `source_depth`, `source_pose`, and `source_action`.

## source_card.json

```json
{
  "sequence_id": "rugd_mini",
  "dataset_name": "RUGD-style public semantic dataset import",
  "source_type": "public",
  "license": "Check the original dataset terms.",
  "citation": "Cite the original dataset/paper/site.",
  "homepage": "https://rugd.vision/",
  "importer": "ml.rugd_adapter.import_rugd_sequence",
  "importer_version": "orwm-rugd-importer-v0.1",
  "source_root": "D:/datasets/RUGD",
  "target_root": "D:/program/worldmodel_codex/data/demo/sequences/rugd_mini",
  "manifest_path": "D:/program/worldmodel_codex/data/demo/sequences/rugd_mini/manifest.json",
  "sensors": ["front_rgb", "semantic_mask", "placeholder_pose"],
  "tags": ["rugd", "public-dataset"],
  "known_limitations": [
    "poses.csv is placeholder, not real odometry."
  ],
  "recommended_next": [
    "Use TartanDrive for real trajectory prediction."
  ]
}
```

## Placeholder Rules

Use placeholders only when they keep the demo loop runnable, and always disclose them:

- Placeholder poses should be marked in `source_card.json`.
- Mock BEV/occupancy should be described as generated, not measured.
- Unknown calibration should be written as explicit `null` fields with a note.
- Missing actions block action-conditioned world models and real RL claims.

## Quality Checker Hooks

The backend currently checks:

- RGB frame count
- semantic mask count
- occupancy/BEV count
- calibration file
- pose stream
- control action stream
- LiDAR files
- depth files
- vehicle ID

When adding a new importer, update `backend/app/services/datasets.py` only if the new dataset has a new sensor convention or a new placeholder rule.

## Adapter Checklist

1. Copy or convert source frames into OR-WM layout.
2. Preserve original paths in `manifest.json`.
3. Write `metadata.json`.
4. Write `source_card.json`.
5. Create placeholders only when needed and disclose them.
6. Run `GET /api/sequences/{sequence_id}/quality`.
7. Run at least one model or prediction path and verify `/api/runs`.

## Implemented Adapters

### RUGD-style semantic import

Entry points:

```text
POST /api/public-datasets/rugd/import
python ml/import_rugd_dataset.py D:/datasets/RUGD --sequence-id rugd_mini --max-samples 24
```

Best for:

- front-view terrain segmentation
- semantic mask review
- traversability/risk map training

Known limitation:

- RUGD does not provide real ego pose/action streams in this demo adapter, so trajectory prediction is marked as placeholder unless another dataset provides real poses.

### TartanDrive-style trajectory/action import

Entry points:

```text
POST /api/public-datasets/tartandrive/import
python ml/import_tartandrive_dataset.py D:/datasets/TartanDriveMini --sequence-id tartandrive_mini --max-samples 64
```

Supported mini-subset input:

```text
states.csv | poses.csv | odometry.csv | trajectory.csv
actions.csv | controls.csv | commands.csv
images/ | rgb/ | camera/ | frames/ | front/
```

Best for:

- real ego pose/action quality checks
- `TinyTrajGRU` trajectory training and prediction
- future action-conditioned world-model work

Known limitation:

- The current importer is a CSV/flat mini-subset bridge, not a complete parser for every original TartanDrive release artifact.
