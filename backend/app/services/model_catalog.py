from __future__ import annotations

from ..schemas import ModelCatalogItem, ModelLaunchAction
from .datasets import sequence_quality


def _status_map(sequence_id: str) -> dict[str, str]:
    quality = sequence_quality(sequence_id)
    return {item.key: item.status for item in quality.items}


def _missing(statuses: dict[str, str], keys: list[str]) -> list[str]:
    return [key for key in keys if statuses.get(key) not in {"ok", "warning", "placeholder"}]


def _launch(
    action_id: str,
    label: str,
    endpoint: str | None,
    body: dict[str, object] | None = None,
    *,
    enabled: bool = True,
    disabled_reason: str | None = None,
    method: str = "POST",
) -> ModelLaunchAction:
    return ModelLaunchAction(
        id=action_id,
        label=label,
        endpoint=endpoint,
        method=method,  # type: ignore[arg-type]
        body=body or {},
        enabled=enabled,
        disabled_reason=disabled_reason,
    )


def _blocked_item(
    *,
    item_id: str,
    name: str,
    task: str,
    adapter: str,
    required: list[str],
    statuses: dict[str, str],
    outputs: list[str],
    recommended_next: list[str],
) -> ModelCatalogItem:
    blockers = _missing(statuses, required)
    return ModelCatalogItem(
        id=item_id,
        name=name,
        task=task,
        adapter=adapter,
        status="blocked" if blockers else "ready",
        source="real_data" if not blockers else "placeholder",
        required_streams=required,
        optional_streams=[],
        outputs=outputs,
        blockers=[f"Missing or unusable stream: {key}" for key in blockers],
        recommended_next=recommended_next,
        launch_actions=[
            _launch(
                f"{item_id}_planned",
                "Planned adapter",
                None,
                enabled=False,
                disabled_reason="Adapter is declared for planning but not implemented yet.",
            )
        ],
    )


def model_catalog(sequence_id: str = "seq_0001") -> list[ModelCatalogItem]:
    statuses = _status_map(sequence_id)
    terrain_source = "real_data" if sequence_id.lower().startswith("rugd") else "synthetic"
    pose_status = statuses.get("poses")
    action_status = statuses.get("actions")
    lidar_status = statuses.get("lidar")

    terrain_blockers = _missing(statuses, ["images", "labels"])
    terrain_status = "blocked" if terrain_blockers else "ready"
    terrain_enabled = terrain_status != "blocked"
    terrain_source_format = "rugd-style" if sequence_id.lower().startswith("rugd") else "orwm-demo"

    trajectory_status = "placeholder" if pose_status == "placeholder" else ("ready" if pose_status == "ok" else "blocked")
    trajectory_blockers = [] if pose_status in {"ok", "placeholder"} else ["Missing or unusable stream: poses"]

    action_world_model_blockers = []
    if statuses.get("occupancy") not in {"ok", "warning"}:
        action_world_model_blockers.append("Missing or unusable stream: occupancy")
    if action_status != "ok":
        action_world_model_blockers.append("Missing or unusable stream: actions")

    return [
        ModelCatalogItem(
            id="terrain_tiny_mlp",
            name="TinyMLP Terrain",
            task="terrain_perception",
            adapter="ml.lightweight_segmentation",
            status=terrain_status,  # type: ignore[arg-type]
            source=terrain_source,  # type: ignore[arg-type]
            required_streams=["images", "labels"],
            optional_streams=["calibration"],
            outputs=["semantic", "traversability", "risk", "overlay"],
            blockers=[f"Missing or unusable stream: {key}" for key in terrain_blockers],
            recommended_next=["Use RUGD/RELLIS-style masks for real terrain perception."],
            launch_actions=[
                _launch(
                    "train",
                    "Train Terrain",
                    "/api/traversability/train",
                    {
                        "sequence_id": sequence_id,
                        "source_format": terrain_source_format,
                        "trainer": "tiny-mlp",
                        "max_samples": 32,
                        "max_pixels": 24000,
                        "epochs": 12,
                    },
                    enabled=terrain_enabled,
                    disabled_reason="RGB frames and semantic masks are required." if not terrain_enabled else None,
                ),
                _launch(
                    "predict",
                    "Segment Frame",
                    "/api/traversability/predict",
                    {"sequence_id": sequence_id, "frame_index": 0},
                    enabled=terrain_enabled,
                    disabled_reason="Train or load a terrain model after masks are available." if not terrain_enabled else None,
                ),
            ],
        ),
        ModelCatalogItem(
            id="trajectory_tiny_gru",
            name="TinyTrajGRU",
            task="trajectory_prediction",
            adapter="ml.trajectory_prediction",
            status=trajectory_status,  # type: ignore[arg-type]
            source="placeholder" if trajectory_status == "placeholder" else ("real_data" if trajectory_status == "ready" else "placeholder"),
            required_streams=["poses"],
            optional_streams=["actions", "imu", "wheel_odometry"],
            outputs=["future_poses", "ade", "fde", "trajectory_plot"],
            blockers=trajectory_blockers,
            recommended_next=["Import TartanDrive-style poses/actions before treating ADE/FDE as real."],
            launch_actions=[
                _launch(
                    "train",
                    "Train Traj",
                    "/api/trajectory/train",
                    {
                        "sequence_id": sequence_id,
                        "model": "tiny-traj-gru",
                        "history": 6,
                        "horizon": 8,
                        "epochs": 60,
                        "hidden_dim": 48,
                        "augment": 96,
                        "seed": 23,
                    },
                    enabled=trajectory_status in {"ready", "placeholder"},
                    disabled_reason="Pose stream is required." if trajectory_status == "blocked" else None,
                ),
                _launch(
                    "predict",
                    "Predict Traj",
                    "/api/trajectory/predict",
                    {"sequence_id": sequence_id, "history": 6, "horizon": 8, "frame_index": 8},
                    enabled=trajectory_status in {"ready", "placeholder"},
                    disabled_reason="Pose stream is required." if trajectory_status == "blocked" else None,
                ),
            ],
        ),
        ModelCatalogItem(
            id="mock_bev_reconstruction",
            name="Mock BEV Reconstruction",
            task="reconstruction",
            adapter="ml.reconstruction.reconstruct_mock_bev",
            status="mock",
            source="mock",
            required_streams=["images"],
            optional_streams=["labels"],
            outputs=["occupancy", "heightmap", "traversability", "risk"],
            blockers=[],
            recommended_next=["Replace with RELLIS-3D LiDAR/depth projection when LiDAR/depth is available."],
            launch_actions=[
                _launch(
                    "run",
                    "Reconstruct",
                    "/api/reconstruction/run",
                    {"sequence_id": sequence_id, "method": "mock-bev", "seed": 17},
                )
            ],
        ),
        ModelCatalogItem(
            id="tiny_bev_world_model",
            name="Tiny BEV World Model",
            task="world_model",
            adapter="ml.world_model",
            status="mock",
            source="synthetic",
            required_streams=["occupancy"],
            optional_streams=["actions"],
            outputs=["future_bev", "ego_motion", "risk", "uncertainty"],
            blockers=[] if statuses.get("occupancy") in {"ok", "warning"} else ["Missing or unusable stream: occupancy"],
            recommended_next=["Train action-conditioned BEV dynamics after real actions are imported."],
            launch_actions=[
                _launch(
                    "train",
                    "Train WM",
                    "/api/world-model/train",
                    {"sequence_id": sequence_id, "model": "tiny-bev-cnn", "epochs": 8, "seed": 13},
                    enabled=statuses.get("occupancy") in {"ok", "warning"},
                    disabled_reason="Occupancy/BEV maps are required." if statuses.get("occupancy") not in {"ok", "warning"} else None,
                ),
                _launch(
                    "predict",
                    "Predict WM",
                    "/api/world-model/predict",
                    {
                        "sequence_id": sequence_id,
                        "action": {"steer": -0.16, "throttle": 0.48, "brake": 0},
                        "horizon": 7,
                        "seed": 21,
                    },
                    enabled=statuses.get("occupancy") in {"ok", "warning"},
                    disabled_reason="Occupancy/BEV maps are required." if statuses.get("occupancy") not in {"ok", "warning"} else None,
                ),
            ],
        ),
        ModelCatalogItem(
            id="toy_rl_policy",
            name="Toy RL Policy",
            task="rl_policy",
            adapter="ml.rl_env",
            status="mock",
            source="toy_env",
            required_streams=[],
            optional_streams=["generated_scene"],
            outputs=["replay", "success_rate", "collision_rate", "path_length"],
            blockers=[],
            recommended_next=["Keep as toy until a simulator or logged-control offline evaluator is connected."],
            launch_actions=[
                _launch(
                    "run",
                    "Run Policy",
                    "/api/rl/train",
                    {"scene_id": None, "algorithm": "ppo-fallback", "episodes": 10, "seed": 31},
                )
            ],
        ),
        _blocked_item(
            item_id="rellis_lidar_bev",
            name="RELLIS-style LiDAR BEV",
            task="reconstruction",
            adapter="future.rellis_adapter",
            required=["lidar", "calibration", "poses"],
            statuses=statuses,
            outputs=["point_density", "heightmap", "occupancy", "bev_confidence"],
            recommended_next=["Import RELLIS-3D-style LiDAR/calibration to enable this adapter."],
        ) if lidar_status != "ok" else ModelCatalogItem(
            id="rellis_lidar_bev",
            name="RELLIS-style LiDAR BEV",
            task="reconstruction",
            adapter="future.rellis_adapter",
            status="ready",
            source="real_data",
            required_streams=["lidar", "calibration", "poses"],
            optional_streams=["labels"],
            outputs=["point_density", "heightmap", "occupancy", "bev_confidence"],
            blockers=[],
            recommended_next=["Run LiDAR BEV reconstruction and compare with mock BEV."],
            launch_actions=[
                _launch(
                    "planned",
                    "Run LiDAR BEV",
                    None,
                    enabled=False,
                    disabled_reason="RELLIS LiDAR adapter is declared but not implemented yet.",
                )
            ],
        ),
        ModelCatalogItem(
            id="action_conditioned_bev_model",
            name="Action-Conditioned BEV Predictor",
            task="world_model",
            adapter="future.action_bev_predictor",
            status="blocked" if action_world_model_blockers else "ready",
            source="placeholder" if action_world_model_blockers else "real_data",
            required_streams=["occupancy", "actions"],
            optional_streams=["poses", "traversability"],
            outputs=["future_occupancy", "future_risk", "uncertainty"],
            blockers=action_world_model_blockers,
            recommended_next=["Import action streams from TartanDrive or simulator logs."],
            launch_actions=[
                _launch(
                    "planned",
                    "Train Action WM",
                    None,
                    enabled=False,
                    disabled_reason="Action-conditioned BEV predictor is a planned adapter.",
                )
            ],
        ),
    ]
