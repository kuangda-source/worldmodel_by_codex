from __future__ import annotations

from pathlib import Path

from .scene_generation import render_bev_assets


def reconstruct_mock_bev(output_dir: Path, *, seed: int, terrain: str = "mountain") -> dict[str, float]:
    output_dir.mkdir(parents=True, exist_ok=True)
    return render_bev_assets(output_dir, seed=seed, terrain=terrain, obstacle_density=0.34, slope=0.46)
