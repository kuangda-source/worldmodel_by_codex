from __future__ import annotations

import json
import math
import random
import struct
import zlib
from pathlib import Path
from typing import Iterable


RGB = tuple[int, int, int]

RUGD_COLORS: dict[str, RGB] = {
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


def clamp(value: float, low: int = 0, high: int = 255) -> int:
    return max(low, min(high, int(value)))


def mix(a: RGB, b: RGB, t: float) -> RGB:
    return (
        clamp(a[0] * (1 - t) + b[0] * t),
        clamp(a[1] * (1 - t) + b[1] * t),
        clamp(a[2] * (1 - t) + b[2] * t),
    )


def write_png(path: Path, width: int, height: int, pixels: list[list[RGB]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    raw = bytearray()
    for row in pixels:
        raw.append(0)
        for r, g, b in row:
            raw.extend((clamp(r), clamp(g), clamp(b)))
    payload = b"\x89PNG\r\n\x1a\n"
    payload += chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    payload += chunk(b"IDAT", zlib.compress(bytes(raw), 6))
    payload += chunk(b"IEND", b"")
    path.write_bytes(payload)


def _terrain_palette(terrain: str, weather: str) -> dict[str, RGB]:
    palettes = {
        "forest": {"trail": (122, 94, 70), "grass": (72, 132, 73), "canopy": (39, 91, 55), "rock": (118, 119, 111)},
        "mountain": {"trail": (136, 112, 88), "grass": (94, 125, 86), "canopy": (68, 104, 73), "rock": (139, 136, 128)},
        "mud": {"trail": (92, 71, 54), "grass": (68, 116, 69), "canopy": (36, 79, 50), "rock": (95, 89, 82)},
        "sand": {"trail": (178, 151, 104), "grass": (129, 137, 91), "canopy": (89, 120, 81), "rock": (154, 139, 113)},
        "gravel": {"trail": (122, 122, 118), "grass": (85, 118, 82), "canopy": (57, 96, 66), "rock": (151, 151, 146)},
        "snow": {"trail": (190, 197, 194), "grass": (131, 153, 136), "canopy": (86, 117, 102), "rock": (159, 164, 163)},
    }
    key = {
        "山地": "mountain",
        "森林": "forest",
        "泥泞": "mud",
        "沙地": "sand",
        "碎石": "gravel",
        "雪地": "snow",
    }.get(terrain, terrain.lower())
    palette = palettes.get(key, palettes["mountain"]).copy()
    if weather in {"rain", "雨天"}:
        palette = {name: mix(color, (38, 55, 62), 0.18) for name, color in palette.items()}
    if weather in {"fog", "雾霾", "雾天"}:
        palette = {name: mix(color, (190, 198, 190), 0.20) for name, color in palette.items()}
    return palette


def _noise(x: int, y: int, seed: int) -> float:
    n = (x * 374761393 + y * 668265263 + seed * 1442695041) & 0xFFFFFFFF
    n = (n ^ (n >> 13)) * 1274126177 & 0xFFFFFFFF
    return ((n ^ (n >> 16)) & 0xFFFF) / 65535.0


def _draw_disc(pixels: list[list[RGB]], cx: int, cy: int, radius: int, color: RGB) -> None:
    height = len(pixels)
    width = len(pixels[0])
    r2 = radius * radius
    for y in range(max(0, cy - radius), min(height, cy + radius + 1)):
        for x in range(max(0, cx - radius), min(width, cx + radius + 1)):
            if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                pixels[y][x] = color


def _draw_rect(pixels: list[list[RGB]], x0: int, y0: int, x1: int, y1: int, color: RGB) -> None:
    height = len(pixels)
    width = len(pixels[0])
    for y in range(max(0, y0), min(height, y1)):
        for x in range(max(0, x0), min(width, x1)):
            pixels[y][x] = color


def render_front_view(path: Path, seed: int, terrain: str, weather: str, frame: int = 0, width: int = 960, height: int = 540) -> None:
    rng = random.Random(seed + frame * 17)
    palette = _terrain_palette(terrain, weather)
    horizon = int(height * 0.28)
    pixels: list[list[RGB]] = []

    for y in range(height):
        row: list[RGB] = []
        for x in range(width):
            if y < horizon:
                t = y / max(1, horizon)
                base = mix((186, 211, 224), (222, 232, 220), t)
            else:
                t = (y - horizon) / max(1, height - horizon)
                base = mix(palette["canopy"], palette["grass"], min(1, t + 0.22))
                texture = (_noise(x // 6, y // 5, seed + frame) - 0.5) * 34
                base = (clamp(base[0] + texture), clamp(base[1] + texture), clamp(base[2] + texture))
            row.append(base)
        pixels.append(row)

    for y in range(horizon, height):
        depth = (y - horizon) / max(1, height - horizon)
        center = width * (0.47 + 0.08 * math.sin(depth * 3.3 + seed * 0.1 + frame * 0.08))
        half_width = width * (0.035 + depth**1.7 * 0.30)
        for x in range(width):
            edge = abs(x - center) / max(1, half_width)
            if edge < 1.0:
                dirt = palette["trail"]
                texture = (_noise(x // 5, y // 4, seed + 101 + frame) - 0.5) * 42
                color = (clamp(dirt[0] + texture), clamp(dirt[1] + texture), clamp(dirt[2] + texture))
                if edge > 0.82:
                    color = mix(color, palette["grass"], (edge - 0.82) / 0.18)
                pixels[y][x] = color

    for _ in range(48):
        base_x = rng.randrange(20, width - 20)
        base_y = rng.randrange(horizon + 20, height + 80)
        trunk_h = rng.randrange(110, 320)
        trunk_w = max(4, int((base_y / height) * rng.randrange(8, 18)))
        trunk_color = mix((68, 46, 34), (118, 95, 75), rng.random() * 0.35)
        _draw_rect(pixels, base_x - trunk_w // 2, base_y - trunk_h, base_x + trunk_w // 2, base_y, trunk_color)
        for _branch in range(3):
            bx = base_x + rng.randrange(-35, 36)
            by = base_y - rng.randrange(95, max(100, trunk_h))
            _draw_rect(pixels, bx - 2, by - 45, bx + 2, by + 8, trunk_color)

    for _ in range(34):
        cx = rng.randrange(0, width)
        cy = rng.randrange(int(height * 0.48), height)
        radius = rng.randrange(4, 18)
        color = mix(palette["rock"], palette["trail"], rng.random() * 0.35)
        _draw_disc(pixels, cx, cy, radius, color)

    if weather in {"rain", "雨天"}:
        rain_color = (176, 190, 196)
        for _ in range(380):
            x = rng.randrange(0, width)
            y = rng.randrange(0, height)
            for d in range(10):
                yy = y + d
                xx = x - d // 3
                if 0 <= xx < width and 0 <= yy < height:
                    pixels[yy][xx] = mix(pixels[yy][xx], rain_color, 0.55)
    if weather in {"fog", "雾霾", "雾天"}:
        for y in range(height):
            haze = 0.20 + 0.25 * (1 - y / height)
            for x in range(width):
                pixels[y][x] = mix(pixels[y][x], (210, 215, 207), haze)

    write_png(path, width, height, pixels)


def render_semantic_mask(path: Path, seed: int, frame: int = 0, width: int = 960, height: int = 540) -> None:
    rng = random.Random(seed + frame * 17)
    horizon = int(height * 0.28)
    pixels: list[list[RGB]] = []

    for y in range(height):
        row: list[RGB] = []
        for x in range(width):
            if y < horizon:
                row.append(RUGD_COLORS["sky"])
            else:
                t = (y - horizon) / max(1, height - horizon)
                row.append(RUGD_COLORS["bush"] if t < 0.34 else RUGD_COLORS["grass"])
        pixels.append(row)

    for y in range(horizon, height):
        depth = (y - horizon) / max(1, height - horizon)
        center = width * (0.47 + 0.08 * math.sin(depth * 3.3 + seed * 0.1 + frame * 0.08))
        half_width = width * (0.035 + depth**1.7 * 0.30)
        for x in range(width):
            edge = abs(x - center) / max(1, half_width)
            if edge < 1.0:
                pixels[y][x] = RUGD_COLORS["dirt"] if edge < 0.76 else RUGD_COLORS["gravel"]

    tree_specs: list[tuple[int, int, int, int]] = []
    for _ in range(48):
        base_x = rng.randrange(20, width - 20)
        base_y = rng.randrange(horizon + 20, height + 80)
        trunk_h = rng.randrange(110, 320)
        trunk_w = max(4, int((base_y / height) * rng.randrange(8, 18)))
        tree_specs.append((base_x, base_y, trunk_h, trunk_w))
    for base_x, base_y, trunk_h, trunk_w in tree_specs:
        _draw_rect(pixels, base_x - trunk_w // 2, base_y - trunk_h, base_x + trunk_w // 2, base_y, RUGD_COLORS["tree"])

    for _ in range(34):
        cx = rng.randrange(0, width)
        cy = rng.randrange(int(height * 0.48), height)
        radius = rng.randrange(4, 18)
        _draw_disc(pixels, cx, cy, radius, RUGD_COLORS["rock"])

    write_png(path, width, height, pixels)


def render_bev_assets(output_dir: Path, seed: int, terrain: str, obstacle_density: float, slope: float, size: int = 256) -> dict[str, float]:
    rng = random.Random(seed)
    palette = _terrain_palette(terrain, "sunny")
    obstacles: list[tuple[int, int, int]] = []
    obstacle_count = int(18 + obstacle_density * 46)
    for _ in range(obstacle_count):
        obstacles.append((rng.randrange(18, size - 18), rng.randrange(18, size - 18), rng.randrange(4, 13)))

    maps = {
        "occupancy": [[(32, 38, 36) for _x in range(size)] for _y in range(size)],
        "traversability": [[(35, 90, 68) for _x in range(size)] for _y in range(size)],
        "risk": [[(36, 48, 57) for _x in range(size)] for _y in range(size)],
        "heightmap": [[(0, 0, 0) for _x in range(size)] for _y in range(size)],
    }
    path_cells = 0
    risky_cells = 0

    for y in range(size):
        progress = y / max(1, size - 1)
        center = size * (0.50 + 0.17 * math.sin(progress * 4.2 + seed * 0.05))
        width = size * (0.10 + 0.04 * math.sin(progress * 7.0 + 1.2))
        for x in range(size):
            n = _noise(x // 4, y // 4, seed)
            height_value = clamp(55 + 150 * (progress * slope + 0.20 * math.sin(x / 21 + seed) + 0.18 * n))
            maps["heightmap"][y][x] = (height_value, height_value, height_value)

            dist = abs(x - center)
            on_trail = dist < width
            edge = min(1, max(0, (dist - width * 0.65) / max(1, width * 0.35)))
            if on_trail:
                path_cells += 1
                maps["occupancy"][y][x] = mix(palette["trail"], (214, 208, 184), 0.15 * n)
                score = clamp(120 + 90 * (1 - edge) - slope * 35 + n * 16)
                maps["traversability"][y][x] = (clamp(180 - score * 0.40), clamp(score), 80)
                risk = clamp(36 + edge * 100 + obstacle_density * 28 + slope * 26)
            else:
                maps["occupancy"][y][x] = mix(palette["grass"], palette["canopy"], n * 0.75)
                maps["traversability"][y][x] = (145, clamp(90 + n * 35), 73)
                risk = clamp(90 + min(120, dist - width) * 0.8 + slope * 34)
            if risk > 145:
                risky_cells += 1
            maps["risk"][y][x] = (clamp(risk + 35), clamp(180 - risk * 0.5), clamp(76 - risk * 0.20))

    for ox, oy, radius in obstacles:
        for name, color in {
            "occupancy": (34, 34, 34),
            "traversability": (160, 49, 43),
            "risk": (217, 94, 57),
            "heightmap": (225, 225, 225),
        }.items():
            _draw_disc(maps[name], ox, oy, radius, color)

    for name, pixels in maps.items():
        write_png(output_dir / f"{name}.png", size, size, pixels)

    total = size * size
    return {
        "path_coverage": round(path_cells / total, 3),
        "risk_ratio": round(risky_cells / total, 3),
        "obstacle_count": float(obstacle_count),
        "slope_score": round(float(slope), 3),
    }


def generate_scene_assets(
    output_dir: Path,
    *,
    seed: int,
    terrain: str,
    weather: str,
    task: str,
    prompt: str,
    obstacle_density: float = 0.35,
    slope: float = 0.45,
) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    render_front_view(output_dir / "front_view.png", seed, terrain, weather)
    metrics = render_bev_assets(output_dir, seed, terrain, obstacle_density, slope)
    metadata = {
        "scene_id": output_dir.name,
        "seed": seed,
        "terrain": terrain,
        "weather": weather,
        "task": task,
        "prompt": prompt,
        "assets": {
            "front_view": "front_view.png",
            "occupancy": "occupancy.png",
            "traversability": "traversability.png",
            "risk": "risk.png",
            "heightmap": "heightmap.png",
        },
        "metrics": metrics,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def generate_demo_sequence(root: Path, seed: int = 5) -> None:
    seq = root / "sequences" / "seq_0001"
    (seq / "images").mkdir(parents=True, exist_ok=True)
    (seq / "occupancy").mkdir(parents=True, exist_ok=True)
    (seq / "labels").mkdir(parents=True, exist_ok=True)
    (seq / "lidar").mkdir(parents=True, exist_ok=True)

    for frame in range(6):
        render_front_view(seq / "images" / f"frame_{frame:04d}.png", seed, "mountain", "sunny", frame=frame)
        render_semantic_mask(seq / "labels" / f"semantic_{frame:04d}.png", seed, frame=frame)
    render_bev_assets(seq / "occupancy", seed=seed, terrain="mountain", obstacle_density=0.38, slope=0.50)

    poses = ["timestamp,x,y,z,yaw,pitch,roll,speed"]
    for frame in range(6):
        poses.append(
            f"{frame * 0.1:.1f},{frame * 1.8:.3f},{math.sin(frame * 0.45) * 0.8:.3f},0.0,"
            f"{-0.04 + frame * 0.011:.3f},{0.08 + frame * 0.004:.3f},{-0.06 + frame * 0.006:.3f},{5.4 + frame * 0.18:.2f}"
        )
    (seq / "poses.csv").write_text("\n".join(poses) + "\n", encoding="utf-8")
    (seq / "calibration.json").write_text(
        json.dumps(
            {
                "camera": {"width": 960, "height": 540, "fx": 615.0, "fy": 615.0, "cx": 480.0, "cy": 270.0},
                "lidar_to_vehicle": [0.0, 0.0, 1.4, 0.0, 0.0, 0.0],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (seq / "metadata.json").write_text(
        json.dumps(
            {
                "scene_id": "seq_0001",
                "terrain": "mountain",
                "weather": "sunny",
                "time_of_day": "day",
                "difficulty": "medium",
                "tags": ["trail", "rocks", "slope", "forest-edge"],
                "has_lidar": False,
                "has_occupancy": True,
                "vehicle_id": "ugv_default",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def urls_for_assets(base_url: str, names: Iterable[str]) -> dict[str, str]:
    return {name.removesuffix(".png"): f"{base_url}/{name}" for name in names}
