from __future__ import annotations

from pathlib import Path

from scene_generation import generate_demo_sequence


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data" / "demo"
    generate_demo_sequence(data_root)
    print(f"Generated demo dataset at {data_root}")


if __name__ == "__main__":
    main()
