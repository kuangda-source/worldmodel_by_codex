from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.rugd_adapter import import_rugd_sequence  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Import a RUGD-style RGB/semantic dataset into OR-WM sequence layout.")
    parser.add_argument("root_path", help="Path to the downloaded RUGD or RUGD-style dataset root.")
    parser.add_argument("--sequence-id", default="rugd_mini", help="Target sequence id under data/demo/sequences.")
    parser.add_argument("--max-samples", type=int, default=24, help="Maximum image/label pairs to import.")
    parser.add_argument("--no-overwrite", action="store_true", help="Do not overwrite an existing target sequence.")
    args = parser.parse_args()

    result = import_rugd_sequence(
        Path(args.root_path).resolve(),
        REPO_ROOT / "data" / "demo",
        sequence_id=args.sequence_id,
        max_samples=args.max_samples,
        overwrite=not args.no_overwrite,
    )
    print(f"Imported {result['imported_frames']} frames into {result['target_root']}")


if __name__ == "__main__":
    main()
