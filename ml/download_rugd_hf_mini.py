from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from huggingface_hub import hf_hub_download, list_repo_files  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a small RUGD subset from a Hugging Face mirror.")
    parser.add_argument("--repo-id", default="WilliamBonilla62/RUGD")
    parser.add_argument("--output", default=str(REPO_ROOT / "data" / "public" / "RUGD_hf_mini"))
    parser.add_argument("--max-pairs", type=int, default=24)
    parser.add_argument("--scene", default="creek")
    args = parser.parse_args()

    repo_id = args.repo_id
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    files = list_repo_files(repo_id, repo_type="dataset")
    label_prefix = f"RUGD_annotations/{args.scene}/"
    frame_prefix = "RUGD_frames-with-annotations/"
    labels = [name for name in files if name.startswith(label_prefix) and name.endswith(".png")]
    if not labels:
        raise FileNotFoundError(f"No labels found for scene '{args.scene}' in {repo_id}")

    selected = labels[: args.max_pairs]
    wanted = ["README.md", "RUGD_annotation-colormap.txt"]
    for label in selected:
        rel = label.removeprefix("RUGD_annotations/")
        frame = f"{frame_prefix}{rel}"
        if frame not in files:
            raise FileNotFoundError(f"Missing matching frame for {label}: {frame}")
        wanted.extend([label, frame])

    for index, filename in enumerate(wanted, start=1):
        print(f"[{index}/{len(wanted)}] {filename}")
        hf_hub_download(repo_id=repo_id, repo_type="dataset", filename=filename, local_dir=output)

    print(f"Downloaded {len(selected)} image/label pairs into {output}")


if __name__ == "__main__":
    main()
