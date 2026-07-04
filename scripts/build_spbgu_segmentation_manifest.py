from __future__ import annotations

import argparse
import json

from data.spbgu_segmentation import SpbguSegmentationManifestBuilder
from hydra.json_config import load_config


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for SPbGU manifest building."""
    parser = argparse.ArgumentParser(description="Build SPbGU AFM segmentation manifest.")
    parser.add_argument("--config", default="configs/segmentation/spbgu_unet.json")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    """Build a train/val manifest from SPbGU source/mask pairs."""
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    rows = SpbguSegmentationManifestBuilder(cfg["manifest"]).build()
    print(json.dumps({"rows": len(rows), "output_csv": cfg["manifest"]["output_csv"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
