from __future__ import annotations

import argparse

from hydra.json_config import load_config
from visualization.spbgu_segmentation import SpbguMaskPredictor


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for SPbGU mask prediction export."""
    parser = argparse.ArgumentParser(description="Predict SPbGU AFM masks and overlays.")
    parser.add_argument("--config", default="configs/segmentation/spbgu_unet.json")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    """Save predicted masks and overlays for rows in a manifest."""
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    pred_cfg = cfg["prediction"]
    checkpoint = args.checkpoint or pred_cfg["checkpoint_path"]
    manifest = args.manifest or pred_cfg["manifest_csv"]
    output_dir = args.output_dir or pred_cfg["output_dir"]
    predictor = SpbguMaskPredictor(checkpoint, device=pred_cfg.get("device", "auto"))
    predictor.save_manifest_predictions(
        manifest,
        output_dir,
        threshold=pred_cfg.get("threshold", 0.5),
        limit=args.limit,
    )
    print(output_dir)


if __name__ == "__main__":
    main()
