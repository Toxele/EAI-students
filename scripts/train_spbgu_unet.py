from __future__ import annotations

import argparse
import json

from hydra.json_config import load_config
from trainers.spbgu_unet_trainer import SpbguCrossValidator, SpbguUNetTrainer


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for holdout and cross-validation training."""
    parser = argparse.ArgumentParser(description="Train SPbGU U-Net segmenter.")
    parser.add_argument("--config", default="configs/segmentation/spbgu_unet.json")
    parser.add_argument("--cv", action="store_true", help="Run configured k-fold cross-validation.")
    parser.add_argument("overrides", nargs="*")
    return parser.parse_args()


def main() -> None:
    """Train a SPbGU segmentation model and print a compact JSON summary."""
    args = parse_args()
    cfg = load_config(args.config, args.overrides)
    if args.cv:
        result = SpbguCrossValidator(cfg["training"]).run()
    else:
        result = SpbguUNetTrainer(cfg["training"]).fit()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
