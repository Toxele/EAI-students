from __future__ import annotations

import argparse
from pathlib import Path

from hydra.json_config import JsonConfig
from trainers.talc_segmentation_trainer import TalcSegmentationTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation/talc_segmenter.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = JsonConfig.load(args.config).merged(args.overrides)
    cfg = config.to_dict()
    Path(cfg["run_dir"]).mkdir(parents=True, exist_ok=True)
    config.save_resolved(Path(cfg["run_dir"]) / "resolved_config.json")
    print(TalcSegmentationTrainer(cfg).fit())


if __name__ == "__main__":
    main()
