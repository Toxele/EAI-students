from __future__ import annotations

import argparse
from pathlib import Path

from training.hydra.json_config import JsonConfig
from training.trainers.classification_trainer import ClassificationTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classifier/nornikel_classifier.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = JsonConfig.load(args.config).merged(args.overrides)
    cfg = config.to_dict()
    Path(cfg["run_dir"]).mkdir(parents=True, exist_ok=True)
    config.save_resolved(Path(cfg["run_dir"]) / "resolved_config.json")
    summary = ClassificationTrainer(cfg).fit()
    print(summary)


if __name__ == "__main__":
    main()

