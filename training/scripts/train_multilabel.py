from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))  # repo root, so training.* and app.* import cleanly

from training.hydra.json_config import JsonConfig
from training.trainers.multilabel_trainer import MultiLabelTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classifier/multilabel_resnet18.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    config = JsonConfig.load(args.config).merged(args.overrides)
    cfg = config.to_dict()
    Path(cfg["run_dir"]).mkdir(parents=True, exist_ok=True)
    config.save_resolved(Path(cfg["run_dir"]) / "resolved_config.json")
    summary = MultiLabelTrainer(cfg).fit()
    print(summary)


if __name__ == "__main__":
    main()
