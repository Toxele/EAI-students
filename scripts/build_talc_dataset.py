from __future__ import annotations

import argparse

from data.talc_dataset_builder import TalcSegmentationDatasetBuilder
from hydra.json_config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation/talc_dataset.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    rows = TalcSegmentationDatasetBuilder(cfg).build()
    manual = sum(row["sample_type"] == "positive_manual" for row in rows)
    weak = sum(row["sample_type"] == "positive_weak" for row in rows)
    negatives = sum(row["sample_type"] == "negative_zero" for row in rows)
    print(
        f"wrote {cfg['output_csv']}: "
        f"manual={manual} weak={weak} negatives={negatives} total={len(rows)}"
    )


if __name__ == "__main__":
    main()
