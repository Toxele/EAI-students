from __future__ import annotations

import argparse
import csv
from pathlib import Path

from training.data.weak_masks import WeakMaskBatchExporter
from training.hydra.json_config import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data/weak_talc_masks.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    exporter = WeakMaskBatchExporter.from_config(cfg)
    results = exporter.export(cfg["input_root"], cfg["output_root"])
    report_path = Path(cfg["output_root"]) / "weak_mask_report.csv"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].__dict__.keys()) if results else [])
        if results:
            writer.writeheader()
            writer.writerows([item.__dict__ for item in results])
    print(f"exported weak masks: {len(results)} -> {cfg['output_root']}")


if __name__ == "__main__":
    main()

