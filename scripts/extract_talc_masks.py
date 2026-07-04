"""
Извлечение PNG-масок талька (использует talc_mask_builder).

Запуск: py scripts/extract_talc_masks.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import PROJECT_ROOT
from app.pipeline.loader import imread_unicode
from app.pipeline.talc_mask_builder import build_talc_mask

PAIRS_CSV = PROJECT_ROOT / "dataset" / "index" / "talc_pairs.csv"
MASKS_DIR = PROJECT_ROOT / "dataset" / "talc_segmentation" / "masks"


def main() -> None:
    """Сохраняет маски для всех пар."""
    if not PAIRS_CSV.is_file():
        print(f"Run build_dataset first. Missing: {PAIRS_CSV}")
        sys.exit(1)

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []

    with PAIRS_CSV.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            path = ROOT / "dataset" / row["annotated_path"]
            stem = Path(row["filename"]).stem
            bgr = imread_unicode(path)
            if bgr is None:
                continue
            result = build_talc_mask(bgr)
            out = MASKS_DIR / f"{stem}.png"
            cv2.imwrite(str(out), result.talc_mask)
            results.append(
                {
                    "filename": stem,
                    "talc_percent": result.talc_percent,
                    "mask_path": str(out.relative_to(PROJECT_ROOT)),
                }
            )

    summary = MASKS_DIR / "summary.csv"
    with summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "talc_percent", "mask_path"])
        writer.writeheader()
        writer.writerows(results)

    print(f"Masks: {len(results)} -> {MASKS_DIR}")


if __name__ == "__main__":
    main()
