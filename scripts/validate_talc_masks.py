"""
Валидация масок талька: один JPEG на кадр (3 панели в ряд).

Запуск: py scripts/validate_talc_masks.py
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
from app.pipeline.talc_mask_builder import build_talc_mask, render_combined_validation

PAIRS_CSV = PROJECT_ROOT / "dataset" / "index" / "talc_pairs.csv"
VALIDATION_DIR = PROJECT_ROOT / "dataset" / "talc_segmentation" / "validation"
MASKS_DIR = PROJECT_ROOT / "dataset" / "talc_segmentation" / "masks"


def safe_stem(filename: str) -> str:
    """Безопасное имя файла."""
    return Path(filename).stem.replace(":", "_")


def process_one(annotated_path: Path, filename: str) -> dict | None:
    """Строит маску и сохраняет validation JPEG + PNG."""
    if not annotated_path.is_file():
        return None

    bgr = imread_unicode(annotated_path)
    if bgr is None:
        return None

    result = build_talc_mask(bgr)
    stem = safe_stem(filename)

    VALIDATION_DIR.mkdir(parents=True, exist_ok=True)
    val_path = VALIDATION_DIR / f"{stem}_validation.jpg"
    cv2.imwrite(str(val_path), render_combined_validation(bgr, result), [int(cv2.IMWRITE_JPEG_QUALITY), 92])

    MASKS_DIR.mkdir(parents=True, exist_ok=True)
    mask_path = MASKS_DIR / f"{stem}.png"
    cv2.imwrite(str(mask_path), result.talc_mask)

    return {
        "filename": filename,
        "talc_percent": result.talc_percent,
        "closure_pixels": int((result.closure_strokes > 0).sum()),
        "validation_jpeg": str(val_path.relative_to(PROJECT_ROOT)),
        "mask_path": str(mask_path.relative_to(PROJECT_ROOT)),
    }


def main() -> None:
    """Обрабатывает пары из dataset/index/talc_pairs.csv."""
    if not PAIRS_CSV.is_file():
        print(f"Run build_dataset first. Missing: {PAIRS_CSV}")
        sys.exit(1)

    results: list[dict] = []
    with PAIRS_CSV.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            annotated = ROOT / "dataset" / row["annotated_path"]
            info = process_one(annotated, row["filename"])
            if info:
                results.append(info)
                print(f"OK {row['filename']}: talc={info['talc_percent']}%")

    summary = VALIDATION_DIR / "summary.csv"
    if results:
        with summary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    print(f"\nDone: {len(results)} images -> {VALIDATION_DIR}")


if __name__ == "__main__":
    main()
