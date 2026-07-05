"""Batch-analyze every image in a folder.

Reuses the same analyze_upload() path as the API, so each file gets the
usual outputs in results/ (overlay JPG, CSV, labels.json) plus a combined
batch_summary.csv with one row per file.

Run: py scripts/batch_analyze.py path/to/folder
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")  # sort_label_ru is Cyrillic; default Windows console codepage can't print it

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.api.routes import analyze_upload
from app.config import RESULTS_DIR

IMAGE_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder of images to analyze")
    parser.add_argument(
        "--summary",
        type=Path,
        default=RESULTS_DIR / "batch_summary.csv",
        help="Where to write the combined summary CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    images = sorted(p for p in args.folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise SystemExit(f"No images found in {args.folder}")

    rows: list[dict[str, object]] = []
    for path in images:
        start = time.monotonic()
        try:
            result = analyze_upload(path.read_bytes(), path.name)
        except ValueError as exc:
            print(f"SKIP {path.name}: {exc}")
            continue
        elapsed = time.monotonic() - start

        print(f"{path.name}: mode={result.mode} sort={result.sort_label_ru} ({elapsed:.1f}s)")
        rows.append(
            {
                "filename": path.name,
                "result_id": result.result_id,
                "mode": result.mode,
                "sort_code": result.sort_code,
                "sort_label_ru": result.sort_label_ru,
                "sulfide_percent": result.sulfide_percent,
                "ordinary_percent": result.ordinary_percent,
                "thin_percent": result.thin_percent,
                "talc_percent": result.talc_percent,
                "seconds": round(elapsed, 1),
            }
        )

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nDone: {len(rows)} files, summary at {args.summary}")


if __name__ == "__main__":
    main()
