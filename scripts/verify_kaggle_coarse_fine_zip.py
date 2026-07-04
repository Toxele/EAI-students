"""
Проверка целостности nornickel_coarse_fine_classification.zip.

Запуск: py scripts/verify_kaggle_coarse_fine_zip.py
"""
from __future__ import annotations

import csv
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_ZIP = ROOT / "dataset" / "kaggle" / "nornickel_coarse_fine_classification.zip"


def main() -> int:
    if not DATA_ZIP.is_file():
        print(f"MISSING: {DATA_ZIP}")
        return 1

    size_mb = DATA_ZIP.stat().st_size / 1e6
    print(f"Zip: {DATA_ZIP} ({size_mb:.1f} MB)")

    with zipfile.ZipFile(DATA_ZIP) as zf:
        bad = zf.testzip()
        if bad:
            print(f"CORRUPT entry: {bad}")
            return 1

        names = set(zf.namelist())
        manifest_paths = [n for n in names if n.endswith("manifest.csv")]
        if not manifest_paths:
            print("ERROR: manifest.csv not found in zip")
            return 1

        manifest_path = manifest_paths[0]
        prefix = manifest_path.rsplit("/", 1)[0] + "/" if "/" in manifest_path else ""
        text = zf.read(manifest_path).decode("utf-8-sig")
        rows = list(csv.DictReader(text.splitlines()))

        missing_images = 0
        for row in rows:
            rel = row["rel_path"]
            full = prefix + rel if prefix else rel
            if full not in names and rel not in names:
                missing_images += 1

        train = sum(r["subset"] == "train" for r in rows)
        val = sum(r["subset"] == "val" for r in rows)
        both = sum(r["tag_coarse"] == "1" and r["tag_fine"] == "1" for r in rows)
        talc = sum(r["tag_talc"] == "1" for r in rows)
        images = sum(1 for n in names if "/images/" in n or n.startswith("images/"))

        print(f"OK: {len(rows)} rows (train={train}, val={val})")
        print(f"    images in zip={images}, missing={missing_images}, coarse+fine both={both}, talc_tag={talc}")
        print(f"    manifest: {manifest_path}")

        if missing_images:
            print("ERROR: some manifest rows have no image in zip")
            return 1

    print("Verification passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
