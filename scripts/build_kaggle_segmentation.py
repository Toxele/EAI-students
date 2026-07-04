"""
Сборка Kaggle-датасета для сегментации талька.

Берёт разметку из dataset/annotations/organized/,
копирует пары image + mask в dataset/kaggle/talc_segmentation/,
делает train/val split, упаковывает zip.

Запуск: py scripts/build_kaggle_segmentation.py
"""
from __future__ import annotations

import csv
import json
import shutil
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline.loader import imread_unicode

ORGANIZED = ROOT / "dataset" / "annotations" / "organized"
CVAT_IMAGES = ROOT / "dataset" / "cvat" / "to_annotate"
MANIFEST = ROOT / "dataset" / "cvat" / "manifest.csv"
OUT_DIR = ROOT / "dataset" / "kaggle" / "talc_segmentation"
ZIP_PATH = ROOT / "dataset" / "kaggle" / "nornickel_talc_segmentation.zip"

VAL_RATIO = 0.18
RANDOM_SEED = 42


def load_manifest() -> dict[str, dict]:
    """cvat_filename → метаданные."""
    by_name: dict[str, dict] = {}
    with MANIFEST.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            by_name[row["cvat_filename"]] = row
    return by_name


def collect_samples() -> list[dict]:
    """
    Собирает список размеченных кадров.

    :return: записи с путями к image/mask и метаданными
    """
    manifest = load_manifest()
    labels_path = ORGANIZED / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError("Сначала: py scripts/import_cvat_annotations.py")

    # Уникальные cvat_filename из CSV
    seen: set[str] = set()
    rows: list[dict] = []
    with labels_path.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            name = row["cvat_filename"]
            if name in seen:
                continue
            seen.add(name)

            stem = Path(name).stem
            mask_files = list((ORGANIZED / "masks").glob(f"*__{stem}.png"))
            if not mask_files:
                continue

            image_path = CVAT_IMAGES / name
            if not image_path.is_file():
                continue

            meta = manifest.get(name, {})
            mask = cv2.imread(str(mask_files[0]), cv2.IMREAD_GRAYSCALE)
            talc_px = int(np.count_nonzero(mask)) if mask is not None else 0
            total_px = mask.size if mask is not None else 1
            talc_pct = round(100.0 * talc_px / total_px, 2)

            rows.append(
                {
                    "sample_id": stem,
                    "cvat_filename": name,
                    "kind": meta.get("kind", row.get("kind", "")),
                    "original_filename": meta.get("original_filename", ""),
                    "source_path": meta.get("source_path", ""),
                    "talc_percent": talc_pct,
                    "image_src": str(image_path),
                    "mask_src": str(mask_files[0]),
                }
            )

    rows.sort(key=lambda r: int(r["sample_id"]))
    return rows


def assign_splits(samples: list[dict]) -> list[dict]:
    """Stratified train/val по kind (ch1 vs panorama_tile)."""
    kinds = [s["kind"] or "unknown" for s in samples]
    indices = list(range(len(samples)))
    train_idx, val_idx = train_test_split(
        indices,
        test_size=VAL_RATIO,
        random_state=RANDOM_SEED,
        stratify=kinds,
    )
    val_set = set(val_idx)
    for i, sample in enumerate(samples):
        sample["split"] = "val" if i in val_set else "train"
    return samples


def write_dataset(samples: list[dict]) -> None:
    """Копирует файлы и пишет metadata."""
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    images_dir = OUT_DIR / "images"
    masks_dir = OUT_DIR / "masks"
    splits_dir = OUT_DIR / "splits"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)
    splits_dir.mkdir(parents=True)

    train_ids: list[str] = []
    val_ids: list[str] = []

    for sample in samples:
        sid = sample["sample_id"]
        shutil.copy2(sample["image_src"], images_dir / f"{sid}.jpg")
        shutil.copy2(sample["mask_src"], masks_dir / f"{sid}.png")
        if sample["split"] == "train":
            train_ids.append(sid)
        else:
            val_ids.append(sid)

    (splits_dir / "train.txt").write_text("\n".join(train_ids) + "\n", encoding="utf-8")
    (splits_dir / "val.txt").write_text("\n".join(val_ids) + "\n", encoding="utf-8")

    meta_fields = [
        "sample_id",
        "cvat_filename",
        "kind",
        "original_filename",
        "source_path",
        "talc_percent",
        "split",
    ]
    # QUOTE_NONNUMERIC: sample_id "000030" не превращается в 30 при чтении pandas
    with (OUT_DIR / "metadata.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=meta_fields,
            quoting=csv.QUOTE_NONNUMERIC,
        )
        writer.writeheader()
        for sample in samples:
            writer.writerow({k: sample[k] for k in meta_fields})

    readme = f"""# Nornickel talc segmentation (Kaggle)

Binary segmentation: **talc** vs background on OM microscopy images.

| Split | Images |
|-------|--------|
| train | {len(train_ids)} |
| val   | {len(val_ids)} |
| total | {len(samples)} |

## Layout

```
images/{{sample_id}}.jpg   # 2272×1704 RGB
masks/{{sample_id}}.png    # 0=bg, 255=talc
metadata.csv
splits/train.txt
splits/val.txt
```

## Kinds

- `ch1_detail` — detailed OM (CVAT task export)
- `panorama_tile` — 4×4 tile from panorama 4.jpg

Upload as Kaggle Dataset, then open `kaggle/talc_segmentation_train.ipynb`.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")

    summary = {
        "total": len(samples),
        "train": len(train_ids),
        "val": len(val_ids),
        "kinds": {},
    }
    for sample in samples:
        kind = sample["kind"] or "unknown"
        summary["kinds"].setdefault(kind, {"train": 0, "val": 0})
        summary["kinds"][kind][sample["split"]] += 1

    with (OUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def build_zip() -> None:
    """Упаковывает OUT_DIR в zip для загрузки на Kaggle."""
    ZIP_PATH.parent.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()

    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(OUT_DIR.rglob("*")):
            if path.is_file():
                arcname = path.relative_to(OUT_DIR.parent)
                zf.write(path, arcname)


def main() -> None:
    """Точка входа."""
    samples = collect_samples()
    if not samples:
        raise SystemExit("Нет размеченных пар image/mask")

    samples = assign_splits(samples)
    write_dataset(samples)
    build_zip()

    train_n = sum(1 for s in samples if s["split"] == "train")
    val_n = len(samples) - train_n
    print(f"Samples: {len(samples)} (train={train_n}, val={val_n})")
    print(f"Folder: {OUT_DIR}")
    print(f"Zip:    {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
