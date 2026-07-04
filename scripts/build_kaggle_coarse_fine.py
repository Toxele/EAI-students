"""
Kaggle-датасет для coarse/fine multi-label классификатора (только данные).

Создаёт:
  dataset/kaggle/coarse_fine_classification/  — images + manifest
  dataset/kaggle/nornickel_coarse_fine_classification.zip

Код обучения — в kaggle/train_coarse_fine_multilabel.ipynb (встроен, без code zip).

Запуск: py scripts/build_kaggle_coarse_fine.py
        py scripts/build_kaggle_coarse_fine.py --skip-images
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.build_coarse_fine_manifest import load_rows, split_rows  # noqa: E402

OUT_DIR = ROOT / "dataset" / "kaggle" / "coarse_fine_classification"
DATA_ZIP = ROOT / "dataset" / "kaggle" / "nornickel_coarse_fine_classification.zip"


def resolve_data_path(rel_path: str) -> Path:
    """Абсолютный путь к файлу в data/."""
    candidate = ROOT / rel_path
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(rel_path)


def safe_rmtree(path: Path) -> None:
    """Удаляет путь (файл или папку), с повторами на Windows."""
    if not path.exists():
        return
    for _ in range(5):
        try:
            if path.is_file() or path.is_symlink():
                path.unlink()
            else:
                shutil.rmtree(path)
            return
        except OSError:
            time.sleep(2)
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink(missing_ok=True)


def copy_with_retry(src: Path, dst: Path, retries: int = 5) -> None:
    """Копирует файл с повторами (Windows lock / AV)."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            if dst.is_file() and dst.stat().st_size == src.stat().st_size:
                return
            shutil.copy2(src, dst)
            if dst.is_file():
                return
        except OSError as exc:
            last_err = exc
            time.sleep(1.5 * (attempt + 1))
    raise OSError(f"copy failed after {retries} tries: {src} -> {dst}") from last_err


def build_dataset(skip_images: bool = False, val_fraction: float = 0.2, seed: int = 42, clean: bool = True) -> list[dict[str, str]]:
    """Копирует кадры в images/ и пишет manifest.csv."""
    index_manifest = ROOT / "dataset" / "index" / "manifest.csv"
    rows = split_rows(load_rows(index_manifest), val_fraction=val_fraction, seed=seed)

    images_dir = OUT_DIR / "images"
    if clean and OUT_DIR.exists():
        safe_rmtree(OUT_DIR)
    images_dir.mkdir(parents=True, exist_ok=True)

    out_rows: list[dict[str, str]] = []
    copied = 0
    for row in rows:
        src = resolve_data_path(row["path"])
        suffix = src.suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".tif", ".tiff"}:
            suffix = ".jpg"
        rel_image = f"images/{row['md5']}{suffix}"
        dst = OUT_DIR / rel_image
        if not skip_images:
            copy_with_retry(src, dst)
            copied += 1
            if copied % 100 == 0:
                print(f"  copied {copied}/{len(rows)}...")
        out_rows.append(
            {
                "md5": row["md5"],
                "rel_path": rel_image,
                "tags": row["tags"],
                "tag_talc": row["tag_talc"],
                "tag_coarse": row["tag_coarse"],
                "tag_fine": row["tag_fine"],
                "ig_label": row["ig_label"],
                "subset": row["subset"],
            }
        )

    fields = [
        "md5",
        "rel_path",
        "tags",
        "tag_talc",
        "tag_coarse",
        "tag_fine",
        "ig_label",
        "subset",
    ]
    with (OUT_DIR / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    train_n = sum(r["subset"] == "train" for r in out_rows)
    val_n = len(out_rows) - train_n
    summary = {
        "total": len(out_rows),
        "train": train_n,
        "val": val_n,
        "coarse_train": sum(r["subset"] == "train" and r["ig_label"] == "coarse" for r in out_rows),
        "fine_train": sum(r["subset"] == "train" and r["ig_label"] == "fine" for r in out_rows),
        "with_talc_tag": sum(r["tag_talc"] == "1" for r in out_rows),
        "skip_images": skip_images,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    readme = f"""# Nornickel coarse/fine classification (Kaggle)

Multi-label: **coarse** (рядовая) / **fine** (тонкая). Talc-тег в manifest — только метаданные.

| Split | Images |
|-------|--------|
| train | {train_n} |
| val   | {val_n} |
| total | {len(out_rows)} |

## Layout

```
images/{{md5}}.jpg
manifest.csv
summary.json
```

Upload as Kaggle Dataset. Код — в `kaggle/train_coarse_fine_multilabel.ipynb`.
"""
    (OUT_DIR / "README.md").write_text(readme, encoding="utf-8")
    if not skip_images:
        print(f"  copied {copied} images total")
    return out_rows


def zip_folder(folder: Path, zip_path: Path, arc_prefix: Path | None = None) -> None:
    """Папка → zip с повторами при PermissionError."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    if zip_path.exists():
        zip_path.unlink()
    prefix = arc_prefix or folder.parent
    files = sorted(p for p in folder.rglob("*") if p.is_file())
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            if zip_path.exists():
                zip_path.unlink()
            with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for path in files:
                    zf.write(path, path.relative_to(prefix).as_posix())
            return
        except (PermissionError, OSError) as exc:
            last_err = exc
            time.sleep(2.0 * (attempt + 1))
    raise OSError(f"zip failed: {zip_path}") from last_err


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-images", action="store_true", help="manifest only, no copy")
    parser.add_argument("--no-clean", action="store_true", help="не удалять OUT_DIR перед копированием")
    args = parser.parse_args()

    rows = build_dataset(skip_images=args.skip_images, clean=not args.no_clean)
    if not args.skip_images:
        zip_folder(OUT_DIR, DATA_ZIP, arc_prefix=OUT_DIR.parent)
        size_mb = DATA_ZIP.stat().st_size / 1e6
        print(f"Data: {OUT_DIR}")
        print(f"Zip:  {DATA_ZIP} ({size_mb:.1f} MB), samples={len(rows)}")
    else:
        print(f"Manifest only: {OUT_DIR / 'manifest.csv'} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
