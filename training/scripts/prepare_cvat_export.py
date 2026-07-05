"""
Подготовка файлов для ручной разметки в CVAT.

Создаёт dataset/cvat/:
  to_annotate/    — 000001.jpg, 000002.jpg, … (чистые кадры + тайлы панорам)
  reference_blue/ — те же номера, где есть синяя экспертная разметка
  manifest.csv    — связь номер ↔ оригинальное имя ↔ путь в source/

Тайлы панорам: 2272×1704 px (как ch1 OM).

Запуск: py scripts/prepare_cvat_export.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))  # repo root, so training.* and app.* import cleanly

from app.pipeline.loader import imread_unicode

SOURCE = ROOT / "dataset" / "source"
OUT = ROOT / "dataset" / "cvat"
TO_ANNOTATE = OUT / "to_annotate"
REFERENCE = OUT / "reference_blue"

# Выходной размер (как ch1 OM)
TILE_W = 2272
TILE_H = 1704

# Каждую панораму — сетка 4×4 = 16 тайлов, каждый → TILE_W×TILE_H
PANO_GRID = 4
IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


@dataclass
class ExportItem:
    """Один файл для экспорта до присвоения номера."""

    kind: str
    original_filename: str
    source_path: str
    clean_src: Path
    reference_src: Path | None = None
    extra: dict = field(default_factory=dict)


def numbered_name(index: int) -> str:
    """Имя файла для CVAT: 000001.jpg."""
    return f"{index:06d}.jpg"


def save_bgr(bgr, dst: Path) -> None:
    """Сохраняет BGR как JPEG."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(dst), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def copy_as_jpeg(src: Path, dst: Path) -> None:
    """Читает изображение и сохраняет как JPEG."""
    bgr = imread_unicode(src)
    if bgr is None:
        shutil.copy2(src, dst)
        return
    save_bgr(bgr, dst)


def scan_ch1(source: Path) -> list[ExportItem]:
    """ch1 оригиналы без папки «Области оталькования»."""
    items: list[ExportItem] = []
    ch1_root = source / "Фото руд по сортам. ч1"
    if not ch1_root.is_dir():
        return items

    annotated: dict[str, Path] = {}
    ann_dir = ch1_root / "Оталькованные руды" / "Области оталькования"
    if ann_dir.is_dir():
        for p in sorted(ann_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXT:
                annotated[p.name] = p

    for p in sorted(ch1_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
            continue
        if "Области оталькования" in p.as_posix():
            continue
        items.append(
            ExportItem(
                kind="ch1_detail",
                original_filename=p.name,
                source_path=p.relative_to(source).as_posix(),
                clean_src=p,
                reference_src=annotated.get(p.name),
            )
        )
    return items


def scan_ch2(source: Path) -> list[ExportItem]:
    """ch2 — все детальные кадры."""
    items: list[ExportItem] = []
    ch2_root = source / "Фото руд по сортам. ч2"
    if not ch2_root.is_dir():
        return items

    for p in sorted(ch2_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
            continue
        items.append(
            ExportItem(
                kind="ch2_detail",
                original_filename=p.name,
                source_path=p.relative_to(source).as_posix(),
                clean_src=p,
            )
        )
    return items


def _pano_sort_key(path: Path) -> tuple:
    """Сортировка панорам: 4, 5, … 10, 11 (по числу в имени)."""
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem))
    return (1, stem)


def scan_panorama_tiles(source: Path) -> list[ExportItem]:
    """
    Каждую панораму делит на 4×4 = 16 частей, ресайз каждой до TILE_W×TILE_H.
    """
    items: list[ExportItem] = []
    pano_root = source / "Панорамы"
    if not pano_root.is_dir():
        return items

    pano_files = sorted(
        (p for p in pano_root.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXT),
        key=_pano_sort_key,
    )

    for pano_path in pano_files:
        bgr = imread_unicode(pano_path)
        if bgr is None:
            continue

        h, w = bgr.shape[:2]
        pano_rel = pano_path.relative_to(source).as_posix()

        for row in range(PANO_GRID):
            y0 = row * h // PANO_GRID
            y1 = (row + 1) * h // PANO_GRID
            for col in range(PANO_GRID):
                x0 = col * w // PANO_GRID
                x1 = (col + 1) * w // PANO_GRID
                crop = bgr[y0:y1, x0:x1]
                tile = cv2.resize(crop, (TILE_W, TILE_H), interpolation=cv2.INTER_AREA)
                orig = f"{pano_path.stem}_tile_r{row:02d}_c{col:02d}.jpg"
                items.append(
                    ExportItem(
                        kind="panorama_tile",
                        original_filename=orig,
                        source_path=f"{pano_rel} grid {row},{col} of 4x4 px [{x0}:{x1},{y0}:{y1}]",
                        clean_src=pano_path,
                        extra={"tile_bgr": tile.copy()},
                    )
                )

    return items


def write_export(items: list[ExportItem]) -> list[dict]:
    """
    Присваивает номера, пишет файлы, возвращает строки manifest.

    Порядок items = порядок номеров. reference_blue использует тот же номер.
    """
    manifest: list[dict] = []

    for idx, item in enumerate(items, start=1):
        cvat_name = numbered_name(idx)
        clean_dst = TO_ANNOTATE / cvat_name

        if item.kind == "panorama_tile" and "tile_bgr" in item.extra:
            save_bgr(item.extra["tile_bgr"], clean_dst)
        else:
            copy_as_jpeg(item.clean_src, clean_dst)

        ref_name = ""
        ref_source = ""
        if item.reference_src is not None:
            ref_name = cvat_name
            ref_source = item.reference_src.relative_to(SOURCE).as_posix()
            copy_as_jpeg(item.reference_src, REFERENCE / cvat_name)

        manifest.append(
            {
                "id": idx,
                "cvat_filename": cvat_name,
                "original_filename": item.original_filename,
                "kind": item.kind,
                "source_path": item.source_path,
                "has_reference_blue": "yes" if ref_name else "no",
                "reference_source_path": ref_source,
            }
        )

    return manifest


def remove_auto_markup_junk() -> None:
    """Удаляет артеfacts автоматической разметки."""
    for path in [
        ROOT / "dataset" / "talc_segmentation",
        ROOT / "data_organized",
    ]:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def write_readme(manifest: list[dict]) -> None:
    """Короткая подпись."""
    n_ref = sum(1 for r in manifest if r["has_reference_blue"] == "yes")
    text = f"""# CVAT export

| Folder | Files |
|--------|-------|
| `to_annotate/` | {len(manifest)} numbered images (`000001.jpg` …) |
| `reference_blue/` | {n_ref} same numbers where blue markup exists |
| `manifest.csv` | id ↔ original filename ↔ source path |

## How to use

1. Import **`to_annotate/`** into CVAT.
2. On a second screen open **`reference_blue/000042.jpg`** for the same **`to_annotate/000042.jpg`** (only 42 ch1 files have a pair).
3. After export from CVAT, merge labels back via **`manifest.csv`** (`id` / `original_filename` / `source_path`).

Numbering order: ch1 → ch2 → panorama tiles (2× zoom-out vs ch1 detail).

Panorama: each file → 16 tiles (4×4 grid), resized to 2272×1704. Order: pano 4, 5, … 17.

Rebuild: `py scripts/prepare_cvat_export.py`
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Собирает cvat/."""
    if not SOURCE.is_dir():
        raise SystemExit("Run build_dataset.py first (dataset/source/ missing).")

    remove_auto_markup_junk()

    if OUT.exists():
        shutil.rmtree(OUT)
    TO_ANNOTATE.mkdir(parents=True)
    REFERENCE.mkdir(parents=True)

    items: list[ExportItem] = []
    items.extend(scan_ch1(SOURCE))
    items.extend(scan_ch2(SOURCE))
    items.extend(scan_panorama_tiles(SOURCE))

    manifest = write_export(items)

    fields = [
        "id",
        "cvat_filename",
        "original_filename",
        "kind",
        "source_path",
        "has_reference_blue",
        "reference_source_path",
    ]
    with (OUT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(manifest)

    write_readme(manifest)

    n_ref = sum(1 for r in manifest if r["has_reference_blue"] == "yes")
    print(f"to_annotate: {len(manifest)}")
    print(f"reference_blue: {n_ref}")
    print(f"manifest: {OUT / 'manifest.csv'}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
