"""
Подготовка файлов для ручной разметки в CVAT.

Создаёт dataset/cvat/:
  to_annotate/   — чистые кадры (без синих линий) + тайлы панорам
  reference_blue/ — те же имена, кадры с синей экспертной разметкой (где есть)

Тайлы панорам: 2272×1704 px (как ch1 OM).

Запуск: py scripts/prepare_cvat_export.py
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline.loader import imread_unicode

SOURCE = ROOT / "dataset" / "source"
OUT = ROOT / "dataset" / "cvat"
TO_ANNOTATE = OUT / "to_annotate"
REFERENCE = OUT / "reference_blue"

# Разрешение эталона (ch1 detail OM)
TILE_W = 2272
TILE_H = 1704

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}


def safe_name(prefix: str, filename: str) -> str:
    """Уникальное имя файла в flat-папке."""
    stem = Path(filename).stem
    ext = Path(filename).suffix.lower()
    if ext not in IMAGE_EXT:
        ext = ".jpg"
    return f"{prefix}__{stem}{ext}"


def copy_image(src: Path, dst: Path) -> None:
    """Копирует или перекодирует изображение в dst."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    bgr = imread_unicode(src)
    if bgr is None:
        shutil.copy2(src, dst)
        return
    cv2.imwrite(str(dst), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])


def collect_ch1(source: Path) -> tuple[list[dict], dict[str, Path]]:
    """
    ch1: оригиналы без «Области оталькования».

    :return: (строки manifest, map clean_name -> annotated path)
    """
    rows: list[dict] = []
    annotated_by_name: dict[str, Path] = {}

    ch1_root = source / "Фото руд по сортам. ч1"
    if not ch1_root.is_dir():
        return rows, annotated_by_name

    ann_dir = ch1_root / "Оталькованные руды" / "Области оталькования"
    if ann_dir.is_dir():
        for p in sorted(ann_dir.iterdir()):
            if p.suffix.lower() in IMAGE_EXT:
                annotated_by_name[p.name] = p

    for p in sorted(ch1_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
            continue
        if "Области оталькования" in p.as_posix():
            continue

        out_name = safe_name("ch1", p.name)
        rel = p.relative_to(source).as_posix()
        dst = TO_ANNOTATE / out_name
        copy_image(p, dst)

        ref_name = ""
        if p.name in annotated_by_name:
            ref_out = REFERENCE / out_name
            copy_image(annotated_by_name[p.name], ref_out)
            ref_name = out_name

        rows.append(
            {
                "filename": out_name,
                "kind": "ch1_detail",
                "source_path": rel,
                "reference_blue": ref_name,
            }
        )

    return rows, annotated_by_name


def collect_ch2(source: Path) -> list[dict]:
    """ch2: все детальные кадры (синей разметки нет)."""
    rows: list[dict] = []
    ch2_root = source / "Фото руд по сортам. ч2"
    if not ch2_root.is_dir():
        return rows

    for p in sorted(ch2_root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
            continue
        out_name = safe_name("ch2", p.name)
        copy_image(p, TO_ANNOTATE / out_name)
        rows.append(
            {
                "filename": out_name,
                "kind": "ch2_detail",
                "source_path": p.relative_to(source).as_posix(),
                "reference_blue": "",
            }
        )
    return rows


def tile_panorama(src: Path, pano_stem: str) -> list[dict]:
    """Режет панораму на тайлы TILE_W×TILE_H без перекрытия."""
    rows: list[dict] = []
    bgr = imread_unicode(src)
    if bgr is None:
        return rows

    h, w = bgr.shape[:2]
    row_idx = 0
    for y in range(0, h - TILE_H + 1, TILE_H):
        col_idx = 0
        for x in range(0, w - TILE_W + 1, TILE_W):
            tile = bgr[y : y + TILE_H, x : x + TILE_W]
            out_name = f"pano__{pano_stem}__r{row_idx:02d}_c{col_idx:02d}.jpg"
            cv2.imwrite(
                str(TO_ANNOTATE / out_name),
                tile,
                [int(cv2.IMWRITE_JPEG_QUALITY), 95],
            )
            rows.append(
                {
                    "filename": out_name,
                    "kind": "panorama_tile",
                    "source_path": f"{src.relative_to(SOURCE).as_posix()} [{x},{y}]",
                    "reference_blue": "",
                }
            )
            col_idx += 1
        row_idx += 1

    return rows


def collect_panoramas(source: Path) -> list[dict]:
    """Все панорамы → сетка тайлов."""
    rows: list[dict] = []
    pano_root = source / "Панорамы"
    if not pano_root.is_dir():
        return rows

    for p in sorted(pano_root.iterdir()):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXT:
            continue
        rows.extend(tile_panorama(p, p.stem))
    return rows


def remove_auto_markup_junk() -> None:
    """Удаляет артеfacts автоматической разметки талька."""
    junk_dirs = [
        ROOT / "dataset" / "talc_segmentation" / "masks",
        ROOT / "dataset" / "talc_segmentation" / "validation",
        ROOT / "dataset" / "talc_segmentation" / "annotated",
        ROOT / "dataset" / "talc_segmentation" / "images",
        ROOT / "data_organized",
    ]
    for path in junk_dirs:
        if path.is_dir():
            shutil.rmtree(path, ignore_errors=True)


def write_readme(manifest_rows: list[dict]) -> None:
    """Короткая подпись к cvat/."""
    n_ref = sum(1 for r in manifest_rows if r["reference_blue"])
    text = f"""# CVAT export — manual talc annotation

| Folder | Purpose |
|--------|---------|
| `to_annotate/` | Clean images for CVAT import ({len(manifest_rows)} files) |
| `reference_blue/` | Same filenames with expert blue strokes ({n_ref} files) |

## Naming

- `ch1__DSCN4708.JPG` — ch1 detail (2272×1704)
- `ch2__58.JPG` — ch2 detail
- `pano__4__r00_c03.jpg` — panorama tile (2272×1704), row/col grid

Open `reference_blue/ch1__….JPG` on a second monitor while annotating the matching file in `to_annotate/`.

Panorama tiles have no blue reference.

## Rebuild

```bash
py scripts/prepare_cvat_export.py
```
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Собирает cvat/ и пишет manifest.csv."""
    if not SOURCE.is_dir():
        raise SystemExit("Run build_dataset.py first (dataset/source/ missing).")

    remove_auto_markup_junk()

    if OUT.exists():
        shutil.rmtree(OUT)
    TO_ANNOTATE.mkdir(parents=True)
    REFERENCE.mkdir(parents=True)

    manifest: list[dict] = []
    manifest.extend(collect_ch1(SOURCE)[0])
    manifest.extend(collect_ch2(SOURCE))
    manifest.extend(collect_panoramas(SOURCE))

    with (OUT / "manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "kind", "source_path", "reference_blue"])
        writer.writeheader()
        writer.writerows(manifest)

    write_readme(manifest)

    kinds = {}
    for row in manifest:
        kinds[row["kind"]] = kinds.get(row["kind"], 0) + 1

    print(f"to_annotate: {len(list(TO_ANNOTATE.iterdir()))} files")
    print(f"reference_blue: {len(list(REFERENCE.iterdir()))} files")
    print("by kind:", kinds)
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()
