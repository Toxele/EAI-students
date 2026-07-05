"""
Импорт разметки CVAT (XML 1.1) → маски, overlay-превью, CSV.

Читает dataset/annotations/*/annotations.xml,
связывает с dataset/cvat/manifest.csv.

Запуск: py scripts/import_cvat_annotations.py
"""
from __future__ import annotations

import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT.parent))  # repo root, so training.* and app.* import cleanly

from app.pipeline.loader import imread_unicode

ANNOT_DIR = ROOT / "dataset" / "annotations"
CVAT_DIR = ROOT / "dataset" / "cvat"
MANIFEST = CVAT_DIR / "manifest.csv"
OUT = ANNOT_DIR / "organized"
MASKS = OUT / "masks"
OVERLAYS = OUT / "overlays"

# CVAT label → наш класс
LABEL_MAP = {"talk": "talc", "talc": "talc"}
TALC_BGR = (200, 120, 40)
OVERLAY_ALPHA = 0.45


def load_manifest() -> dict[str, dict]:
    """cvat_filename → строка manifest."""
    by_name: dict[str, dict] = {}
    if not MANIFEST.is_file():
        return by_name
    with MANIFEST.open(encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            by_name[row["cvat_filename"]] = row
    return by_name


def parse_points(text: str) -> np.ndarray:
    """CVAT points 'x,y;x,y' → array N×2 int32."""
    pairs = []
    for part in text.strip().split(";"):
        if not part:
            continue
        x, y = part.split(",")
        pairs.append([int(float(x)), int(float(y))])
    return np.array(pairs, dtype=np.int32)


def parse_cvat_xml(path: Path) -> dict:
    """Парсит annotations.xml → структура для экспорта."""
    root = ET.parse(path).getroot()
    source_name = path.parent.name

    raw_label = root.find(".//label/name")
    cvat_label = raw_label.text if raw_label is not None else "unknown"
    class_name = LABEL_MAP.get(cvat_label, cvat_label)

    images: list[dict] = []
    for img_el in root.findall("image"):
        name = img_el.get("name", "")
        width = int(img_el.get("width", 0))
        height = int(img_el.get("height", 0))
        polygons: list[dict] = []
        for poly in img_el.findall("polygon"):
            pts = parse_points(poly.get("points", ""))
            if len(pts) < 3:
                continue
            polygons.append(
                {
                    "label": LABEL_MAP.get(poly.get("label", cvat_label), class_name),
                    "points": pts,
                    "n_vertices": len(pts),
                }
            )
        if polygons:
            images.append(
                {
                    "cvat_filename": name,
                    "width": width,
                    "height": height,
                    "polygons": polygons,
                    "polygon_count": len(polygons),
                }
            )

    return {
        "source": source_name,
        "cvat_label": cvat_label,
        "class_name": class_name,
        "annotated_images": images,
    }


def polygon_area(pts: np.ndarray) -> float:
    """Площадь полигона (px²)."""
    return float(cv2.contourArea(pts.reshape(-1, 1, 2)))


def build_mask(height: int, width: int, polygons: list[dict]) -> np.ndarray:
    """Бинарная маска 0/255 из списка полигонов."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for poly in polygons:
        cv2.fillPoly(mask, [poly["points"]], 255)
    return mask


def render_overlay(bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Полупрозрачная маска талька поверх фото."""
    out = bgr.copy().astype(np.float32)
    px = mask > 0
    color = np.array(TALC_BGR, dtype=np.float32)
    out[px] = (1 - OVERLAY_ALPHA) * out[px] + OVERLAY_ALPHA * color
    return out.astype(np.uint8)


def process_source(parsed: dict, manifest: dict[str, dict]) -> list[dict]:
    """
    Строит маски и overlay для одного CVAT-экспорта.

    :return: строки для labels.csv
    """
    rows: list[dict] = []
    source = parsed["source"]

    for item in parsed["annotated_images"]:
        cvat_name = item["cvat_filename"]
        meta = manifest.get(cvat_name, {})
        image_path = CVAT_DIR / "to_annotate" / cvat_name
        bgr = imread_unicode(image_path)
        if bgr is None:
            continue

        h, w = bgr.shape[:2]
        mask = build_mask(h, w, item["polygons"])
        talc_px = int(np.count_nonzero(mask))
        talc_pct = round(100.0 * talc_px / (h * w), 2)

        stem = Path(cvat_name).stem
        mask_path = MASKS / f"{source}__{stem}.png"
        overlay_path = OVERLAYS / f"{source}__{stem}.jpg"

        cv2.imwrite(str(mask_path), mask)
        cv2.imwrite(
            str(overlay_path),
            render_overlay(bgr, mask),
            [int(cv2.IMWRITE_JPEG_QUALITY), 92],
        )

        for poly_idx, poly in enumerate(item["polygons"], start=1):
            rows.append(
                {
                    "source_export": source,
                    "cvat_filename": cvat_name,
                    "id": meta.get("id", ""),
                    "original_filename": meta.get("original_filename", ""),
                    "kind": meta.get("kind", ""),
                    "source_path": meta.get("source_path", ""),
                    "class": poly["label"],
                    "polygon_index": poly_idx,
                    "vertices": poly["n_vertices"],
                    "area_px": round(polygon_area(poly["points"]), 1),
                    "talc_percent_image": talc_pct,
                    "mask_file": str(mask_path.relative_to(OUT)),
                    "overlay_file": str(overlay_path.relative_to(OUT)),
                }
            )

    return rows


def write_readme(sources: list[dict], all_rows: list[dict]) -> None:
    """README для organized/."""
    n_images = len({(r["source_export"], r["cvat_filename"]) for r in all_rows})
    text = f"""# CVAT annotations (organized)

Imported from CVAT XML exports in `../task_2398159/` and `../job_4195731/`.

| Folder | Content |
|--------|---------|
| `masks/` | Binary PNG masks (talc = white) |
| `overlays/` | Preview: talc overlay on source image |
| `labels.csv` | All polygons + link to manifest |
| `summary.json` | Counts per export |

## Label note

CVAT label **`talk`** → stored as **`talc`** (typo in CVAT task).

## Your exports

"""
    for src in sources:
        n_img = len(src["annotated_images"])
        n_poly = sum(i["polygon_count"] for i in src["annotated_images"])
        text += f"- **{src['source']}**: {n_img} images, {n_poly} polygons\n"

    text += f"""
## Totals

- Annotated images: {n_images}
- Polygon rows in CSV: {len(all_rows)}

## Mapping

Join with `../../cvat/manifest.csv` on `cvat_filename`.

Regenerate: `py scripts/import_cvat_annotations.py`
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    """Импортирует все annotations.xml из dataset/annotations/."""
    manifest = load_manifest()

    if OUT.exists():
        import shutil

        shutil.rmtree(OUT)
    MASKS.mkdir(parents=True)
    OVERLAYS.mkdir(parents=True)

    all_rows: list[dict] = []
    sources: list[dict] = []

    for xml_path in sorted(ANNOT_DIR.glob("*/annotations.xml")):
        parsed = parse_cvat_xml(xml_path)
        sources.append(parsed)
        all_rows.extend(process_source(parsed, manifest))

    # labels.csv
    if all_rows:
        fields = list(all_rows[0].keys())
        with (OUT / "labels.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)

    # summary.json
    summary = {
        "exports": [
            {
                "source": s["source"],
                "cvat_label": s["cvat_label"],
                "class_name": s["class_name"],
                "annotated_images": len(s["annotated_images"]),
                "polygons": sum(i["polygon_count"] for i in s["annotated_images"]),
                "files": [i["cvat_filename"] for i in s["annotated_images"]],
            }
            for s in sources
        ],
        "unique_images": len({(r["source_export"], r["cvat_filename"]) for r in all_rows}),
        "total_polygons": len(all_rows),
    }
    with (OUT / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    write_readme(sources, all_rows)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"\nOutput: {OUT}")


if __name__ == "__main__":
    main()
