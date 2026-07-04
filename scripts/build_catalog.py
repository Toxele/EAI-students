"""Build structured catalog for Nornickel ore microscopy dataset."""
import csv
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS

BASE = Path(r"c:\Users\Professional\PycharmProjects\Nornickel\data")
OUT = Path(r"c:\Users\Professional\PycharmProjects\Nornickel\data_organized")

MAG_PATTERNS = [
    re.compile(r"(\d+)\s*[xх×]\b", re.I),
    re.compile(r"(\d+)\s*[xх×]\.", re.I),
    re.compile(r"\b(\d+)x\b", re.I),
]


def parse_mag(name: str) -> int | None:
    for pattern in MAG_PATTERNS:
        match = pattern.search(name)
        if match:
            return int(match.group(1))
    return None


def get_image_info(path: Path) -> dict:
    try:
        with Image.open(path) as img:
            exif_raw = img.getexif() or {}
            exif = {TAGS.get(k, k): str(v)[:200] for k, v in exif_raw.items() if k in TAGS}
            return {
                "width": img.width,
                "height": img.height,
                "mode": img.mode,
                "format": img.format,
                "exif": exif,
            }
    except Exception as exc:
        return {"error": str(exc)}


def file_hash(path: Path, block: int = 65536) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        while chunk := handle.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def estimate_fov_mm(width_px: int, magnification: int | None) -> str | None:
    """Rough field-of-view estimate for reflected-light microscopy."""
    if not magnification or not width_px:
        return None
    # Typical microscope FOV at 10x ~ 1.2-1.5 mm; scales ~ linearly with 1/mag.
    fov_mm = 12.0 / magnification
    return fov_mm


def main() -> None:
    OUT.mkdir(exist_ok=True)

    records: list[dict] = []
    hash_map: dict[str, list[str]] = {}

    for root, _, files in os.walk(BASE):
        for filename in sorted(files):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".tiff")):
                continue

            path = Path(root) / filename
            rel = path.relative_to(BASE).as_posix()
            parts = rel.split("/")
            digest = file_hash(path)

            record = {
                "path": str(path),
                "relative_path": rel,
                "filename": filename,
                "dataset_part": parts[0] if parts else "",
                "label_folder": parts[1] if len(parts) > 1 else "",
                "subfolder": parts[2] if len(parts) > 2 else "",
                "label_ore_sort": None,
                "label_intergrowth": None,
                "has_talc_annotation": False,
                "magnification_from_name": parse_mag(filename),
                "sample_id": None,
                "md5": digest,
                "notes": [],
            }

            if "Панорамы" in parts[0]:
                record["image_type"] = "panorama"
            elif "ч1" in parts[0]:
                record["image_type"] = "labeled_ch1"
                if len(parts) > 1:
                    record["label_ore_sort"] = parts[1]
                record["has_talc_annotation"] = "Области оталькования" in rel
            elif "ч2" in parts[0]:
                record["image_type"] = "labeled_ch2"
                if len(parts) > 1:
                    record["label_intergrowth"] = parts[1]
            else:
                record["image_type"] = "other"

            sample_match = re.search(r"(\d{6,})", filename)
            if sample_match:
                record["sample_id"] = sample_match.group(1)
            elif filename.upper().startswith("DSCN"):
                record["sample_id"] = Path(filename).stem
            else:
                record["sample_id"] = Path(filename).stem

            if "аншлиф" in filename.lower():
                record["notes"].append("anshlif_in_name")

            info = get_image_info(path)
            record.update(info)

            mag = record["magnification_from_name"]
            width = record.get("width")
            if width and mag:
                record["estimated_fov_mm"] = round(estimate_fov_mm(width, mag), 2)
                record["estimated_um_per_px"] = round(
                    estimate_fov_mm(width, mag) * 1000 / width, 2
                )

            records.append(record)
            hash_map.setdefault(digest, []).append(rel)

    for record in records:
        record["duplicate_count"] = len(hash_map[record["md5"]])
        record["duplicate_paths"] = (
            hash_map[record["md5"]] if len(hash_map[record["md5"]]) > 1 else []
        )

    conflicts = []
    for digest, paths in hash_map.items():
        if len(paths) < 2:
            continue
        labels = set()
        for rel_path in paths:
            for record in records:
                if record["relative_path"] == rel_path:
                    labels.add(
                        (
                            record.get("label_ore_sort"),
                            record.get("label_intergrowth"),
                            record.get("has_talc_annotation"),
                        )
                    )
        if len(labels) > 1:
            conflicts.append({"md5": digest, "paths": paths, "labels": list(labels)})

    ch1_by_name: dict[str, list[dict]] = {}
    for record in records:
        if record["image_type"] == "labeled_ch1":
            ch1_by_name.setdefault(record["filename"], []).append(record)

    talc_pairs = []
    for filename, items in ch1_by_name.items():
        if len(items) == 2:
            talc_pairs.append(
                {
                    "filename": filename,
                    "original": next(
                        i["relative_path"]
                        for i in items
                        if not i["has_talc_annotation"]
                    ),
                    "annotated": next(
                        i["relative_path"] for i in items if i["has_talc_annotation"]
                    ),
                    "label_ore_sort": items[0]["label_ore_sort"],
                }
            )

    ch1_unique = {}
    for record in records:
        if record["image_type"] != "labeled_ch1" or record["has_talc_annotation"]:
            continue
        key = record["filename"]
        ch1_unique[key] = record

    ch2_index = {}
    for record in records:
        if record["image_type"] != "labeled_ch2":
            continue
        ch2_index.setdefault(record["filename"], []).append(record)

    cross_dataset_same_name = []
    for filename, ch1_rec in ch1_unique.items():
        if filename in ch2_index:
            cross_dataset_same_name.append(
                {
                    "filename": filename,
                    "ch1": {
                        "path": ch1_rec["relative_path"],
                        "label": ch1_rec["label_ore_sort"],
                    },
                    "ch2": [
                        {
                            "path": r["relative_path"],
                            "label": r["label_intergrowth"],
                        }
                        for r in ch2_index[filename]
                    ],
                }
            )

    with open(OUT / "catalog.json", "w", encoding="utf-8") as handle:
        json.dump(records, handle, ensure_ascii=False, indent=2)

    fields = [
        "relative_path",
        "filename",
        "image_type",
        "label_ore_sort",
        "label_intergrowth",
        "has_talc_annotation",
        "magnification_from_name",
        "estimated_fov_mm",
        "estimated_um_per_px",
        "width",
        "height",
        "sample_id",
        "md5",
        "duplicate_count",
        "notes",
    ]
    with open(OUT / "catalog.csv", "w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)

    duplicate_payload = {
        "exact_duplicate_groups": {
            digest: paths for digest, paths in hash_map.items() if len(paths) > 1
        },
        "cross_label_conflicts": conflicts,
        "ch1_talc_annotation_pairs": talc_pairs,
        "same_filename_ch1_ch2": cross_dataset_same_name,
    }
    with open(OUT / "duplicates.json", "w", encoding="utf-8") as handle:
        json.dump(duplicate_payload, handle, ensure_ascii=False, indent=2)

    summary = {
        "total_images": len(records),
        "by_type": dict(Counter(record["image_type"] for record in records)),
        "by_ore_sort_ch1": dict(
            Counter(
                record["label_ore_sort"]
                for record in records
                if record["label_ore_sort"]
            )
        ),
        "by_intergrowth_ch2": dict(
            Counter(
                record["label_intergrowth"]
                for record in records
                if record["label_intergrowth"]
            )
        ),
        "with_mag_in_name": sum(1 for record in records if record["magnification_from_name"]),
        "magnifications": dict(
            Counter(
                record["magnification_from_name"]
                for record in records
                if record["magnification_from_name"]
            )
        ),
        "exact_duplicate_groups": sum(1 for paths in hash_map.values() if len(paths) > 1),
        "cross_label_conflicts": len(conflicts),
        "ch1_talc_annotation_pairs": len(talc_pairs),
        "same_filename_ch1_ch2": len(cross_dataset_same_name),
        "panorama_sizes": [
            (record["filename"], record.get("width"), record.get("height"))
            for record in records
            if record["image_type"] == "panorama"
        ],
        "exif_populated_count": sum(
            1 for record in records if record.get("exif")
        ),
    }
    with open(OUT / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved to {OUT}")


if __name__ == "__main__":
    main()
