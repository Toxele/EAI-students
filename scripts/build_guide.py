"""Generate human-readable guides from catalog."""
import csv
import json
import shutil
from pathlib import Path

ROOT = Path(r"c:\Users\Professional\PycharmProjects\Nornickel")
ORG = ROOT / "data_organized"
DATA = ROOT / "data"
REF = ORG / "reference_from_postanovka"
REF.mkdir(exist_ok=True)

for img in (ROOT / "task" / "docx_images").glob("*"):
    if img.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        shutil.copy2(img, REF / img.name)

with open(ORG / "catalog.json", encoding="utf-8") as f:
    catalog = json.load(f)
with open(ORG / "duplicates.json", encoding="utf-8") as f:
    dups = json.load(f)

# Disputed / spornoe
rows = []
for item in dups["cross_label_conflicts"]:
    labels = []
    for path in item["paths"]:
        for rec in catalog:
            if rec["relative_path"] == path:
                label = rec.get("label_ore_sort") or rec.get("label_intergrowth")
                labels.append(f"{path} -> {label}")
    rows.append(
        {
            "md5": item["md5"],
            "paths": " | ".join(item["paths"]),
            "labels": " | ".join(labels),
            "why_disputed": "Same pixels, different folder label",
        }
    )

with open(ORG / "spornoe.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["md5", "paths", "labels", "why_disputed"])
    w.writeheader()
    w.writerows(rows)

# Talc annotation pairs index
with open(ORG / "talc_pairs.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["filename", "original", "annotated", "label_ore_sort"])
    w.writeheader()
    w.writerows(dups["ch1_talc_annotation_pairs"])

# By magnification cheat sheet
mag_rows = []
for rec in catalog:
    if rec["image_type"] != "labeled_ch1" or rec["has_talc_annotation"]:
        continue
    mag_rows.append(
        {
            "filename": rec["filename"],
            "label": rec["label_ore_sort"],
            "mag": rec.get("magnification_from_name"),
            "width": rec.get("width"),
            "height": rec.get("height"),
            "est_fov_mm": rec.get("estimated_fov_mm"),
            "est_um_per_px": rec.get("estimated_um_per_px"),
            "path": rec["relative_path"],
        }
    )

with open(ORG / "ch1_by_magnification.csv", "w", encoding="utf-8-sig", newline="") as f:
    w = csv.DictWriter(
        f,
        fieldnames=[
            "filename",
            "label",
            "mag",
            "width",
            "height",
            "est_fov_mm",
            "est_um_per_px",
            "path",
        ],
    )
    w.writeheader()
    w.writerows(sorted(mag_rows, key=lambda r: (r["label"] or "", r["mag"] or 0, r["filename"])))

print(f"spornoe: {len(rows)} rows")
print(f"talc_pairs: {len(dups['ch1_talc_annotation_pairs'])}")
print(f"ch1_by_magnification: {len(mag_rows)} rows")
