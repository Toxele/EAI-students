# CVAT export

| Folder | Files |
|--------|-------|
| `to_annotate/` | 1403 numbered images (`000001.jpg` …) |
| `reference_blue/` | 43 same numbers where blue markup exists |
| `manifest.csv` | id ↔ original filename ↔ source path |

## How to use

1. Import **`to_annotate/`** into CVAT.
2. On a second screen open **`reference_blue/000042.jpg`** for the same **`to_annotate/000042.jpg`** (only 42 ch1 files have a pair).
3. After export from CVAT, merge labels back via **`manifest.csv`** (`id` / `original_filename` / `source_path`).

Numbering order: ch1 → ch2 → panorama tiles (2× zoom-out vs ch1 detail).

Panorama: each file → 16 tiles (4×4 grid), resized to 2272×1704. Order: pano 4, 5, … 17.

Rebuild: `py scripts/prepare_cvat_export.py`
