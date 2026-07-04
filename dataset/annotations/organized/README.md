# CVAT annotations (organized)

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

- **job_4195731**: 12 images, 39 polygons
- **task_2398159**: 51 images, 224 polygons

## Totals

- Annotated images: 63
- Polygon rows in CSV: 263

## Mapping

Join with `../../cvat/manifest.csv` on `cvat_filename`.

Regenerate: `py scripts/import_cvat_annotations.py`
