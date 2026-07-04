# CVAT export — manual talc annotation

| Folder | Purpose |
|--------|---------|
| `to_annotate/` | Clean images for CVAT import (2020 files) |
| `reference_blue/` | Same filenames with expert blue strokes (43 files) |

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
