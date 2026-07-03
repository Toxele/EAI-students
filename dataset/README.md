# dataset — Nornickel ore microscopy

Clean copy of `data/` with English folder labels for ML.

## Layout

| Path | Contents |
|------|----------|
| `source/` | Full mirror of `data/` (all files, all paths) |
| `classification/` | One canonical image per MD5, sorted by compound folder |
| `talc_segmentation/` | *(removed)* — use `cvat/` for manual annotation |
| `cvat/` | **CVAT import:** `to_annotate/` + `reference_blue/` |
| `index/` | `paths.csv`, `manifest.csv`, `by_md5.json`, `talc_pairs.csv`, `summary.json` |

## Folder names (`classification/`)

Two axes joined with `__`:

- **Talc:** `talc_bearing` | `non_talc_bearing` | `talc_mixed`
- **Intergrowth:** `coarse` | `fine` | `coarse_and_fine` | `unknown_intergrowth`

Mapping: ch1 row ore ↔ ch2 intergrowth (`рядовые`→`coarse`, `тонкие`→`fine`).

## Rebuild

```bash
py scripts/build_dataset.py
py scripts/prepare_cvat_export.py
```

After rebuild you can remove `data/`.

## Notes

- `non_talc_bearing` = folder label, not proof of zero talc.
- Talc: annotate manually in CVAT from `cvat/to_annotate/`; blue hints in `cvat/reference_blue/` (42 ch1 files).
