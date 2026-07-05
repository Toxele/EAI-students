# Nornickel talc segmentation (Kaggle)

Binary segmentation: **talc** vs background on OM microscopy images.

| Split | Images |
|-------|--------|
| train | 51 |
| val   | 12 |
| total | 63 |

## Layout

```
images/{sample_id}.jpg   # 2272×1704 RGB
masks/{sample_id}.png    # 0=bg, 255=talc
metadata.csv
splits/train.txt
splits/val.txt
```

## Kinds

- `ch1_detail` — detailed OM (CVAT task export)
- `panorama_tile` — 4×4 tile from panorama 4.jpg

Upload as Kaggle Dataset, then open `kaggle/talc_segmentation_train.ipynb`.
