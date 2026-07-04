# Kaggle — сегментация талька (fast_768)

| Файл | Назначение |
|------|------------|
| `train_talc_segmentation.ipynb` | Обучение с нуля → `best_talk.pt` |
| `../dataset/kaggle/nornickel_talc_segmentation.zip` | Dataset для Kaggle |
| `../models/weights/best_talk.pt` | Веса после обучения (~84 MB) |
| `../models/talc_unetpp.py` | Загрузка + inference |

## Kaggle

1. Upload zip как Dataset
2. Import notebook, GPU T4
3. Run All → скачай `best_talk.pt` в `models/weights/`

## Локально

```bash
py scripts/import_cvat_annotations.py
py scripts/build_kaggle_segmentation.py
```
