# Kaggle — Nornickel ML

## Coarse vs Fine (~45–60 мин на GPU T4)

Классификатор **рядовая / труднообогатимая**. Тальк — через сегментацию, не через эти модели.

**Один data zip** — два **разных** notebook (не смешивать):

| Файл | Подход | Выход |
|------|--------|-------|
| `train_coarse_fine_multilabel.ipynb` | Multi-label BCE (coarse + fine), ambiguous → [0.6, 0.6] | `best_coarse_fine_multilabel.pt` |
| `train_coarse_fine_binary.ipynb` | Soft binary (1 logit), ambiguous → 0.5 | `best_coarse_fine_binary.pt` |
| `../dataset/kaggle/nornickel_coarse_fine_classification.zip` | ~995 фото + manifest.csv | общий датасет |

Grid в каждом notebook: **resnet18 / resnet34 / efficientnet_b0**, cosine LR, early stop ~45–60 мин.

### 1. Сборка data zip (локально)

```bash
py scripts/build_coarse_fine_manifest.py
py scripts/build_kaggle_coarse_fine.py
py scripts/verify_kaggle_coarse_fine_zip.py
```

Правила manifest:
- только detail-кадры с **одним** ig-тегом (coarse **или** fine) в zip;
- кадры с talc + coarse/fine **включены**;
- train/val split стратифицирован по coarse vs fine (не по ambiguous).

### 2. Kaggle

1. Upload **`nornickel_coarse_fine_classification.zip`** as Dataset (один раз)
2. Import нужный notebook (**multilabel** или **binary** — отдельно!)
3. Add Data → dataset
4. **Settings → Accelerator → GPU T4**
5. **Run All**
6. Скачай `best_coarse_fine_multilabel.pt` или `best_coarse_fine_binary.pt` → `models/weights/`

Сравнение: смотри `ig_macro_f1` (multilabel) vs `clean_f1` / `composite_score` (binary) в grid summary.

---

## Сегментация талька (fast_768)

| Файл | Назначение |
|------|------------|
| `train_talc_segmentation.ipynb` | Обучение → `best_talk.pt` |
| `../dataset/kaggle/nornickel_talc_segmentation.zip` | Dataset |

```bash
py scripts/import_cvat_annotations.py
py scripts/build_kaggle_segmentation.py
```
