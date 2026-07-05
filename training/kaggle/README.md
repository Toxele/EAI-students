# Kaggle — Nornickel ML

## Coarse vs Fine (~45–60 min on a T4 GPU)

Classifier for **ordinary / thin** intergrowth. Talc is handled by
segmentation, not by these models.

**One data zip** — two **different** notebooks (do not mix them):

| File | Approach | Output |
|------|--------|-------|
| `train_coarse_fine_multilabel.ipynb` | Multi-label BCE (coarse + fine), ambiguous → [0.6, 0.6] | `best_coarse_fine_multilabel.pt` |
| `train_coarse_fine_binary.ipynb` | Soft binary (1 logit), ambiguous → 0.5 | `best_coarse_fine_binary.pt` (production model) |
| `../dataset/kaggle/nornickel_coarse_fine_classification.zip` | ~995 photos + manifest.csv | shared dataset |

Each notebook grids **resnet18 / resnet34 / efficientnet_b0** with cosine LR
and early stopping, ~45–60 min total.

### 1. Build the data zip (locally)

```bash
python -m training.scripts.build_coarse_fine_manifest
python -m training.scripts.build_kaggle_coarse_fine
python -m training.scripts.verify_kaggle_coarse_fine_zip
```

Manifest rules:
- only detail frames with **exactly one** ig-tag (coarse **or** fine) go into the zip;
- frames with talc + coarse/fine **are included**;
- the train/val split is stratified by coarse vs. fine (not by ambiguous).

### 2. Kaggle

1. Upload **`nornickel_coarse_fine_classification.zip`** as a Dataset (once)
2. Import the notebook you need (**multilabel** or **binary** — separately!)
3. Add Data → dataset
4. **Settings → Accelerator → GPU T4**
5. **Run All**
6. Download `best_coarse_fine_multilabel.pt` or `best_coarse_fine_binary.pt` → rename to `weights/classifier.pt`

Compare runs via `ig_macro_f1` (multilabel) vs. `clean_f1` / `composite_score`
(binary) in the grid summary.

---

## Talc segmentation (fast_768)

| File | Purpose |
|------|------------|
| `train_talc_segmentation.ipynb` | Training → checkpoint |
| `../dataset/kaggle/nornickel_talc_segmentation.zip` | Dataset |

```bash
python -m training.scripts.import_cvat_annotations
python -m training.scripts.build_kaggle_segmentation
```

Download the resulting checkpoint to `weights/segmentator.pt`.
