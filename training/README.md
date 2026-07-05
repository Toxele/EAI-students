# Training pipeline

Research/training code used to produce the model weights that `app/` loads
at inference time. This is not needed to run the product (see the root
[README.md](../README.md)) unless you want to retrain a model — with one
exception: `training/models/classifiers.py` and `training/models/talc_unetpp.py`
define the network architectures, and `app/models/ore_inclusion_classifier.py`
/ `app/models/talc_segmenter.py` import them directly to load the checkpoints
in `weights/`. Keep `training/models/` in place even in a deployment that
never trains.

Install: `pip install -r requirements.txt -r training/requirements.txt` from
the repo root.

## Layout

```text
training/
  configs/                 JSON training configs (classifier, segmentation)
  data/                    manifest, splits, datasets, annotations, weak masks
  hydra/                   lightweight JSON config loader (Hydra-style overrides)
  losses/                  loss functions
  models/                  classifier + Unet++ factories (also used by app/ at inference)
  trainers/                training loops
  loggers/                 MLflow helpers
  visualization/           dataset visualizations
  scripts/                 CLI entry points (see below)
  kaggle/                  notebooks meant to run on Kaggle (GPU)
  notebooks/               local experiment notebooks
  dataset/                 built dataset (manifest/classification/CVAT export) — gitignored except structure
```

Invoke scripts as modules from the repo root, e.g.:

```bash
python -m training.scripts.build_manifest
```

## Dataset build

Build the manifest and grouped train/val split from raw photos in
`training/data/`:

```bash
python -m training.scripts.build_manifest
```

Extract weak talc masks from blue contour images:

```bash
python -m training.scripts.extract_weak_masks
```

Save quick audit plots:

```bash
python -m training.scripts.visualize_dataset
```

## Classifier (3-class: ordinary / thin / talc)

Train:

```bash
python -m training.scripts.train_classifier
```

Evaluate (confusion matrix, ROC-AUC, error examples):

```bash
python -m training.scripts.evaluate_classifier --checkpoint artifacts/runs/classifier_baseline/best.pt --mlflow
```

Training writes `history.csv`, `metrics.json`, `per_class_metrics.csv`,
`confusion.csv`, and `best.pt` (selected by validation macro-F1).

Predict with probabilities and an `uncertain` decision layer:

```bash
python -m training.scripts.predict_classifier
```

`final_label=uncertain` is set when max probability is low, the top-1/top-2
margin is small, or the manifest row is a conflicting duplicate.

Grad-CAM activation overlays (weak explanations, not expert masks — use them
to inspect what the classifier relies on and to prioritize manual review):

```bash
python -m training.scripts.explain_classifier
```

Override JSON config values Hydra-style:

```bash
python -m training.scripts.train_classifier data.image_size=256 trainer.epochs=3 model.name=small_cnn
```

## Coarse/fine binary classifier (ordinary vs. thin — production model)

This is the model actually loaded by `app/models/ore_inclusion_classifier.py`
(`weights/classifier.pt`).

```bash
python -m training.scripts.build_coarse_fine_manifest
python -m training.scripts.train_coarse_fine
```

Or train on Kaggle GPU — see `training/kaggle/README.md` for the full
zip-build → upload → run → download-weights workflow
(`training/kaggle/train_coarse_fine_binary.ipynb`).

## Talc segmentation (production model)

This is the model loaded by `app/models/talc_segmenter.py`
(`weights/segmentator.pt`, Unet++ EfficientNet-B4, fast_768).

```bash
python -m training.scripts.import_cvat_annotations
python -m training.scripts.build_kaggle_segmentation
```

Train on Kaggle: `training/kaggle/train_talc_segmentation.ipynb`. Download
the resulting checkpoint to `weights/segmentator.pt`.

Before segmentation training, inspect color-domain shift between annotated
images and panoramas:

```bash
python -m training.scripts.eda_color_domains --output-dir artifacts/eda/color_domains
```

Generate a lightweight browser editor for manually closing missed talc
contours:

```bash
python -m training.scripts.make_mask_editor --prefer-overlay
python -m training.scripts.serve_mask_editor
```

Open `http://127.0.0.1:8765/artifacts/mask_editor/index.html`. Use `Brush` to
close broken contours and `Fill` to create a semi-transparent area mask inside
the selected contour. Blue overlay lines plus red manual lines are treated as
fill boundaries. The editor stores binary `*_manual_mask.png` masks; source
images are not modified. Click `Choose autosave folder` and select
`artifacts/manual_masks/talc_masks/`. The current mask autosaves when you move
to another image, and saved masks are loaded back when you return. Opening
through `localhost` matters for stable browser folder access. `Download mask`
is available as a fallback.

## Manual annotation

Create a CSV template for only conflicting duplicates:

```bash
python -m training.scripts.make_annotation_template --conflicts-only
```

Or generate a small browser-based manual labeling UI:

```bash
python -m training.scripts.make_annotation_ui --conflicts-only
```

Open `artifacts/annotation_ui/index.html`, label examples with buttons or keys
`1` = ordinary, `2` = thin, `3` = talc, `0` = skip, then export CSV to
`artifacts/annotations/manual_labels.csv`:

```csv
rel_path,label,comment,annotator
...,ordinary,looks like ordinary intergrowth,me
...,thin,rechecked with teammate,me
```

Allowed classifier labels: `ordinary`, `thin`, `talc`.

Apply manual labels to a new manifest:

```bash
python -m training.scripts.apply_annotations
python -m training.scripts.train_classifier manifest_csv=artifacts/manifests/nornikel_manifest_annotated.csv
```

## Notes

- The classifier baseline intentionally excludes `label_conflict=true` rows by
  default — exact duplicate images appear under different class folders in
  the original dataset.
- Weak talc masks are not ground truth. They are derived from blue line
  drawings and should be treated as noisy pseudo-labels.
- For PRs to `main`, do not add raw datasets, Kaggle notebook outputs, or
  `artifacts/` — only code, configs, and notebooks without large outputs.
