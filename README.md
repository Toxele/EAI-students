# Nornickel Ore Microscopy Toolkit

Small OOP pipeline for the hackathon data:

- manifest and data audit with duplicate / label-conflict detection;
- weak talc mask extraction from blue contour markup;
- image classifier for `ordinary`, `thin`, `talc`;
- JSON configs with a lightweight Hydra-like override loader;
- visual diagnostics and manual-label CSV templates.

## Layout

```text
configs/                 JSON configs
data/                    manifest, splits, datasets, annotations, weak masks
hydra/                   lightweight JSON config loader
losses/                  loss functions
models/                  classifier factory
trainers/                training loops
visualization/           dataset visualizations
scripts/                 CLI entry points
artifacts/               generated outputs, ignored by git
```

## Commands

Build the manifest and grouped train/val split:

```bash
python -m scripts.build_manifest
```

Extract weak talc masks from blue contour images:

```bash
python -m scripts.extract_weak_masks
```

Save quick audit plots:

```bash
python -m scripts.visualize_dataset
```

Train the classifier:

```bash
python -m scripts.train_classifier
```

Training writes:

- `history.csv` with loss, accuracy, macro precision/recall/F1 per epoch;
- `metrics.json` with final validation metrics;
- `per_class_metrics.csv` with class-wise precision/recall/F1;
- `confusion.csv`;
- `best.pt` selected by validation macro-F1.

Predict with probabilities and an `uncertain` decision layer:

```bash
python -m scripts.predict_classifier
```

The prediction script marks `final_label=uncertain` when max probability is low,
top-1/top-2 margin is small, or the manifest row is a conflicting duplicate.

Generate Grad-CAM activation overlays:

```bash
python -m scripts.explain_classifier
```

These overlays are weak explanations, not expert masks. Use them to inspect what
the classifier relies on and to prioritize manual review.

## Talc Segmentation Pipeline

Extract blue-line weak masks:

```bash
python -m scripts.extract_weak_masks
```

Build a balanced-ish talc segmentation dataset with zero-mask negative examples:

```bash
python -m scripts.build_talc_dataset
```

By default this uses manual area masks from
`artifacts/manual_masks/talk_lines/` as positive talc masks and adds zero-mask
negative examples from ordinary/thin classification images. Weak masks are not
mixed in unless `include_weak_fallback=true` is passed.

Train the talc segmenter:

```bash
python -m scripts.train_talc_segmenter
```

Predict talc masks and estimated talc fractions:

```bash
python -m scripts.predict_talc_masks
```

Generate a lightweight browser editor for manually closing missed talc contours:

```bash
python -m scripts.make_mask_editor --prefer-overlay
python -m scripts.serve_mask_editor
```

Open `http://127.0.0.1:8765/artifacts/mask_editor/index.html`. Use `Brush` to
close broken contours and `Fill` to create a semi-transparent area mask inside
the selected contour. Blue overlay lines plus red manual lines are treated as
fill boundaries. The editor stores binary `*_manual_mask.png` masks; source
images are not modified. Click `Choose autosave folder` and select
`artifacts/manual_masks/talc_masks/`. After that, the current mask is saved
automatically when you move to another image, and saved masks are loaded back
when you return to an image. Opening through `localhost` is important for stable
browser folder access. `Download mask` remains available as a fallback.

Notebook with visual review helpers:

```text
notebooks/talc_pipeline_demo.ipynb
notebooks/color_domain_eda.ipynb
```

Before segmentation training, inspect color-domain shift between green marked
images, classification part 2, and panoramas:

```bash
python -m scripts.eda_color_domains --output-dir artifacts/eda/color_domains
```

The segmenter config includes moderate gray-domain augmentation to expose the
manual green talc masks to darker, lower-saturation styles.

Override JSON values in a Hydra-like style:

```bash
python -m scripts.train_classifier data.image_size=256 trainer.epochs=3 model.name=small_cnn
```

## Manual Annotation

Yes, you can label the dataset yourself without moving source files.

Create a CSV template for only conflicting duplicates:

```bash
python -m scripts.make_annotation_template --conflicts-only
```

Or generate a small browser-based manual labeling UI:

```bash
python -m scripts.make_annotation_ui --conflicts-only
```

Open `artifacts/annotation_ui/index.html`, label examples with buttons or keys
`1` = ordinary, `2` = thin, `3` = talc, `0` = skip, then export CSV and place it at
`artifacts/annotations/manual_labels.csv`.

Edit `artifacts/annotations/manual_labels.csv`:

```csv
rel_path,label,comment,annotator
...,ordinary,looks like ordinary intergrowth,me
...,thin,rechecked with teammate,me
```

Allowed classifier labels are currently:

- `ordinary`
- `thin`
- `talc`

The current trainer uses the generated manifest. Manual labels are stored separately so the raw archive remains untouched.
Apply manual labels to a new manifest:

```bash
python -m scripts.apply_annotations
python -m scripts.train_classifier manifest_csv=artifacts/manifests/nornikel_manifest_annotated.csv
```

## Notes

The classifier baseline intentionally excludes `label_conflict=true` rows by default. That is important because exact duplicate images appear under different class folders in the original dataset.

Weak talc masks are not ground truth. They are derived from blue line drawings and should be treated as noisy pseudo-labels.
