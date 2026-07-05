# Model weights (not in git)

Models load weights from this directory (see `TALC_SEGMENTER_WEIGHTS` /
`ORE_CLASSIFIER_WEIGHTS` in `app/config.py`; both paths are overridable via
environment variables):

```
weights/segmentator.pt   # Unet++ talc segmentation
weights/classifier.pt    # binary coarse/fine ResNet34 (F1≈0.95)
```

Download both files from Google Drive once: `py scripts/download_weights.py`
(requires `pip install gdown`; file ids are set in the script itself).

Coarse/fine training: `training/scripts/train_coarse_fine.py`,
Kaggle: `training/kaggle/train_coarse_fine_*.ipynb`.
