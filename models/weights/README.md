# Веса моделей (не в git — положить вручную)

```
best_talk.pt                 # Unet++ talc segmentation
best_coarse_fine_binary.pt   # binary coarse/fine ResNet34 (F1≈0.95)
```

Обучение coarse/fine: `scripts/train_coarse_fine.py`, Kaggle: `kaggle/train_coarse_fine_*.ipynb`
