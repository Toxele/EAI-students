# Веса моделей

Файлы `.pt` не в git (кроме `best_talk.pt`). Остальные веса — отдельно.

```
best_talk.pt                 # Unet++ talc segmentation (в репозитории)
best_coarse_fine_binary.pt   # binary coarse/fine ResNet34 — положить сюда вручную
```

Обучение: `scripts/train_coarse_fine.py`, Kaggle: `kaggle/train_coarse_fine_*.ipynb`
