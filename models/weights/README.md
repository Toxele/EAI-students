# Веса моделей (не в git)

Модели грузят веса отсюда же — `models/weights/` (см. `TALC_SEGMENTER_WEIGHTS` /
`ORE_CLASSIFIER_WEIGHTS` в `app/config.py`, путь переопределяется переменной
окружения):

```
models/weights/segmentator.pt   # Unet++ talc segmentation
models/weights/classifier.pt    # binary coarse/fine ResNet34 (F1≈0.95)
```

Разово скачать оба файла с Google Drive: `py scripts/download_weights.py`
(требуется `pip install gdown`, id файлов заданы в самом скрипте).

Обучение coarse/fine: `scripts/train_coarse_fine.py`, Kaggle: `kaggle/train_coarse_fine_*.ipynb`
