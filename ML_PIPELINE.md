# ML Pipeline

Документ описывает ML/DL часть проекта: где лежит код, какие ноутбуки запускать, как обучать модели, где искать веса и как делать инференс.

## Структура Папок

```text
configs/
  classifier/
    nornikel_classifier_domain_aug_effnet_b0.json
    nornikel_classifier_domain_aug_resnet34.json
    model_card.json
  segmentation/
    talc_segmenter.json
    talc_segmenter_domain_aug.json
    spbgu_unet.json

data/
  datasets.py                  # classification dataset + domain augmentations
  segmentation_datasets.py     # talc segmentation dataset
  spbgu_segmentation.py        # SPbGU/AFM txt->bmp dataset and manifest builder
  talc_dataset_builder.py

models/
  classifiers.py               # resnet/efficientnet/convnext classifier factory
  segmentation.py              # TinyUNet + segmentation_models_pytorch U-Net factory

trainers/
  classification_trainer.py
  talc_segmentation_trainer.py
  spbgu_unet_trainer.py

visualization/
  talc_review.py
  spbgu_segmentation.py        # SPbGU mask/overlay exporter

scripts/
  build_manifest.py
  train_classifier.py
  predict_classifier.py
  build_talc_dataset.py
  train_talc_segmenter.py
  predict_talc_masks.py
  build_spbgu_segmentation_manifest.py
  train_spbgu_unet.py
  predict_spbgu_masks.py

notebooks/
  classifier_experiments/classifier_domain_aug.ipynb
  talc_domain_generalization/talc_domain_generalization.ipynb
  spbgu_unet_experiments/spbgu_unet_pipeline.ipynb
  color_domain_eda.ipynb
  mlflow_metrics_dashboard.ipynb
```

Локальные данные и outputs не коммитятся:

```text
dataset/
spbgu_data/
kaggle_notebooks/
artifacts/
notebooks/**/outputs/
```

## Установка

Сначала активируй conda env проекта. PyTorch под CUDA 12.6 ставь отдельно командой с официального сайта PyTorch под свою ОС. После этого:

```powershell
pip install -r requirements.txt
```

Для SPbGU U-Net нужен пакет:

```powershell
pip install segmentation-models-pytorch
```

Он уже добавлен в `requirements.txt`, но если окружение создано раньше, его надо доустановить.

## Основные Модели

### 1. Классификатор Руды На 3 Класса

Классы:

```text
ordinary  # рядовая
thin      # труднообогатимая
talc      # оталькованная
```

Основной кандидат для команды:

```text
notebooks/classifier_experiments/outputs/runs/effnet_b0_domain_aug/best.pt
```

Конфиг обучения:

```text
configs/classifier/nornikel_classifier_domain_aug_effnet_b0.json
```

Метрики рядом с весами:

```text
notebooks/classifier_experiments/outputs/runs/effnet_b0_domain_aug/best_metrics.json
notebooks/classifier_experiments/outputs/runs/effnet_b0_domain_aug/best_per_class_metrics.csv
notebooks/classifier_experiments/outputs/runs/effnet_b0_domain_aug/best_confusion.csv
```

5-fold CV веса лежат здесь:

```text
notebooks/classifier_experiments/outputs/runs/effnet_b0_domain_aug_5fold/fold_00/best.pt
...
notebooks/classifier_experiments/outputs/runs/effnet_b0_domain_aug_5fold/fold_04/best.pt
```

Fold-веса нужны для проверки устойчивости. Для сервера обычно передаём один основной `effnet_b0_domain_aug/best.pt`.

Обучение:

```powershell
python -m scripts.build_manifest
python -m scripts.train_classifier --config configs/classifier/nornikel_classifier_domain_aug_effnet_b0.json
```

Ноутбук:

```text
notebooks/classifier_experiments/classifier_domain_aug.ipynb
```

В ноутбуке есть обычное обучение и опциональная 5-fold ячейка. Для CV поставь:

```python
RUN_5FOLD = True
```

Инференс через скрипт:

```powershell
python -m scripts.predict_classifier --config configs/classifier/predict_classifier.json checkpoint=notebooks/classifier_experiments/outputs/runs/effnet_b0_domain_aug/best.pt
```

На выходе CSV с `raw_label`, `final_label`, `confidence`, `margin`, `prob_ordinary`, `prob_thin`, `prob_talc`. Политика `uncertain` срабатывает при низкой уверенности или маленьком разрыве между top-1/top-2.

### 2. Сегментатор Талька На OM

Задача: получить probability map талька, бинаризовать по порогу и посчитать долю талька.

Основные веса:

```text
artifacts/runs/talc_segmenter/best.pt
```

Конфиг обучения:

```text
configs/segmentation/talc_segmenter.json
```

Доменный эксперимент:

```text
configs/segmentation/talc_segmenter_domain_aug.json
notebooks/talc_domain_generalization/talc_domain_generalization.ipynb
```

Обучение:

```powershell
python -m scripts.build_talc_dataset
python -m scripts.train_talc_segmenter --config configs/segmentation/talc_segmenter.json
```

Инференс:

```powershell
python -m scripts.predict_talc_masks --config configs/segmentation/predict_talc_masks.json checkpoint=artifacts/runs/talc_segmenter/best.pt
```

Скрипт сохраняет:

```text
masks/
probability/
overlays/
talc_prediction_report.csv
```

Бизнес-правило:

```text
if talc_fraction > 0.10:
    final_class = "talc"
else:
    final_class = classifier ordinary/thin decision
```

### 3. SPbGU / АСМ U-Net

Эта ветка использует SPbGU данные как отдельную задачу сегментации АСМ:

```text
source: .txt NT-MDT ASCII height map
target: .bmp colored instance/region mask -> binary foreground mask
```

Важно: `.bmp` в `spbgu_data` выглядит как instance-разметка областей разными цветами. В текущем U-Net пайплайне она бинаризуется как `mask > 0`, то есть модель учит foreground/background, а не instance segmentation.

Конфиг:

```text
configs/segmentation/spbgu_unet.json
```

Ноутбук:

```text
notebooks/spbgu_unet_experiments/spbgu_unet_pipeline.ipynb
```

Сбор manifest:

```powershell
python -m scripts.build_spbgu_segmentation_manifest --config configs/segmentation/spbgu_unet.json
```

Holdout обучение:

```powershell
python -m scripts.train_spbgu_unet --config configs/segmentation/spbgu_unet.json
```

K-fold CV:

```powershell
python -m scripts.train_spbgu_unet --config configs/segmentation/spbgu_unet.json --cv
```

Ожидаемый основной путь весов после обучения:

```text
notebooks/spbgu_unet_experiments/outputs/runs/spbgu_unet_resnet34_txt/best.pt
```

CV веса:

```text
notebooks/spbgu_unet_experiments/outputs/runs/spbgu_unet_resnet34_txt_4fold/fold_00/best.pt
...
notebooks/spbgu_unet_experiments/outputs/runs/spbgu_unet_resnet34_txt_4fold/fold_03/best.pt
```

Экспорт масок и overlay:

```powershell
python -m scripts.predict_spbgu_masks --config configs/segmentation/spbgu_unet.json
```

Можно явно указать checkpoint:

```powershell
python -m scripts.predict_spbgu_masks --config configs/segmentation/spbgu_unet.json --checkpoint notebooks/spbgu_unet_experiments/outputs/runs/spbgu_unet_resnet34_txt/best.pt
```

Выход:

```text
notebooks/spbgu_unet_experiments/outputs/predictions/spbgu_unet_resnet34_txt/
  predicted_masks/
  overlays/
  gt_prediction_overlay/
  prediction_report.csv
```

## Инференс На Сервере

### Классификатор

Минимальная логика:

```python
import torch
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

from models.classifiers import ClassifierFactory

checkpoint = torch.load("notebooks/classifier_experiments/outputs/runs/effnet_b0_domain_aug/best.pt", map_location="cpu")
cfg = checkpoint["config"]
classes = checkpoint.get("classes", cfg["classes"])
model = ClassifierFactory.create(cfg["model"], len(classes))
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

image = ImageOps.exif_transpose(Image.open("image.jpg")).convert("RGB")
image = TF.resize(image, [cfg["data"]["image_size"], cfg["data"]["image_size"]])
x = TF.to_tensor(image)
x = TF.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

with torch.no_grad():
    probs = torch.softmax(model(x.unsqueeze(0)), dim=1)[0]

result = {cls: float(prob) for cls, prob in zip(classes, probs)}
```

### Binary Segmentation

Для `talc_segmenter` и `spbgu_unet` схема одинаковая:

```python
import cv2
import torch
import numpy as np
from PIL import Image, ImageOps
import torchvision.transforms.functional as TF

from models.segmentation import SegmentationFactory

checkpoint = torch.load("PATH_TO_BEST_PT", map_location="cpu")
cfg = checkpoint["config"]
model = SegmentationFactory.create(cfg["model"])
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

image = ImageOps.exif_transpose(Image.open("image.jpg")).convert("RGB")
original_size = image.size
image_size = cfg["data"]["image_size"]
x = TF.resize(image, [image_size, image_size])
x = TF.to_tensor(x)
x = TF.normalize(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

with torch.no_grad():
    prob_small = torch.sigmoid(model(x.unsqueeze(0)))[0, 0].numpy()

prob = cv2.resize(prob_small, original_size, interpolation=cv2.INTER_LINEAR)
mask = (prob >= 0.5).astype(np.uint8) * 255
fraction = float((mask > 0).mean())
```

Для SPbGU `.txt` используй готовый helper:

```python
from visualization.spbgu_segmentation import SpbguMaskPredictor

predictor = SpbguMaskPredictor("notebooks/spbgu_unet_experiments/outputs/runs/spbgu_unet_resnet34_txt/best.pt")
prob, source_image = predictor.predict_image("spbgu_data/.../9A003_1.txt", input_kind="txt")
```

## Что Давать Команде

Для текущего демо/сервера:

```text
1. Классификатор:
   notebooks/classifier_experiments/outputs/runs/effnet_b0_domain_aug/best.pt

2. Тальк-сегментатор:
   artifacts/runs/talc_segmenter/best.pt

3. SPbGU U-Net после обучения:
   notebooks/spbgu_unet_experiments/outputs/runs/spbgu_unet_resnet34_txt/best.pt
```

К каждому весу желательно прикладывать:

```text
resolved_config.json
best_metrics.json
history.csv
```

Для PR в `main` не добавляй датасеты, Kaggle notebooks и outputs. Коммитить стоит только код, конфиги, ноутбуки без больших outputs и этот документ.
