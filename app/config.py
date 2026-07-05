"""
Конфигурация путей и порогов проекта.

Все магические числа собраны здесь, чтобы не размазывать по коду.
"""
import os
from pathlib import Path

# Корень репозитория (папка Nornickel)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Папка с исходными фото для обучения / сравнения классификатора
DATA_DIR = PROJECT_ROOT / "data"

# ch1 — сорта руды (3 папки с метками в имени папки)
CH1_DATA_DIR = DATA_DIR / "Фото руд по сортам. ч1"

# Куда складывать загруженные пользователем файлы
UPLOAD_DIR = PROJECT_ROOT / "uploads"

# Куда складывать результаты анализа (overlay, json)
RESULTS_DIR = PROJECT_ROOT / "results"

# Пикселей больше этого — считаем панорамой
PANORAMA_PIXEL_THRESHOLD = 50_000_000

# Порог доли талька для сорта «оталькованная» (из постановки)
TALC_PERCENT_THRESHOLD = 10.0

# Максимальная сторона изображения при обработке stub (чтобы не умереть на 15k×10k)
MAX_PROCESS_SIDE = 4096

# Панорама режется на тайлы не больше этого размера и обрабатывается
# поштучно (см. app/pipeline/tiling.py) — модели обучены на масштабе
# отдельных кадров, глобальный downscale гигантской панорамы схлопывает
# тальк/зёрна до суб-пикселя.
PANORAMA_TILE_SIZE = 2500

# Контекстное поле вокруг каждого тайла (px, до обрезки по границам
# изображения) — без него модель на границе тайла не видит соседних
# пикселей, и предсказания смежных тайлов на стыке расходятся (видимые
# швы по сетке). В результат идёт только центральная область тайла.
PANORAMA_TILE_MARGIN = 256

# Порог яркости для сульфидов на панораме (перцентиль по grayscale)
BRIGHT_PERCENTILE = 88

# Минимальная площадь blob в пикселях (отсечь шум)
MIN_BLOB_AREA = 30

# Папка с весами моделей — не хранится в git (см. .gitignore: models/weights/*.pt).
# Разово подкачивается скриптом scripts/download_weights.py.
WEIGHTS_DIR = PROJECT_ROOT / "models" / "weights"

# Unet++ fast_768 — сегментация талька (Kaggle). Путь можно переопределить
# переменной окружения.
_DEFAULT_TALC_WEIGHTS = WEIGHTS_DIR / "segmentator.pt"
TALC_SEGMENTER_WEIGHTS = Path(os.environ.get("TALC_SEGMENTER_WEIGHTS", _DEFAULT_TALC_WEIGHTS))

# Coarse/fine (рядовая/труднообогатимая) бинарный классификатор — ResNet34
# (scripts/train_coarse_fine.py, kaggle/train_coarse_fine_binary.ipynb).
_DEFAULT_ORE_CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "classifier.pt"
ORE_CLASSIFIER_WEIGHTS = Path(
    os.environ.get("ORE_CLASSIFIER_WEIGHTS", _DEFAULT_ORE_CLASSIFIER_WEIGHTS)
)
