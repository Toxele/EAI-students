"""
Конфигурация путей и порогов проекта.

Все магические числа собраны здесь, чтобы не размазывать по коду.
"""
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

# Порог яркости для сульфидов на панораме (перцентиль по grayscale)
BRIGHT_PERCENTILE = 88

# Минимальная площадь blob в пикселях (отсечь шум)
MIN_BLOB_AREA = 30

# Unet++ fast_768 — сегментация талька (Kaggle)
TALC_SEGMENTER_WEIGHTS = PROJECT_ROOT / "models" / "weights" / "best_talk.pt"
