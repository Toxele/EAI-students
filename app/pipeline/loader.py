"""
Загрузка изображений с диска или bytes.

cv2.imread не работает с кириллицей в пути на Windows — читаем через np.fromfile + imdecode.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import MAX_PROCESS_SIDE


def imread_unicode(path: Path) -> NDArray[np.uint8] | None:
    """
    Читает BGR изображение с диска (работает с кириллицей в пути).

    :param path: путь к файлу
    :return: BGR uint8 или None
    """
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def load_image(path: Path) -> NDArray[np.uint8]:
    """Читает файл и возвращает RGB uint8."""
    bgr = imread_unicode(path)
    if bgr is None:
        raise ValueError(f"Не удалось прочитать изображение: {path}")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return maybe_downscale(rgb)


def load_image_from_bytes(data: bytes) -> NDArray[np.uint8]:
    """Читает изображение из bytes (upload)."""
    arr = np.frombuffer(data, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Не удалось декодировать изображение из bytes")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return maybe_downscale(rgb)


def maybe_downscale(image_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Уменьшает картинку, если длинная сторона > MAX_PROCESS_SIDE."""
    height, width = image_rgb.shape[:2]
    max_side = max(height, width)

    if max_side <= MAX_PROCESS_SIDE:
        return image_rgb

    scale = MAX_PROCESS_SIDE / max_side
    new_w = int(width * scale)
    new_h = int(height * scale)

    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
