"""
Определение режима: панорама или детальный OM-снимок.

По размеру исходного изображения (до downscale можно смотреть meta).
"""
from __future__ import annotations

from app.config import PANORAMA_PIXEL_THRESHOLD


def detect_mode(width: int, height: int) -> str:
    """
    Возвращает 'panorama' или 'detail'.

    :param width: ширина в пикселях
    :param height: высота в пикселях
    """
    total_pixels = width * height

    if total_pixels >= PANORAMA_PIXEL_THRESHOLD:
        return "panorama"
    return "detail"
