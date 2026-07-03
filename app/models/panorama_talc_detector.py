"""
STUB: Детектор талька для панорамы / низкого разрешения.

Логика: тёмные области (не яркие сульфиды) → кандидаты в тальк.
Позже заменить на обученную модель сегментации.
"""
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray


class PanoramaTalcDetector:
    """Тальк = тёмная матрица минус яркие сульфидные зёрна."""

    def __init__(self, dark_percentile: float = 35.0) -> None:
        # Пиксели темнее этого перцентиля считаем «тёмными»
        self.dark_percentile = dark_percentile

    def predict(
        self, image_rgb: NDArray[np.uint8], sulfide_mask: NDArray[np.uint8] | None = None
    ) -> tuple[NDArray[np.uint8], float]:
        """
        Возвращает бинарную маску талька (0/255) и долю % от площади кадра.

        :param image_rgb: RGB изображение
        :param sulfide_mask: маска сульфидов — исключаем из талька
        """
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
        threshold = float(np.percentile(gray, self.dark_percentile))
        _, dark_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

        if sulfide_mask is not None:
            dark_mask[sulfide_mask > 0] = 0

        # Убираем мелкий шум
        kernel = np.ones((3, 3), np.uint8)
        dark_mask = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)

        total = gray.shape[0] * gray.shape[1]
        talc_percent = 100.0 * np.count_nonzero(dark_mask) / max(total, 1)
        return dark_mask, float(talc_percent)
