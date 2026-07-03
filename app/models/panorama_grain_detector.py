"""
STUB: Детектор зёрен (сульфидных включений) для панорамы.

Сейчас: порог яркости + connected components (OpenCV).
Потом: заменить на обученную модель детекции (YOLO / Mask R-CNN).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import BRIGHT_PERCENTILE, MIN_BLOB_AREA


@dataclass
class Grain:
    """Одно найденное зерно (сульфидное включение)."""

    grain_id: int
    bbox: tuple[int, int, int, int]
    area: int
    intergrowth_type: str
    gray_ratio: float


class PanoramaGrainDetector:
    """STUB-детектор ярких зёрен на панораме."""

    def __init__(self, bright_percentile: float = BRIGHT_PERCENTILE) -> None:
        self.bright_percentile = bright_percentile

    def detect(self, image_rgb: NDArray[np.uint8]) -> tuple[list[Grain], NDArray[np.uint8]]:
        """Находит зёрна на изображении. Возвращает список Grain и маску сульфидов."""
        gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)

        threshold_value = float(np.percentile(gray, self.bright_percentile))
        _, bright_mask = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        kernel = np.ones((3, 3), np.uint8)
        bright_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel)

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            bright_mask, connectivity=8
        )

        grains: list[Grain] = []
        grain_id = 0

        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area < MIN_BLOB_AREA:
                continue

            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            w = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            h = int(stats[label_idx, cv2.CC_STAT_HEIGHT])

            patch_gray = gray[y : y + h, x : x + w]
            patch_bright = bright_mask[y : y + h, x : x + w]

            intergrowth_type, gray_ratio = self._classify_intergrowth(patch_gray, patch_bright)

            grains.append(
                Grain(
                    grain_id=grain_id,
                    bbox=(x, y, w, h),
                    area=area,
                    intergrowth_type=intergrowth_type,
                    gray_ratio=gray_ratio,
                )
            )
            grain_id += 1

        return grains, bright_mask

    def _classify_intergrowth(
        self, patch_gray: NDArray[np.uint8], patch_bright: NDArray[np.uint8]
    ) -> tuple[str, float]:
        """STUB: рядовое vs тонкое по доле серого внутри bbox."""
        if patch_gray.size == 0:
            return "ordinary", 0.0

        values = patch_gray[patch_bright > 0]
        if values.size == 0:
            values = patch_gray.reshape(-1)

        p25, p75 = np.percentile(values, [25, 75])
        gray_pixels = np.sum((values >= p25) & (values <= p75))
        gray_ratio = float(gray_pixels / max(values.size, 1))

        if gray_ratio > 0.35:
            return "thin", gray_ratio
        return "ordinary", gray_ratio
