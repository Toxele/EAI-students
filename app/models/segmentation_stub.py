"""
STUB: Сегментатор для детальных OM-снимков.

Сейчас: возвращает пустую маску талька и нулевые проценты.
Потом: U-Net с классами sulfide / talc / matrix / gray.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SegmentationResult:
    """Результат сегментации: маски и доли фаз."""

    overlay_rgb: NDArray[np.uint8]
    talc_mask: NDArray[np.uint8]  # бинарная маска талька (H×W)
    talc_percent: float
    sulfide_percent: float
    matrix_percent: float


class SegmentationStub:
    """STUB: passthrough — talc_mask пустая, проценты = заглушка."""

    def predict(self, image_rgb: NDArray[np.uint8]) -> SegmentationResult:
        height, width = image_rgb.shape[:2]
        empty_mask = np.zeros((height, width), dtype=np.uint8)

        return SegmentationResult(
            overlay_rgb=image_rgb.copy(),
            talc_mask=empty_mask,
            talc_percent=0.0,
            sulfide_percent=0.0,
            matrix_percent=100.0,
        )
