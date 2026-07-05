"""Shared segmentation result type and empty-mask fallback."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class SegmentationResult:
    """Segmentation output: masks and phase percentages."""

    overlay_rgb: NDArray[np.uint8]
    talc_mask: NDArray[np.uint8]  # binary talc mask (H x W)
    talc_percent: float
    sulfide_percent: float
    matrix_percent: float
    # Per-pixel confidence map (H x W, 0..255); higher means more confident.
    talc_confidence: NDArray[np.uint8]


def empty_segmentation_result(image_rgb: NDArray[np.uint8]) -> SegmentationResult:
    """Fallback result used when no trained segmenter is available."""
    height, width = image_rgb.shape[:2]
    empty_mask = np.zeros((height, width), dtype=np.uint8)

    return SegmentationResult(
        overlay_rgb=image_rgb.copy(),
        talc_mask=empty_mask,
        talc_percent=0.0,
        sulfide_percent=0.0,
        matrix_percent=100.0,
        talc_confidence=empty_mask,
    )
