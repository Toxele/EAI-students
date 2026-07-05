"""Talc segmenter — Unet++ fast_768 trained on Kaggle.

Falls back to an empty mask when weights are missing or torch is unavailable.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from app.config import TALC_SEGMENTER_WEIGHTS
from app.models.segmentation_stub import SegmentationResult, empty_segmentation_result

logger = logging.getLogger(__name__)


class TalcSegmenter:
    """Unet++ talc segmentation inference, loaded from TALC_SEGMENTER_WEIGHTS."""

    def __init__(self, weights_path=None) -> None:
        self._model = None
        self._device = None
        self._img_size = (576, 768)
        self._ready = False
        self._weights = weights_path or TALC_SEGMENTER_WEIGHTS
        self._try_load()

    @property
    def ready(self) -> bool:
        """True once weights are loaded and the model is ready for inference."""
        return self._ready

    def _try_load(self) -> None:
        if not self._weights.exists():
            logger.warning("Talc segmenter weights not found: %s", self._weights)
            return
        try:
            import torch

            from training.models.talc_unetpp import load_talc_model

            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model, meta = load_talc_model(self._weights, self._device)
            self._img_size = tuple(meta["img_size"])
            self._ready = True
            logger.info(
                "Talc segmenter loaded (Dice=%s, %s)",
                meta.get("val_dice"),
                self._weights.name,
            )
        except ImportError as exc:
            logger.warning("Talc segmenter disabled: %s", exc)

    def predict(self, image_rgb: NDArray[np.uint8]) -> SegmentationResult:
        """Talc mask and overlay for the given image."""
        if not self._ready or self._model is None:
            return empty_segmentation_result(image_rgb)

        from training.models.talc_unetpp import predict_talc_mask

        talc_mask, talc_percent, talc_confidence = predict_talc_mask(
            self._model,
            image_rgb,
            img_size=self._img_size,
            device=self._device,
        )
        overlay = image_rgb.copy()
        px = talc_mask > 0
        overlay[px] = (overlay[px].astype(np.float32) * 0.55 + np.array([255, 80, 30]) * 0.45).astype(
            np.uint8
        )

        return SegmentationResult(
            overlay_rgb=overlay,
            talc_mask=talc_mask,
            talc_percent=talc_percent,
            sulfide_percent=0.0,
            matrix_percent=max(0.0, 100.0 - talc_percent),
            talc_confidence=talc_confidence,
        )
