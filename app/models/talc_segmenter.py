"""
Сегментатор талька — Unet++ fast_768 (Kaggle).

Если весов нет или torch не установлен — пустая маска.
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from app.config import TALC_SEGMENTER_WEIGHTS
from app.models.segmentation_stub import SegmentationResult

logger = logging.getLogger(__name__)


class TalcSegmenter:
    """Inference Unet++ по models/weights/best_talk.pt."""

    def __init__(self, weights_path=None) -> None:
        self._model = None
        self._device = None
        self._img_size = (576, 768)
        self._ready = False
        self._weights = weights_path or TALC_SEGMENTER_WEIGHTS
        self._try_load()

    def _try_load(self) -> None:
        if not self._weights.exists():
            logger.warning("Talc segmenter weights not found: %s", self._weights)
            return
        try:
            import torch

            from models.talc_unetpp import load_talc_model

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
        """Маска талька + overlay."""
        height, width = image_rgb.shape[:2]
        empty = np.zeros((height, width), dtype=np.uint8)

        if not self._ready or self._model is None:
            return SegmentationResult(
                overlay_rgb=image_rgb.copy(),
                talc_mask=empty,
                talc_percent=0.0,
                sulfide_percent=0.0,
                matrix_percent=100.0,
            )

        from models.talc_unetpp import predict_talc_mask

        talc_mask, talc_percent = predict_talc_mask(
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
        )
