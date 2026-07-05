"""Coarse/fine intergrowth classifier — ordinary vs thin.

Binary ResNet (1 logit, sigmoid), trained by
training/scripts/train_coarse_fine.py / kaggle/train_coarse_fine_binary.ipynb
(see training/trainers/coarse_fine_trainer.py — target_coarse=1 for ordinary, 0 for thin).

When weights are missing or torch is unavailable, .ready stays False and the
caller falls back to its own heuristic instead of calling predict().
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import ORE_CLASSIFIER_WEIGHTS

logger = logging.getLogger(__name__)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass
class OreClassificationResult:
    """Classifier decision for a single inclusion/image."""

    intergrowth_type: str  # "ordinary" | "thin"
    prob_ordinary: float


class OreInclusionClassifier:
    """Binary coarse/fine classifier inference, loaded from ORE_CLASSIFIER_WEIGHTS."""

    def __init__(self, weights_path=None) -> None:
        self._model = None
        self._device = None
        self._image_size = 384
        self._ready = False
        self._weights = weights_path or ORE_CLASSIFIER_WEIGHTS
        self._try_load()

    @property
    def ready(self) -> bool:
        """True once weights are loaded and the model is ready for inference."""
        return self._ready

    def _try_load(self) -> None:
        if not self._weights.exists():
            logger.warning("Ore classifier weights not found: %s", self._weights)
            return
        try:
            import torch

            from training.models.classifiers import ClassifierFactory

            checkpoint = torch.load(self._weights, map_location="cpu", weights_only=False)
            cfg = checkpoint.get("config", {})
            self._image_size = cfg.get("data", {}).get("image_size", 384)
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._model = ClassifierFactory.create(cfg.get("model", {"name": "resnet34"}), num_classes=1)
            self._model.load_state_dict(checkpoint["model_state_dict"])
            self._model.to(self._device)
            self._model.eval()
            self._ready = True
            logger.info("Ore classifier loaded (%s)", self._weights.name)
        except ImportError as exc:
            logger.warning("Ore classifier disabled: %s", exc)

    def predict(self, image_rgb: NDArray[np.uint8]) -> OreClassificationResult:
        """Classify a patch/image as an ordinary or thin intergrowth."""
        if not self._ready or self._model is None:
            raise RuntimeError("Ore classifier is not ready — check .ready before calling predict()")

        import torch

        resized = cv2.resize(
            image_rgb, (self._image_size, self._image_size), interpolation=cv2.INTER_AREA
        )
        x = resized.astype(np.float32) / 255.0
        x = (x - IMAGENET_MEAN) / IMAGENET_STD
        tensor = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(self._device)

        with torch.no_grad():
            prob_ordinary = float(torch.sigmoid(self._model(tensor)).item())

        intergrowth_type = "ordinary" if prob_ordinary >= 0.5 else "thin"
        return OreClassificationResult(intergrowth_type=intergrowth_type, prob_ordinary=prob_ordinary)
