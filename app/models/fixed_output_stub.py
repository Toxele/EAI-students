"""
Временная заглушка вывода модели — до интеграции обученного классификатора.

Возвращает одни и те же зёрна и маску талька независимо от содержимого фото
(масштабированные под размер конкретного изображения), чтобы фронтенд и
пайплайн отчётов можно было доделывать, не дожидаясь весов реальной модели.

Классы и русские подписи берутся из configs/classifier/model_card.json —
контракта будущей модели (nornikel_ore_classifier_3class). Когда модель
будет обучена, эти классы заменяются реальным инференсом с тем же интерфейсом.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

from app.config import PROJECT_ROOT
from app.models.classifier_stub import ClassificationResult
from app.models.panorama_grain_detector import Grain

MODEL_CARD_PATH = PROJECT_ROOT / "configs" / "classifier" / "model_card.json"

SORT_CODE_BY_CLASS_NAME = {
    "ordinary": "ryadovaya",
    "thin": "trudnoobogatimaya",
    "talc": "otalkovannaya",
}

# Фиксированные зёрна: (rel_x, rel_y, rel_w, rel_h, intergrowth_type, gray_ratio)
FIXED_GRAINS_REL: tuple[tuple[float, float, float, float, str, float], ...] = (
    (0.10, 0.12, 0.18, 0.15, "ordinary", 0.15),
    (0.55, 0.08, 0.22, 0.18, "thin", 0.62),
    (0.30, 0.55, 0.20, 0.20, "ordinary", 0.20),
    (0.68, 0.50, 0.16, 0.22, "thin", 0.58),
    (0.42, 0.30, 0.10, 0.10, "ordinary", 0.10),
    (0.80, 0.75, 0.12, 0.12, "thin", 0.70),
)

# Фиксированный контур талька в относительных координатах (0..1) изображения
FIXED_TALC_POLYGON_REL: tuple[tuple[float, float], ...] = (
    (0.06, 0.60),
    (0.13, 0.46),
    (0.24, 0.40),
    (0.34, 0.46),
    (0.39, 0.58),
    (0.34, 0.74),
    (0.24, 0.82),
    (0.13, 0.78),
)
# Мелкие вкрапления талька: (rel_cx, rel_cy, rel_radius)
FIXED_TALC_BLOBS_REL: tuple[tuple[float, float, float], ...] = (
    (0.72, 0.20, 0.08),
    (0.85, 0.35, 0.05),
)


@dataclass
class ModelCard:
    """Метаданные будущей модели (classes/preprocessing/postprocessing)."""

    classes_by_name: dict[str, dict]

    @classmethod
    def load(cls, path: Path = MODEL_CARD_PATH) -> "ModelCard":
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(classes_by_name={c["name"]: c for c in raw["classes"]})


class FixedPanoramaGrainDetector:
    """Заглушка: одни и те же зёрна независимо от содержимого фото."""

    def detect(self, image_rgb: NDArray[np.uint8]) -> tuple[list[Grain], NDArray[np.uint8]]:
        h, w = image_rgb.shape[:2]
        sulfide_mask = np.zeros((h, w), dtype=np.uint8)
        grains: list[Grain] = []

        for grain_id, (rx, ry, rw, rh, itype, gray_ratio) in enumerate(FIXED_GRAINS_REL):
            x = int(rx * w)
            y = int(ry * h)
            gw = max(1, int(rw * w))
            gh = max(1, int(rh * h))
            sulfide_mask[y : y + gh, x : x + gw] = 255
            grains.append(
                Grain(
                    grain_id=grain_id,
                    bbox=(x, y, gw, gh),
                    area=gw * gh,
                    intergrowth_type=itype,
                    gray_ratio=gray_ratio,
                )
            )

        return grains, sulfide_mask


class FixedPanoramaTalcDetector:
    """Заглушка: одна и та же маска талька независимо от содержимого фото."""

    def predict(
        self, image_rgb: NDArray[np.uint8], sulfide_mask: NDArray[np.uint8] | None = None
    ) -> tuple[NDArray[np.uint8], float]:
        h, w = image_rgb.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        polygon = np.array(
            [[int(rx * w), int(ry * h)] for rx, ry in FIXED_TALC_POLYGON_REL], dtype=np.int32
        )
        cv2.fillPoly(mask, [polygon], 255)

        min_side = min(w, h)
        for rx, ry, rr in FIXED_TALC_BLOBS_REL:
            cv2.circle(mask, (int(rx * w), int(ry * h)), int(rr * min_side), 255, thickness=-1)

        total = w * h
        talc_percent = 100.0 * np.count_nonzero(mask) / max(total, 1)
        return mask, float(talc_percent)


class FixedOreClassifier:
    """Заглушка классификатора сорта руды — фиксированный класс из model_card.json."""

    FIXED_CLASS_NAME = "talc"

    def __init__(self, model_card: ModelCard | None = None) -> None:
        self.model_card = model_card or ModelCard.load()

    def predict(self, image_rgb: NDArray[np.uint8]) -> ClassificationResult:
        cls = self.model_card.classes_by_name[self.FIXED_CLASS_NAME]
        return ClassificationResult(
            sort_code=SORT_CODE_BY_CLASS_NAME[cls["name"]],
            sort_label_ru=cls["display_name"],
            best_distance=0,
            matched_reference="model_card_stub",
        )
