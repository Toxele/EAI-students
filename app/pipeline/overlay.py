"""
Сборка overlay-картинки для UI (цветные bbox зёрен и маска талька).
"""
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.models.panorama_grain_detector import Grain

# Цвета overlay в RGB (для легенды UI)
COLOR_ORDINARY_RGB = (0, 200, 0)   # зелёный — рядовое срастание
COLOR_THIN_RGB = (220, 40, 40)     # красный — тонкое срастание
COLOR_TALC_RGB = (40, 120, 255)    # синий — тальк


def draw_grain_overlay(
    image_rgb: NDArray[np.uint8],
    grains: list[Grain],
    talc_mask: NDArray[np.uint8] | None = None,
    alpha: float = 0.45,
) -> NDArray[np.uint8]:
    """
    Рисует полупрозрачные bbox зёрен поверх изображения.

    Зелёный = ordinary (рядовое), красный = thin (тонкое).
    Опционально накладывает синюю маску талька.
    """
    base = image_rgb.copy().astype(np.float32)

    # Синяя маска талька (если есть)
    if talc_mask is not None and talc_mask.any():
        talc_layer = np.zeros_like(base)
        talc_layer[talc_mask > 0] = COLOR_TALC_RGB
        mask_bool = talc_mask > 0
        base[mask_bool] = base[mask_bool] * (1 - alpha) + talc_layer[mask_bool] * alpha

    result = base.astype(np.uint8)
    bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    for grain in grains:
        x, y, w, h = grain.bbox
        if grain.intergrowth_type == "ordinary":
            color_bgr = (COLOR_ORDINARY_RGB[2], COLOR_ORDINARY_RGB[1], COLOR_ORDINARY_RGB[0])
        else:
            color_bgr = (COLOR_THIN_RGB[2], COLOR_THIN_RGB[1], COLOR_THIN_RGB[0])
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color_bgr, 2)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def draw_talc_layer(
    image_rgb: NDArray[np.uint8],
    talc_mask: NDArray[np.uint8],
    alpha: float = 0.55,
) -> NDArray[np.uint8]:
    """Слой «Тальк»: синяя полупрозрачная маска на фоне изображения."""
    base = image_rgb.copy().astype(np.float32)
    talc_layer = np.zeros_like(base)
    mask_bool = talc_mask > 0
    talc_layer[mask_bool] = COLOR_TALC_RGB
    base[mask_bool] = base[mask_bool] * (1 - alpha) + talc_layer[mask_bool] * alpha
    return base.astype(np.uint8)


def draw_type_layer(
    image_rgb: NDArray[np.uint8],
    grains: list[dict],
) -> NDArray[np.uint8]:
    """Слой «Тип»: bbox зёрен по status/intergrowth_type."""
    result = image_rgb.copy()
    bgr = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)

    for grain in grains:
        if grain.get("status") == "false_positive":
            continue
        x, y, w, h = grain["bbox"]
        itype = grain.get("status") or grain.get("intergrowth_type", "ordinary")
        if itype == "ordinary":
            color_bgr = (COLOR_ORDINARY_RGB[2], COLOR_ORDINARY_RGB[1], COLOR_ORDINARY_RGB[0])
        elif itype == "thin":
            color_bgr = (COLOR_THIN_RGB[2], COLOR_THIN_RGB[1], COLOR_THIN_RGB[0])
        else:
            color_bgr = (180, 180, 0)  # жёлтый — неопределённый
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color_bgr, 2)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_overlay(overlay_rgb: NDArray[np.uint8], path: str) -> None:
    """Сохраняет overlay в файл (JPEG/PNG по расширению)."""
    bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
