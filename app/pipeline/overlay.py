"""Overlay image rendering for the UI (grain type layer and talc layer)."""
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

# Overlay colors in RGB (used for the UI legend)
COLOR_ORDINARY_RGB = (0, 200, 0)   # green — ordinary intergrowth
COLOR_THIN_RGB = (220, 40, 40)     # red — thin intergrowth
COLOR_TALC_RGB = (40, 120, 255)    # blue — talc


def draw_talc_layer(
    image_rgb: NDArray[np.uint8],
    talc_mask: NDArray[np.uint8],
    alpha: float = 0.55,
) -> NDArray[np.uint8]:
    """Talc layer: semi-transparent blue mask over the image."""
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
    """Type layer: grain bboxes colored by status/intergrowth_type."""
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
            color_bgr = (180, 180, 0)  # yellow — undetermined
        cv2.rectangle(bgr, (x, y), (x + w, y + h), color_bgr, 2)

    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def save_overlay(overlay_rgb: NDArray[np.uint8], path: str) -> None:
    """Save an overlay to a file (format determined by extension)."""
    bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, bgr)
