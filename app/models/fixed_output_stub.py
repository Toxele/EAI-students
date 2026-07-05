"""Placeholder grain/talc layout used as a fallback when no trained detector is available.

Returns the same fixed grains and talc mask regardless of image content
(scaled to the actual image size), so the pipeline and UI stay functional
before/without model weights.
"""
from __future__ import annotations

import cv2
import numpy as np
from numpy.typing import NDArray

from app.models.panorama_grain_detector import Grain

# Fixed grains: (rel_x, rel_y, rel_w, rel_h, intergrowth_type, gray_ratio)
FIXED_GRAINS_REL: tuple[tuple[float, float, float, float, str, float], ...] = (
    (0.10, 0.12, 0.18, 0.15, "ordinary", 0.15),
    (0.55, 0.08, 0.22, 0.18, "thin", 0.62),
    (0.30, 0.55, 0.20, 0.20, "ordinary", 0.20),
    (0.68, 0.50, 0.16, 0.22, "thin", 0.58),
    (0.42, 0.30, 0.10, 0.10, "ordinary", 0.10),
    (0.80, 0.75, 0.12, 0.12, "thin", 0.70),
)

# Fixed talc contour in relative image coordinates (0..1)
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
# Small talc blobs: (rel_cx, rel_cy, rel_radius)
FIXED_TALC_BLOBS_REL: tuple[tuple[float, float, float], ...] = (
    (0.72, 0.20, 0.08),
    (0.85, 0.35, 0.05),
)


class FixedPanoramaGrainDetector:
    """Fallback: the same grains regardless of image content."""

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
    """Fallback: the same talc mask regardless of image content."""

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
