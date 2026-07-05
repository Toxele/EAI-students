"""
Detecting the mode: panorama or detail OM shot.

Based on the original image size (metadata can be read before downscaling).
"""
from __future__ import annotations

from app.config import PANORAMA_PIXEL_THRESHOLD


def detect_mode(width: int, height: int) -> str:
    """
    Returns 'panorama' or 'detail'.

    :param width: width in pixels
    :param height: height in pixels
    """
    total_pixels = width * height

    if total_pixels >= PANORAMA_PIXEL_THRESHOLD:
        return "panorama"
    return "detail"
