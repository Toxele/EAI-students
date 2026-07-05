"""
Computing metrics over the grain list, accounting for user corrections.

k — total inclusions (excluding false_positive)
l — ordinary, j — thin
"""
from __future__ import annotations

from typing import Any

# Grain statuses
STATUS_ORDINARY = "ordinary"
STATUS_THIN = "thin"
STATUS_UNCERTAIN = "uncertain"
STATUS_FALSE_POSITIVE = "false_positive"


def grain_confidence(gray_ratio: float) -> tuple[float, float]:
    """
    STUB: ordinary/thin confidence from the detector's gray_ratio.

    Replace later with the classification model's softmax.
    """
    conf_thin = min(1.0, max(0.0, gray_ratio / 0.5))
    conf_ordinary = 1.0 - conf_thin
    return round(conf_ordinary, 3), round(conf_thin, 3)


def compute_grain_counts(grains: list[dict[str, Any]]) -> dict[str, int]:
    """Computes k, l, j, and others over the current grain list."""
    active = [g for g in grains if g.get("status", STATUS_ORDINARY) != STATUS_FALSE_POSITIVE]
    ordinary = [g for g in active if g.get("status") == STATUS_ORDINARY or g.get("intergrowth_type") == STATUS_ORDINARY]
    thin = [g for g in active if g.get("status") == STATUS_THIN or g.get("intergrowth_type") == STATUS_THIN]
    uncertain = [g for g in active if g.get("status") == STATUS_UNCERTAIN]

    # If status isn't set, fall back to intergrowth_type
    def is_ordinary(g: dict) -> bool:
        s = g.get("status") or g.get("intergrowth_type", STATUS_ORDINARY)
        return s == STATUS_ORDINARY

    def is_thin(g: dict) -> bool:
        s = g.get("status") or g.get("intergrowth_type", STATUS_ORDINARY)
        return s == STATUS_THIN

    ordinary = [g for g in active if is_ordinary(g)]
    thin = [g for g in active if is_thin(g)]
    uncertain = [g for g in active if (g.get("status") or g.get("intergrowth_type")) == STATUS_UNCERTAIN]

    return {
        "total_k": len(active),
        "ordinary_l": len(ordinary),
        "thin_j": len(thin),
        "uncertain": len(uncertain),
        "false_positive": len([g for g in grains if g.get("status") == STATUS_FALSE_POSITIVE]),
    }


def compute_intergrowth_percent(grains: list[dict[str, Any]]) -> tuple[float, float]:
    """Ordinary/thin area shares among active grains."""
    active = [g for g in grains if g.get("status", STATUS_ORDINARY) != STATUS_FALSE_POSITIVE]
    if not active:
        return 50.0, 50.0

    def area_for_type(target: str) -> int:
        total = 0
        for g in active:
            s = g.get("status") or g.get("intergrowth_type", STATUS_ORDINARY)
            if s == target:
                total += int(g.get("area", 0))
        return total

    ordinary_area = area_for_type(STATUS_ORDINARY)
    thin_area = area_for_type(STATUS_THIN)
    uncertain_area = area_for_type(STATUS_UNCERTAIN)
    # Split uncertain ones evenly for the rule engine
    half_unc = uncertain_area / 2
    ordinary_area += int(half_unc)
    thin_area += int(uncertain_area - half_unc)

    total_area = ordinary_area + thin_area
    if total_area == 0:
        return 50.0, 50.0

    ordinary_pct = 100.0 * ordinary_area / total_area
    thin_pct = 100.0 * thin_area / total_area
    return round(ordinary_pct, 2), round(thin_pct, 2)


def enrich_grain(grain: dict[str, Any]) -> dict[str, Any]:
    """Adds default status and confidence."""
    item = dict(grain)
    conf_o, conf_t = grain_confidence(float(item.get("gray_ratio", 0)))
    item.setdefault("status", item.get("intergrowth_type", STATUS_ORDINARY))
    item.setdefault("conf_ordinary", conf_o)
    item.setdefault("conf_thin", conf_t)
    return item


def sulfide_area_percent(grains: list[dict[str, Any]], total_pixels: int) -> float:
    """Sulfide area share of the whole frame."""
    active = [g for g in grains if g.get("status") != STATUS_FALSE_POSITIVE]
    area = sum(int(g.get("area", 0)) for g in active)
    return round(100.0 * area / max(total_pixels, 1), 2)
