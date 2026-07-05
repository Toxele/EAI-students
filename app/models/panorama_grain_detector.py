"""Grain (sulfide inclusion) domain model shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Grain:
    """A single detected grain (sulfide inclusion)."""

    grain_id: int
    bbox: tuple[int, int, int, int]
    area: int
    intergrowth_type: str
    gray_ratio: float
