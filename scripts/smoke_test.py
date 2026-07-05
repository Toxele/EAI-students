"""Quick end-to-end smoke test for the analysis pipeline.

Generates synthetic detail/panorama images in memory so it does not depend
on any local dataset. Verifies the pipeline runs without crashing; it does
not check classification accuracy.

Run: py scripts/smoke_test.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.stdout.reconfigure(encoding="utf-8")  # sort_label_ru is Cyrillic; default Windows console codepage can't print it

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.pipeline.analyzer import Analyzer


def synthetic_image(width: int, height: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def main() -> None:
    analyzer = Analyzer()

    detail = synthetic_image(1200, 900, seed=1)
    result = analyzer.analyze(detail, detail.shape[1], detail.shape[0])
    print("detail:", result.mode, result.sort_label_ru, "grains:", result.grain_count)

    panorama = synthetic_image(9000, 6000, seed=2)
    result = analyzer.analyze(panorama, panorama.shape[1], panorama.shape[0])
    print("panorama:", result.mode, result.sort_label_ru, "grains:", result.grain_count)

    print("OK")


if __name__ == "__main__":
    main()
