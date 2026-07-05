"""
One-time download of model weights from Google Drive into weights/
(see app/config.py: WEIGHTS_DIR).

Downloads:
  - Unet++ talc segmenter            -> TALC_SEGMENTER_WEIGHTS
  - ResNet coarse/fine classifier    -> ORE_CLASSIFIER_WEIGHTS

Run (once): py scripts/download_weights.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import gdown
except ImportError as exc:
    raise SystemExit("Missing dependency, run: pip install gdown") from exc

from app.config import ORE_CLASSIFIER_WEIGHTS, TALC_SEGMENTER_WEIGHTS, WEIGHTS_DIR

# (Google Drive file id, destination path) — order matches the links
# provided for the case: talc segmenter, then classifier.
DOWNLOADS = [
    ("1UJA-9u68n3mC4lATmoK4uSstX4AtqB3D", TALC_SEGMENTER_WEIGHTS),
    ("17e8wTKVmGdwmNB_2OXhT-yeVPpKrmkXA", ORE_CLASSIFIER_WEIGHTS),
]


def main() -> None:
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    for file_id, destination in DOWNLOADS:
        if destination.exists():
            print(f"skip (already present): {destination}")
            continue
        print(f"downloading {file_id} -> {destination}")
        gdown.download(id=file_id, output=str(destination), quiet=False)
        if not destination.exists():
            raise RuntimeError(f"download failed: {destination}")
    print(f"weights ready in {WEIGHTS_DIR}")


if __name__ == "__main__":
    main()
