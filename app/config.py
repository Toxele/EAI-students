"""Project paths and thresholds.

All magic numbers live here instead of being scattered through the code.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Where uploaded user files are stored
UPLOAD_DIR = PROJECT_ROOT / "uploads"

# Where analysis results are stored (overlay, json)
RESULTS_DIR = PROJECT_ROOT / "results"

# Images with more pixels than this are treated as a panorama
PANORAMA_PIXEL_THRESHOLD = 50_000_000

# Talc percentage threshold for the "talc ore" class (from the case spec)
TALC_PERCENT_THRESHOLD = 10.0

# Max image side processed directly (to avoid choking on 15k x 10k input)
MAX_PROCESS_SIDE = 4096

# Panoramas are cut into tiles no larger than this and processed one by one
# (see app/pipeline/tiling.py) — the models are trained at single-frame
# scale, and a global downscale of a giant panorama collapses talc/grains
# below a pixel.
PANORAMA_TILE_SIZE = 2500

# Context margin around each tile (px, clipped at image edges). Without it
# the model can't see neighboring pixels near a tile border, so predictions
# from adjacent tiles disagree at the seam (visible grid lines). Only the
# tile's core region is written to the result.
PANORAMA_TILE_MARGIN = 256

# Weight files are not committed to git (see .gitignore); fetch them once
# with scripts/download_weights.py.
WEIGHTS_DIR = PROJECT_ROOT / "weights"

# Unet++ fast_768 talc segmenter (trained on Kaggle). Overridable via env var.
_DEFAULT_TALC_WEIGHTS = WEIGHTS_DIR / "segmentator.pt"
TALC_SEGMENTER_WEIGHTS = Path(os.environ.get("TALC_SEGMENTER_WEIGHTS", _DEFAULT_TALC_WEIGHTS))

# Coarse/fine (ordinary/thin) binary classifier — ResNet34
# (training/scripts/train_coarse_fine.py, kaggle/train_coarse_fine_binary.ipynb).
_DEFAULT_ORE_CLASSIFIER_WEIGHTS = WEIGHTS_DIR / "classifier.pt"
ORE_CLASSIFIER_WEIGHTS = Path(
    os.environ.get("ORE_CLASSIFIER_WEIGHTS", _DEFAULT_ORE_CLASSIFIER_WEIGHTS)
)
