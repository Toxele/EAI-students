"""Unet++ (EfficientNet-B4) talc segmentation.

Training: kaggle/train_talc_segmentation.ipynb
Weights: weights/segmentator.pt
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray

try:
    import torch
    import segmentation_models_pytorch as smp
except ImportError:  # pragma: no cover
    torch = None
    smp = None

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "segmentator.pt"
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_talc_model(checkpoint_path: str | Path | None = None, device=None):
    """Load the Unet++ checkpoint (Kaggle fast_768 run).

    :return: (model, meta) — meta: img_size, encoder, val_dice, val_iou, epoch
    """
    if torch is None or smp is None:
        raise ImportError(
            "Requires torch and segmentation-models-pytorch: "
            "pip install torch segmentation-models-pytorch"
        )

    path = Path(checkpoint_path or DEFAULT_WEIGHTS)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
    encoder = checkpoint.get("encoder", "efficientnet-b4")
    img_size = tuple(checkpoint.get("img_size", (576, 768)))

    model = smp.UnetPlusPlus(
        encoder_name=encoder,
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None,
    )
    model.load_state_dict(checkpoint["model_state"])
    dev = device or torch.device("cpu")
    model = model.to(dev)
    model.eval()

    meta = {
        "encoder": encoder,
        "img_size": img_size,
        "val_dice": checkpoint.get("val_dice"),
        "val_iou": checkpoint.get("val_iou"),
        "epoch": checkpoint.get("epoch"),
        "experiment": checkpoint.get("experiment", "fast_768"),
        "checkpoint": str(path.resolve()),
    }
    return model, meta


def _to_tensor(image_rgb: NDArray[np.uint8], img_h: int, img_w: int, device):
    """RGB uint8 → tensor 1×3×H×W с ImageNet-нормализацией."""
    resized = cv2.resize(image_rgb, (img_w, img_h), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    x = torch.from_numpy(x.transpose(2, 0, 1)).unsqueeze(0).to(device)
    return x, resized.shape[:2]


@torch.no_grad()
def predict_talc_mask(
    model,
    image_rgb: NDArray[np.uint8],
    img_size: tuple[int, int] = (576, 768),
    device=None,
    threshold: float = 0.5,
) -> tuple[NDArray[np.uint8], float, NDArray[np.uint8]]:
    """
    Предсказание маски талька на полном разрешении.

    :param img_size: (H, W) как при обучении
    :return: бинарная маска H×W (0/255), доля талька %, карта уверенности
        H×W (0..255 — sigmoid-вероятность талька, чем выше тем увереннее)
    """
    if torch is None:
        raise ImportError("torch не установлен")

    dev = device or next(model.parameters()).device
    img_h, img_w = img_size
    full_h, full_w = image_rgb.shape[:2]

    batch, _ = _to_tensor(image_rgb, img_h, img_w, dev)
    prob = torch.sigmoid(model(batch))[0, 0].cpu().numpy()

    mask_small = (prob > threshold).astype(np.uint8) * 255
    mask_full = cv2.resize(mask_small, (full_w, full_h), interpolation=cv2.INTER_NEAREST)
    talc_percent = 100.0 * np.count_nonzero(mask_full) / max(full_h * full_w, 1)

    confidence_small = (prob * 255.0).astype(np.uint8)
    confidence_full = cv2.resize(confidence_small, (full_w, full_h), interpolation=cv2.INTER_LINEAR)
    return mask_full, round(talc_percent, 2), confidence_full
