from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    import torch
    import torchvision.transforms.functional as TF
except ImportError:  # pragma: no cover
    torch = None

from data.spbgu_segmentation import read_spbgu_binary_mask, read_spbgu_image, resolve_project_path
from models.segmentation import SegmentationFactory


class SpbguMaskPredictor:
    """Load an SPbGU segmenter checkpoint and export masks/overlays."""

    def __init__(self, checkpoint_path: str | Path, device: str = "auto") -> None:
        """Load model weights and preprocessing config from a checkpoint."""
        if torch is None:
            raise ImportError("SpbguMaskPredictor requires torch and torchvision.")
        self.checkpoint_path = resolve_project_path(checkpoint_path)
        self.device = self._resolve_device(device)
        checkpoint = torch.load(self.checkpoint_path, map_location=self.device)
        self.cfg = checkpoint["config"]
        self.image_size = int(self.cfg["data"]["image_size"])
        self.in_channels = int(self.cfg["model"].get("in_channels", 3))
        self.model = SegmentationFactory.create(self.cfg["model"]).to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict_image(self, source_path: str | Path, input_kind: str = "txt") -> tuple[np.ndarray, Image.Image]:
        """Predict a probability map resized to the original source image size."""
        image = read_spbgu_image(source_path, input_kind)
        tensor = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
        prob = cv2.resize(prob, image.size, interpolation=cv2.INTER_LINEAR)
        return prob, image

    def save_manifest_predictions(
        self,
        manifest_csv: str | Path,
        output_dir: str | Path,
        threshold: float = 0.5,
        limit: int | None = None,
    ) -> None:
        """Save predicted masks and overlays for every row in a manifest."""
        output_dir = resolve_project_path(output_dir)
        masks_dir = output_dir / "predicted_masks"
        overlays_dir = output_dir / "overlays"
        comparisons_dir = output_dir / "gt_prediction_overlay"
        for directory in [masks_dir, overlays_dir, comparisons_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        rows = _read_rows(resolve_project_path(manifest_csv))
        if limit is not None:
            rows = rows[:limit]
        report_rows: list[dict[str, Any]] = []
        for row in rows:
            prob, image = self.predict_image(row["source_path"], row.get("input_kind", "txt"))
            pred_mask = (prob >= threshold).astype(np.uint8) * 255
            name = _safe_name(row.get("rel_path") or row.get("sample_id") or Path(row["source_path"]).name)
            mask_path = masks_dir / f"{name}.png"
            overlay_path = overlays_dir / f"{name}.png"
            comparison_path = comparisons_dir / f"{name}.png"
            Image.fromarray(pred_mask, mode="L").save(mask_path)
            _save_overlay(image, pred_mask, overlay_path, color=(255, 0, 0))
            if row.get("mask_path"):
                gt_mask = np.asarray(read_spbgu_binary_mask(row["mask_path"]))
                _save_gt_pred_comparison(image, gt_mask, pred_mask, comparison_path)
            report_rows.append(
                {
                    "sample_id": row.get("sample_id", ""),
                    "source_path": row["source_path"],
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path),
                    "pred_fraction": float((pred_mask > 0).mean()),
                }
            )
        _write_report(output_dir / "prediction_report.csv", report_rows)

    def _preprocess(self, image: Image.Image):
        """Resize and normalize an image exactly as in training."""
        image = TF.resize(image, [self.image_size, self.image_size])
        tensor = TF.to_tensor(image)
        if self.in_channels == 1:
            tensor = tensor.mean(dim=0, keepdim=True)
            return (tensor - 0.5) / 0.5
        return TF.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    @staticmethod
    def _resolve_device(device: str):
        """Resolve an explicit device string or select CUDA automatically."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)


def load_checkpoint_config(checkpoint_path: str | Path) -> dict[str, Any]:
    """Read the JSON-compatible config embedded in a checkpoint."""
    if torch is None:
        raise ImportError("load_checkpoint_config requires torch.")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    return checkpoint.get("config", {})


def _save_overlay(image: Image.Image, mask: np.ndarray, output_path: Path, color: tuple[int, int, int]) -> None:
    """Save a semi-transparent mask overlay on an RGB image."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    if mask.shape[:2] != rgb.shape[:2]:
        mask = cv2.resize(mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = rgb.copy()
    overlay[mask > 0] = 0.55 * overlay[mask > 0] + 0.45 * np.asarray(color, dtype=np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(output_path)


def _save_gt_pred_comparison(image: Image.Image, gt_mask: np.ndarray, pred_mask: np.ndarray, output_path: Path) -> None:
    """Save GT and prediction as green/red overlay on the same source image."""
    rgb = np.asarray(image.convert("RGB")).astype(np.float32)
    if gt_mask.shape[:2] != rgb.shape[:2]:
        gt_mask = cv2.resize(gt_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    if pred_mask.shape[:2] != rgb.shape[:2]:
        pred_mask = cv2.resize(pred_mask, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay = rgb.copy()
    gt = gt_mask > 0
    pred = pred_mask > 0
    overlay[gt] = 0.55 * overlay[gt] + 0.45 * np.asarray([0, 255, 0], dtype=np.float32)
    overlay[pred] = 0.55 * overlay[pred] + 0.45 * np.asarray([255, 0, 0], dtype=np.float32)
    overlap = gt & pred
    overlay[overlap] = 0.45 * overlay[overlap] + 0.55 * np.asarray([255, 255, 0], dtype=np.float32)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8), mode="RGB").save(output_path)


def _read_rows(manifest_csv: str | Path) -> list[dict[str, str]]:
    """Read manifest rows from CSV."""
    with Path(manifest_csv).open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a prediction CSV report."""
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_name(value: str) -> str:
    """Create a filesystem-friendly artifact name from a path-like value."""
    cleaned = value.replace("\\", "__").replace("/", "__").replace(":", "_")
    return cleaned.rsplit(".", 1)[0]
