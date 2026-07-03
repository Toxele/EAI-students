from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score

try:
    import torch
    import torchvision.transforms.functional as TF
except ImportError:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from data.segmentation_datasets import TalcSegmentationDataset
from loggers.mlflow_utils import MlflowRun
from losses.segmentation import TalcSegmentationLoss
from models.segmentation import SegmentationFactory


def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_checkpoint(path: str, device):
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]
    model = SegmentationFactory.create(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def binary_metrics(prob: np.ndarray, mask: np.ndarray, threshold: float) -> dict[str, float]:
    pred = prob >= threshold
    target = mask > 0
    intersection = np.logical_and(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    union = np.logical_or(pred, target).sum()
    dice = (2 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)
    iou = (intersection + 1e-7) / (union + 1e-7)
    auc = float("nan")
    if len(np.unique(target)) > 1:
        auc = float(roc_auc_score(target.astype(np.uint8).ravel(), prob.ravel()))
    return {
        "dice": float(dice),
        "f1": float(dice),
        "iou": float(iou),
        "roc_auc": auc,
        "true_fraction": float(target.mean()),
        "pred_fraction": float(pred.mean()),
        "fraction_mae": float(abs(pred.mean() - target.mean())),
    }


def overlay_prediction(image_rgb: np.ndarray, target: np.ndarray, pred: np.ndarray) -> np.ndarray:
    out = image_rgb.copy()
    true_mask = target > 0
    pred_mask = pred > 0
    overlap = np.logical_and(true_mask, pred_mask)
    only_true = np.logical_and(true_mask, ~pred_mask)
    only_pred = np.logical_and(pred_mask, ~true_mask)
    out[only_true] = (0.55 * out[only_true] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    out[only_pred] = (0.55 * out[only_pred] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
    out[overlap] = (0.50 * out[overlap] + 0.50 * np.array([255, 220, 0])).astype(np.uint8)
    return out


def evaluate_subset(model, cfg, subset: str, output_dir: Path, device, threshold: float) -> dict[str, float]:
    dataset = TalcSegmentationDataset(
        cfg["dataset_csv"],
        subset=subset,
        image_size=cfg["data"]["image_size"],
        augment=False,
    )
    criterion = TalcSegmentationLoss(**cfg["loss"])
    overlay_dir = output_dir / subset / "overlays"
    mask_dir = output_dir / subset / "pred_masks"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    losses = []
    iterator = tqdm(range(len(dataset)), desc=f"eval {subset}", unit="image") if tqdm is not None else range(len(dataset))
    with torch.no_grad():
        for idx in iterator:
            item = dataset[idx]
            image_tensor = item["image"].unsqueeze(0).to(device)
            target_tensor = item["mask"].unsqueeze(0).to(device)
            logits = model(image_tensor)
            loss = criterion(logits, target_tensor)
            losses.append(float(loss.detach().cpu()))
            prob = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            target_small = item["mask"][0].numpy()
            metrics = binary_metrics(prob, target_small, threshold)

            with Image.open(item["image_path"]) as img:
                original = np.asarray(ImageOps.exif_transpose(img).convert("RGB"))
            h, w = original.shape[:2]
            prob_full = cv2.resize(prob, (w, h), interpolation=cv2.INTER_CUBIC)
            target_full = cv2.resize(target_small, (w, h), interpolation=cv2.INTER_NEAREST)
            pred_full = (prob_full >= threshold).astype(np.uint8) * 255
            name = f"{subset}_{idx:04d}_{Path(item['image_path']).stem}"
            pred_path = mask_dir / f"{name}_pred.png"
            overlay_path = overlay_dir / f"{name}_overlay.jpg"
            cv2.imencode(".png", pred_full)[1].tofile(str(pred_path))
            Image.fromarray(overlay_prediction(original, target_full, pred_full)).save(overlay_path, quality=92)
            rows.append(
                {
                    "subset": subset,
                    "image_path": item["image_path"],
                    "mask_path": item["mask_path"],
                    "sample_type": item["sample_type"],
                    "prediction_path": str(pred_path),
                    "overlay_path": str(overlay_path),
                    "loss": float(loss.detach().cpu()),
                    **metrics,
                }
            )
    report_path = output_dir / f"{subset}_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    aggregate = {"loss": float(np.mean(losses)) if losses else 0.0}
    for key in ["dice", "f1", "iou", "roc_auc", "fraction_mae"]:
        values = np.array([row[key] for row in rows], dtype=np.float32)
        values = values[~np.isnan(values)]
        aggregate[key] = float(values.mean()) if values.size else float("nan")
    return aggregate


def main() -> None:
    if torch is None:
        raise ImportError("evaluate_talc_segmenter requires torch.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/runs/talc_segmenter/best.pt")
    parser.add_argument("--output-dir", default="artifacts/evaluation/talc_segmenter")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg = load_checkpoint(args.checkpoint, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow_cfg = {
        **cfg.get("mlflow", {}),
        "enabled": args.mlflow or cfg.get("mlflow", {}).get("enabled", False),
        "run_name": "evaluate_talc_segmenter",
    }
    with MlflowRun(mlflow_cfg, run_name="evaluate_talc_segmenter") as mlrun:
        mlrun.log_params_flat({"checkpoint": args.checkpoint, "threshold": args.threshold})
        summary = {}
        for subset in ["train", "val"]:
            metrics = evaluate_subset(model, cfg, subset, output_dir, device, args.threshold)
            summary[subset] = metrics
            mlrun.log_metrics(metrics, prefix=f"{subset}_")
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        mlrun.log_artifacts(output_dir, artifact_path="evaluation")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
