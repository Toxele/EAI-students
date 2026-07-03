from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import torch
    import torchvision.transforms.functional as TF
except ImportError:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from data.manifest import read_manifest
from hydra.json_config import load_config
from models.segmentation import SegmentationFactory


def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_model(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model = SegmentationFactory.create(cfg["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, cfg


def read_image(path: str, image_size: int):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    original = np.array(image)
    resized = TF.resize(image, [image_size, image_size])
    tensor = TF.to_tensor(resized)
    tensor = TF.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    return original, tensor.unsqueeze(0)


def image_pixel_count(path: str) -> int:
    with Image.open(path) as image:
        width, height = image.size
    return width * height


def overlay_mask(original_rgb: np.ndarray, prob: np.ndarray, alpha: float) -> np.ndarray:
    h, w = original_rgb.shape[:2]
    prob = cv2.resize(prob, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = np.zeros_like(original_rgb)
    heat[:, :, 2] = 255
    blended = cv2.addWeighted(original_rgb, 1.0, heat, alpha, 0)
    return np.where(prob[:, :, None] > 0.5, blended, original_rgb)


def safe_name(rel_path: str) -> str:
    return "__".join(Path(rel_path).parts).replace(" ", "_")


def main() -> None:
    if torch is None:
        raise ImportError("predict_talc_masks requires torch.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation/predict_talc_masks.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = resolve_device(cfg["device"])
    model, model_cfg = load_model(cfg["checkpoint"], device)
    rows = [row for row in read_manifest(cfg["manifest_csv"]) if row.get("source") in cfg.get("include_sources", [])]
    if cfg.get("max_images", 0):
        rows = rows[: cfg["max_images"]]

    output_dir = Path(cfg["output_dir"])
    mask_dir = output_dir / "masks"
    prob_dir = output_dir / "probability"
    overlay_dir = output_dir / "overlays"
    for path in [mask_dir, prob_dir, overlay_dir]:
        path.mkdir(parents=True, exist_ok=True)

    image_size = cfg.get("image_size", model_cfg["data"]["image_size"])
    report = []
    iterator = tqdm(rows, desc="predict talc", unit="image") if tqdm is not None else rows
    with torch.no_grad():
        for row in iterator:
            if cfg.get("max_pixels", 0) and image_pixel_count(row["path"]) > cfg["max_pixels"]:
                continue
            original, tensor = read_image(row["path"], image_size)
            logits = model(tensor.to(device))
            prob_small = torch.sigmoid(logits)[0, 0].detach().cpu().numpy()
            h, w = original.shape[:2]
            prob = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_CUBIC)
            mask = (prob >= cfg["threshold"]).astype(np.uint8) * 255
            name = safe_name(row["rel_path"])
            prob_path = prob_dir / f"{Path(name).stem}_prob.png"
            mask_path = mask_dir / f"{Path(name).stem}_mask.png"
            overlay_path = overlay_dir / f"{Path(name).stem}_overlay.jpg"
            cv2.imencode(".png", np.uint8(np.clip(prob, 0, 1) * 255))[1].tofile(str(prob_path))
            cv2.imencode(".png", mask)[1].tofile(str(mask_path))
            Image.fromarray(overlay_mask(original, prob, cfg.get("overlay_alpha", 0.45))).save(overlay_path, quality=92)
            report.append(
                {
                    "rel_path": row["rel_path"],
                    "folder_label": row.get("label", ""),
                    "talc_fraction": float((mask > 0).mean()),
                    "mask_path": str(mask_path),
                    "probability_path": str(prob_path),
                    "overlay_path": str(overlay_path),
                }
            )
    with (output_dir / "talc_prediction_report.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()) if report else [])
        if report:
            writer.writeheader()
            writer.writerows(report)
    print(f"wrote {output_dir}: rows={len(report)}")


if __name__ == "__main__":
    main()
