from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import torch
    import torchvision.transforms as T
except ImportError:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from hydra.json_config import load_config
from models.classifiers import ClassifierFactory


class GradCam:
    def __init__(self, model, target_layer) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _module, _inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, _module, _grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor, target_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(image_tensor)
        logits[:, target_idx].sum().backward()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1).relu()
        cam = cam[0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= max(cam.max(), 1e-8)
        return cam


def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_checkpoint_model(checkpoint_path: str | Path, device, fallback_classes: list[str]):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    classes = checkpoint.get("classes", fallback_classes)
    model = ClassifierFactory.create(cfg.get("model", {"name": "small_cnn"}), len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def find_target_layer(model):
    if hasattr(model, "layer4"):
        return model.layer4[-1]
    if hasattr(model, "features"):
        for layer in reversed(model.features):
            if hasattr(layer, "weight") and len(getattr(layer, "weight").shape) == 4:
                return layer
    raise ValueError("Could not infer Grad-CAM target layer for this model.")


def transform_image(path: str | Path, image_size: int):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    original = np.array(image)
    transform = T.Compose(
        [
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return original, transform(image).unsqueeze(0)


def overlay_cam(original_rgb: np.ndarray, cam: np.ndarray, alpha: float) -> np.ndarray:
    h, w = original_rgb.shape[:2]
    cam = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(original_rgb, 1.0 - alpha, heat, alpha, 0)


def safe_name(rel_path: str) -> str:
    return "__".join(Path(rel_path).parts).replace(" ", "_")


def main() -> None:
    if torch is None:
        raise ImportError("explain_classifier requires torch and torchvision.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classifier/explain_classifier.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = resolve_device(cfg.get("device", "auto"))
    model, classes = load_checkpoint_model(cfg["checkpoint"], device, cfg["classes"])
    cammer = GradCam(model, find_target_layer(model))

    with Path(cfg["manifest_csv"]).open("r", newline="", encoding="utf-8") as f:
        rows = [row for row in csv.DictReader(f) if row.get("source") in cfg.get("include_sources", [])]
    if cfg.get("only_uncertain_or_conflict", True):
        rows = [row for row in rows if row.get("label_conflict", "").lower() == "true"]
    rows = rows[: cfg.get("max_images", 48)]

    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    report: list[dict[str, Any]] = []
    iterator = tqdm(rows, desc="gradcam", unit="image") if tqdm is not None else rows
    for row in iterator:
        original, tensor = transform_image(row["path"], cfg["data"]["image_size"])
        tensor = tensor.to(device)
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
        target_idx = int(np.argmax(probs))
        cam = cammer(tensor, target_idx)
        overlay = overlay_cam(original, cam, cfg.get("alpha", 0.45))
        out_path = output_dir / f"{safe_name(row['rel_path'])}__{classes[target_idx]}.jpg"
        Image.fromarray(overlay).save(out_path, quality=92)
        item = {
            "rel_path": row["rel_path"],
            "folder_label": row["label"],
            "pred_label": classes[target_idx],
            "confidence": float(probs[target_idx]),
            "overlay": str(out_path),
            "label_conflict": row.get("label_conflict", ""),
        }
        for cls, prob in zip(classes, probs.tolist()):
            item[f"prob_{cls}"] = float(prob)
        report.append(item)

    report_path = output_dir / "gradcam_report.csv"
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0].keys()) if report else [])
        if report:
            writer.writeheader()
            writer.writerows(report)
    print(f"wrote {output_dir}: images={len(report)}")


if __name__ == "__main__":
    main()

