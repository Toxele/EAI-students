from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from data.datasets import OreClassificationDataset
from loggers.mlflow_utils import MlflowRun
from losses.classification import LabelSmoothingCrossEntropy
from models.classifiers import ClassifierFactory


def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_checkpoint(path: str, device):
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]
    classes = checkpoint.get("classes", cfg["classes"])
    model = ClassifierFactory.create(cfg["model"], len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg, classes


def evaluate_subset(model, cfg, classes: list[str], subset: str, output_dir: Path, device) -> dict[str, float]:
    dataset = OreClassificationDataset(
        manifest_csv=cfg["manifest_csv"],
        classes=classes,
        subset=subset,
        image_size=cfg["data"]["image_size"],
        include_sources=cfg.get("include_sources"),
        exclude_conflicts=cfg.get("exclude_conflicts", True),
        mask_channel=cfg.get("mask_channel"),
    )
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=False, num_workers=cfg["data"].get("num_workers", 0))
    criterion = LabelSmoothingCrossEntropy()
    rows = []
    y_true = []
    y_pred = []
    y_prob = []
    losses = []
    iterator = tqdm(loader, desc=f"eval cls {subset}", unit="batch") if tqdm is not None else loader
    with torch.no_grad():
        for batch in iterator:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            losses.extend([float(loss.detach().cpu())] * labels.numel())
            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.cpu().tolist())
            y_prob.extend(probs.cpu().numpy().tolist())
            for path, true_idx, pred_idx, prob in zip(batch["path"], labels.cpu().tolist(), preds.cpu().tolist(), probs.cpu().numpy().tolist()):
                row = {
                    "subset": subset,
                    "path": path,
                    "true_label": classes[true_idx],
                    "pred_label": classes[pred_idx],
                    "pred_prob": float(prob[pred_idx]),
                    "is_error": true_idx != pred_idx,
                }
                for idx, label in enumerate(classes):
                    row[f"prob_{label}"] = float(prob[idx])
                rows.append(row)

    y_prob_arr = np.array(y_prob, dtype=np.float32)
    metrics = {
        "loss": float(np.mean(losses)) if losses else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    try:
        metrics["roc_auc_ovr_macro"] = float(roc_auc_score(y_true, y_prob_arr, multi_class="ovr", average="macro"))
    except ValueError:
        metrics["roc_auc_ovr_macro"] = float("nan")

    subset_dir = output_dir / subset
    subset_dir.mkdir(parents=True, exist_ok=True)
    with (subset_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(classes))))
    with (subset_dir / "confusion.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *classes])
        for label, row in zip(classes, matrix.tolist()):
            writer.writerow([label, *row])

    save_error_sheet(rows, subset_dir / "error_examples.jpg")
    return metrics


def save_error_sheet(rows: list[dict], output_path: Path, max_items: int = 30, thumb: int = 180) -> None:
    errors = [row for row in rows if row["is_error"]]
    errors = sorted(errors, key=lambda row: row["pred_prob"], reverse=True)[:max_items]
    if not errors:
        return
    cols = 5
    sheet_rows = int(np.ceil(len(errors) / cols))
    sheet = Image.new("RGB", (cols * thumb, sheet_rows * (thumb + 46)), (24, 28, 32))
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(errors):
        x = (idx % cols) * thumb
        y = (idx // cols) * (thumb + 46)
        with Image.open(row["path"]) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
            image.thumbnail((thumb, thumb))
            sheet.paste(image, (x, y))
        text = f"T:{row['true_label']} P:{row['pred_label']} {row['pred_prob']:.2f}"
        draw.text((x + 3, y + thumb + 4), text[:32], fill=(235, 238, 242))
    sheet.save(output_path, quality=92)


def main() -> None:
    if torch is None:
        raise ImportError("evaluate_classifier requires torch.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/runs/classifier_baseline/best.pt")
    parser.add_argument("--output-dir", default="artifacts/evaluation/classifier")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--mlflow", action="store_true")
    args = parser.parse_args()

    device = resolve_device(args.device)
    model, cfg, classes = load_checkpoint(args.checkpoint, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mlflow_cfg = {
        **cfg.get("mlflow", {}),
        "enabled": args.mlflow or cfg.get("mlflow", {}).get("enabled", False),
        "run_name": "evaluate_classifier",
    }
    with MlflowRun(mlflow_cfg, run_name="evaluate_classifier") as mlrun:
        mlrun.log_params_flat({"checkpoint": args.checkpoint})
        summary = {}
        for subset in ["train", "val"]:
            metrics = evaluate_subset(model, cfg, classes, subset, output_dir, device)
            summary[subset] = metrics
            mlrun.log_metrics(metrics, prefix=f"{subset}_")
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        mlrun.log_artifacts(output_dir, artifact_path="classifier_evaluation")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
