from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

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
from loggers.mlflow_utils import MlflowRun
from models.classifiers import ClassifierFactory
from models.segmentation import SegmentationFactory


FINAL_CLASSES = ["ordinary", "thin", "talc"]


def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_classifier(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    classes = checkpoint.get("classes", cfg["classes"])
    model = ClassifierFactory.create(cfg["model"], len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg, classes


def load_segmenter(checkpoint_path: str, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint["config"]
    model = SegmentationFactory.create(cfg["model"]).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg


def image_tensor(path: str, image_size: int):
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    resized = TF.resize(image, [image_size, image_size])
    tensor = TF.to_tensor(resized)
    return TF.normalize(tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)).unsqueeze(0)


def predict_talc_fraction(segmenter, image_path: str, image_size: int, mask_threshold: float, device) -> float:
    tensor = image_tensor(image_path, image_size).to(device)
    with torch.no_grad():
        logits = segmenter(tensor)
        prob = torch.sigmoid(logits)[0, 0]
    return float((prob >= mask_threshold).float().mean().detach().cpu())


def predict_base_class(classifier, classes: list[str], image_path: str, image_size: int, device) -> tuple[str, float, dict[str, float]]:
    tensor = image_tensor(image_path, image_size).to(device)
    with torch.no_grad():
        probs = torch.softmax(classifier(tensor), dim=1)[0].detach().cpu().numpy()
    pred_idx = int(np.argmax(probs))
    return classes[pred_idx], float(probs[pred_idx]), {label: float(probs[idx]) for idx, label in enumerate(classes)}


def filter_rows(rows: list[dict[str, str]], subset: str, include_sources: list[str], exclude_conflicts: bool) -> list[dict[str, str]]:
    output = []
    for row in rows:
        if subset != "all" and row.get("subset") != subset:
            continue
        if include_sources and row.get("source") not in include_sources:
            continue
        if exclude_conflicts and str(row.get("label_conflict", "")).lower() == "true":
            continue
        if row.get("label") not in FINAL_CLASSES:
            continue
        output.append(row)
    return output


def evaluate(args) -> dict[str, dict[str, float]]:
    device = resolve_device(args.device)
    classifier, classifier_cfg, base_classes = load_classifier(args.base_checkpoint, device)
    segmenter, segmenter_cfg = load_segmenter(args.segmenter_checkpoint, device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows = read_manifest(args.manifest)
    include_sources = classifier_cfg.get("include_sources", ["classification"])
    seg_image_size = args.segmenter_image_size or segmenter_cfg["data"]["image_size"]
    cls_image_size = args.classifier_image_size or classifier_cfg["data"]["image_size"]

    summary = {}
    with MlflowRun(
        {
            **classifier_cfg.get("mlflow", {}),
            "enabled": args.mlflow or classifier_cfg.get("mlflow", {}).get("enabled", False),
            "experiment": args.mlflow_experiment,
            "run_name": "business_rule_talc_override",
        },
        run_name="business_rule_talc_override",
    ) as mlrun:
        mlrun.log_params_flat(
            {
                "base_checkpoint": args.base_checkpoint,
                "segmenter_checkpoint": args.segmenter_checkpoint,
                "talc_fraction_threshold": args.talc_fraction_threshold,
                "mask_threshold": args.mask_threshold,
                "subsets": args.subsets,
            }
        )
        for subset in args.subsets:
            rows = filter_rows(all_rows, subset, include_sources, classifier_cfg.get("exclude_conflicts", True))
            subset_dir = output_dir / subset
            subset_dir.mkdir(parents=True, exist_ok=True)
            predictions = []
            iterator = tqdm(rows, desc=f"business rule {subset}", unit="image") if tqdm is not None else rows
            for row in iterator:
                talc_fraction = predict_talc_fraction(segmenter, row["path"], seg_image_size, args.mask_threshold, device)
                base_pred, base_prob, base_probs = predict_base_class(classifier, base_classes, row["path"], cls_image_size, device)
                talc_override = talc_fraction > args.talc_fraction_threshold
                final_pred = "talc" if talc_override else base_pred
                out_row = {
                    "subset": subset,
                    "path": row["path"],
                    "rel_path": row["rel_path"],
                    "true_label": row["label"],
                    "final_pred": final_pred,
                    "is_error": row["label"] != final_pred,
                    "talc_fraction": talc_fraction,
                    "talc_override": talc_override,
                    "base_pred": base_pred,
                    "base_prob": base_prob,
                }
                for label, prob in base_probs.items():
                    out_row[f"prob_base_{label}"] = prob
                predictions.append(out_row)
                if tqdm is not None:
                    iterator.set_postfix(talc=f"{talc_fraction:.3f}", pred=final_pred)

            metrics = write_outputs(predictions, subset_dir)
            summary[subset] = metrics
            mlrun.log_metrics(metrics, prefix=f"{subset}_")
        (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        mlrun.log_artifacts(output_dir, artifact_path="business_rule_evaluation")
    return summary


def write_outputs(rows: list[dict], output_dir: Path) -> dict[str, float]:
    fieldnames = list(rows[0].keys()) if rows else []
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    y_true = [FINAL_CLASSES.index(row["true_label"]) for row in rows]
    y_pred = [FINAL_CLASSES.index(row["final_pred"]) for row in rows]
    matrix = confusion_matrix(y_true, y_pred, labels=list(range(len(FINAL_CLASSES)))) if rows else np.zeros((3, 3), dtype=int)
    with (output_dir / "confusion.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred", *FINAL_CLASSES])
        for label, row in zip(FINAL_CLASSES, matrix.tolist()):
            writer.writerow([label, *row])

    save_error_sheet(rows, output_dir / "error_examples.jpg")

    non_talc_rows = [row for row in rows if row["true_label"] in {"ordinary", "thin"}]
    talc_true = [1 if row["true_label"] == "talc" else 0 for row in rows]
    talc_pred = [1 if row["final_pred"] == "talc" else 0 for row in rows]
    metrics = {
        "n": float(len(rows)),
        "accuracy": float(accuracy_score(y_true, y_pred)) if rows else 0.0,
        "macro_precision": float(precision_score(y_true, y_pred, average="macro", labels=list(range(3)), zero_division=0)) if rows else 0.0,
        "macro_recall": float(recall_score(y_true, y_pred, average="macro", labels=list(range(3)), zero_division=0)) if rows else 0.0,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=list(range(3)), zero_division=0)) if rows else 0.0,
        "talc_precision": float(precision_score(talc_true, talc_pred, zero_division=0)) if rows else 0.0,
        "talc_recall": float(recall_score(talc_true, talc_pred, zero_division=0)) if rows else 0.0,
        "talc_f1": float(f1_score(talc_true, talc_pred, zero_division=0)) if rows else 0.0,
        "talc_override_rate": float(np.mean([bool(row["talc_override"]) for row in rows])) if rows else 0.0,
        "mean_talc_fraction": float(np.mean([row["talc_fraction"] for row in rows])) if rows else 0.0,
    }
    if non_talc_rows:
        metrics["base_accuracy_on_true_non_talc"] = float(
            np.mean([row["true_label"] == row["base_pred"] for row in non_talc_rows])
        )
    else:
        metrics["base_accuracy_on_true_non_talc"] = 0.0
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def save_error_sheet(rows: list[dict], output_path: Path, max_items: int = 30, thumb: int = 180) -> None:
    errors = [row for row in rows if row["is_error"]]
    errors = sorted(errors, key=lambda row: max(float(row.get("base_prob", 0.0)), float(row.get("talc_fraction", 0.0))), reverse=True)
    errors = errors[:max_items]
    if not errors:
        return
    cols = 5
    sheet_rows = int(np.ceil(len(errors) / cols))
    sheet = Image.new("RGB", (cols * thumb, sheet_rows * (thumb + 62)), (24, 28, 32))
    draw = ImageDraw.Draw(sheet)
    for idx, row in enumerate(errors):
        x = (idx % cols) * thumb
        y = (idx // cols) * (thumb + 62)
        with Image.open(row["path"]) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
            image.thumbnail((thumb, thumb))
            sheet.paste(image, (x, y))
        text_1 = f"T:{row['true_label']} P:{row['final_pred']}"
        text_2 = f"talc={row['talc_fraction']:.2f} base={row['base_pred']}:{row['base_prob']:.2f}"
        draw.text((x + 3, y + thumb + 4), text_1[:32], fill=(235, 238, 242))
        draw.text((x + 3, y + thumb + 24), text_2[:34], fill=(235, 238, 242))
    sheet.save(output_path, quality=92)


def main() -> None:
    if torch is None:
        raise ImportError("evaluate_business_rule requires torch.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-checkpoint", default="artifacts/runs/base_ore_classifier/best.pt")
    parser.add_argument("--segmenter-checkpoint", default="artifacts/runs/talc_segmenter/best.pt")
    parser.add_argument("--manifest", default="artifacts/manifests/nornikel_manifest.csv")
    parser.add_argument("--output-dir", default="artifacts/evaluation/business_rule")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--subsets", nargs="+", default=["train", "val"])
    parser.add_argument("--talc-fraction-threshold", type=float, default=0.10)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--segmenter-image-size", type=int, default=0)
    parser.add_argument("--classifier-image-size", type=int, default=0)
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--mlflow-experiment", default="nornikel_business_rule")
    args = parser.parse_args()
    summary = evaluate(args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
