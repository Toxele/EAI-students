"""
Оценка multi-label чекпоинта на train/val и сохранение предсказаний.

Запуск:
  py scripts/evaluate_multilabel.py --checkpoint artifacts/runs/multilabel_resnet18/best.pt
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from training.data.multilabel_dataset import MultiLabelOreDataset
from training.models.classifiers import ClassifierFactory


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    cfg = checkpoint["config"]
    tags = checkpoint.get("tags", cfg["tags"])
    threshold = float(checkpoint.get("threshold", cfg.get("threshold", 0.5)))
    model = ClassifierFactory.create(cfg["model"], len(tags)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, cfg, tags, threshold


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, tags: list[str], threshold: float) -> dict:
    """Считает полный набор multi-label метрик."""
    y_pred = (y_prob >= threshold).astype(np.int32)
    per_tag = []
    f1_list, prec_list, rec_list, auc_list = [], [], [], []

    for idx, name in enumerate(tags):
        true_col = y_true[:, idx]
        pred_col = y_pred[:, idx]
        prob_col = y_prob[:, idx]
        prec = float(precision_score(true_col, pred_col, zero_division=0))
        rec = float(recall_score(true_col, pred_col, zero_division=0))
        f1 = float(f1_score(true_col, pred_col, zero_division=0))
        auc = float("nan")
        if len(np.unique(true_col)) >= 2:
            auc = float(roc_auc_score(true_col, prob_col))
        prec_list.append(prec)
        rec_list.append(rec)
        f1_list.append(f1)
        if not np.isnan(auc):
            auc_list.append(auc)
        per_tag.append(
            {
                "tag": name,
                "precision": prec,
                "recall": rec,
                "f1": f1,
                "auc": auc,
                "support_pos": int(true_col.sum()),
            }
        )

    return {
        "macro_precision": float(np.mean(prec_list)),
        "macro_recall": float(np.mean(rec_list)),
        "macro_f1": float(np.mean(f1_list)),
        "macro_auc": float(np.mean(auc_list)) if auc_list else float("nan"),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "exact_match": float(accuracy_score(y_true, y_pred)),
        "hamming_loss": float(hamming_loss(y_true, y_pred)),
        "threshold": threshold,
        "per_tag": per_tag,
    }


def evaluate_subset(model, cfg, tags, threshold, subset, device) -> tuple[dict, list[dict]]:
    """Inference на subset; возвращает метрики и строки predictions.csv."""
    dataset = MultiLabelOreDataset(
        manifest_csv=cfg["manifest_csv"],
        subset=subset,
        image_size=cfg["data"]["image_size"],
        augmentation=False,
    )
    loader = DataLoader(dataset, batch_size=cfg["data"]["batch_size"], shuffle=False)
    y_true, y_prob = [], []
    pred_rows: list[dict] = []

    with torch.no_grad():
        for batch in loader:
            logits = model(batch["image"].to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
            labels = batch["labels"].numpy()
            y_true.append(labels)
            y_prob.append(probs)
            for i in range(len(probs)):
                pred_bits = (probs[i] >= threshold).astype(int)
                pred_rows.append(
                    {
                        "subset": subset,
                        "path": batch["path"][i],
                        "md5": batch["md5"][i],
                        "true_tags": batch["tags"][i],
                        "pred_talc": int(pred_bits[0]),
                        "pred_coarse": int(pred_bits[1]),
                        "pred_fine": int(pred_bits[2]),
                        "prob_talc": float(probs[i][0]),
                        "prob_coarse": float(probs[i][1]),
                        "prob_fine": float(probs[i][2]),
                    }
                )

    y_true_arr = np.vstack(y_true)
    y_prob_arr = np.vstack(y_prob)
    metrics = compute_metrics(y_true_arr, y_prob_arr, tags, threshold)
    metrics["subset"] = subset
    metrics["n_samples"] = len(y_true_arr)
    return metrics, pred_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    device = resolve_device(args.device)
    model, cfg, tags, threshold = load_checkpoint(checkpoint_path, device)
    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_path.parent / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = {}
    all_preds: list[dict] = []
    for subset in ("train", "val"):
        metrics, preds = evaluate_subset(model, cfg, tags, threshold, subset, device)
        all_metrics[subset] = metrics
        all_preds.extend(preds)
        (output_dir / f"metrics_{subset}.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    (output_dir / "metrics_all.json").write_text(
        json.dumps(all_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (output_dir / "predictions.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_preds[0].keys()))
        writer.writeheader()
        writer.writerows(all_preds)

    print(json.dumps(all_metrics, indent=2, ensure_ascii=False))
    print(f"Saved to {output_dir}")


if __name__ == "__main__":
    main()
