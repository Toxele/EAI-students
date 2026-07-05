from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

try:
    import torch
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from training.data.datasets import OreClassificationDataset
from training.hydra.json_config import load_config
from training.models.classifiers import ClassifierFactory


class UncertaintyPolicy:
    def __init__(
        self,
        min_confidence: float = 0.6,
        min_margin: float = 0.15,
        mark_label_conflicts_uncertain: bool = True,
    ) -> None:
        self.min_confidence = min_confidence
        self.min_margin = min_margin
        self.mark_label_conflicts_uncertain = mark_label_conflicts_uncertain

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "UncertaintyPolicy":
        return cls(
            min_confidence=cfg.get("min_confidence", 0.6),
            min_margin=cfg.get("min_margin", 0.15),
            mark_label_conflicts_uncertain=cfg.get("mark_label_conflicts_uncertain", True),
        )

    def decide(self, label: str, probs: list[float], label_conflict: bool) -> tuple[str, str]:
        ordered = sorted(probs, reverse=True)
        confidence = ordered[0]
        margin = ordered[0] - ordered[1] if len(ordered) > 1 else ordered[0]
        reasons: list[str] = []
        if self.mark_label_conflicts_uncertain and label_conflict:
            reasons.append("label_conflict")
        if confidence < self.min_confidence:
            reasons.append(f"low_confidence:{confidence:.4f}")
        if margin < self.min_margin:
            reasons.append(f"low_margin:{margin:.4f}")
        if reasons:
            return "uncertain", ";".join(reasons)
        return label, ""


def resolve_device(device: str):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def load_checkpoint_model(checkpoint_path: str | Path, device, fallback_classes: list[str]):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cfg = checkpoint.get("config", {})
    classes = checkpoint.get("classes", fallback_classes)
    model_cfg = cfg.get("model", {"name": "small_cnn"})
    model = ClassifierFactory.create(model_cfg, len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, classes


def main() -> None:
    if torch is None:
        raise ImportError("predict_classifier requires torch.")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/classifier/predict_classifier.json")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config, args.overrides)
    device = resolve_device(cfg.get("device", "auto"))
    model, classes = load_checkpoint_model(cfg["checkpoint"], device, cfg["classes"])
    dataset = OreClassificationDataset(
        manifest_csv=cfg["manifest_csv"],
        classes=classes,
        subset=None,
        image_size=cfg["data"]["image_size"],
        include_sources=cfg.get("include_sources"),
        exclude_conflicts=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        num_workers=cfg["data"].get("num_workers", 0),
    )
    policy = UncertaintyPolicy.from_config(cfg["uncertainty"])
    rows: list[dict[str, Any]] = []
    iterator = loader
    if tqdm is not None:
        iterator = tqdm(loader, desc="predict", unit="batch")
    with torch.no_grad():
        offset = 0
        for batch in iterator:
            images = batch["image"].to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).detach().cpu()
            for i in range(probs.shape[0]):
                source_row = dataset.rows[offset + i]
                prob_list = [float(x) for x in probs[i].tolist()]
                pred_idx = int(probs[i].argmax().item())
                raw_label = classes[pred_idx]
                final_label, reason = policy.decide(
                    raw_label,
                    prob_list,
                    str(source_row.get("label_conflict", "")).lower() == "true",
                )
                row = {
                    "path": source_row["path"],
                    "rel_path": source_row["rel_path"],
                    "folder_label": source_row["label"],
                    "final_label": final_label,
                    "raw_label": raw_label,
                    "uncertainty_reason": reason,
                    "confidence": max(prob_list),
                    "margin": sorted(prob_list, reverse=True)[0] - sorted(prob_list, reverse=True)[1],
                    "label_conflict": source_row.get("label_conflict", ""),
                    "duplicate_group": source_row.get("duplicate_group", ""),
                }
                for cls, prob in zip(classes, prob_list):
                    row[f"prob_{cls}"] = prob
                rows.append(row)
            offset += probs.shape[0]

    output = Path(cfg["output_csv"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {output}: rows={len(rows)}")


if __name__ == "__main__":
    main()

