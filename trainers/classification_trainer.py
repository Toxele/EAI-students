from __future__ import annotations

import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
except ImportError:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from data.datasets import OreClassificationDataset
from losses.classification import LabelSmoothingCrossEntropy
from models.classifiers import ClassifierFactory


class ClassificationTrainer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        if torch is None:
            raise ImportError("ClassificationTrainer requires torch.")
        self.cfg = cfg
        self.run_dir = Path(cfg["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.classes = cfg["classes"]
        self.device = self._resolve_device(cfg.get("device", "auto"))
        self._set_seed(cfg.get("seed", 42))

        self.train_dataset = OreClassificationDataset(
            manifest_csv=cfg["manifest_csv"],
            classes=self.classes,
            subset="train",
            image_size=cfg["data"]["image_size"],
            include_sources=cfg.get("include_sources"),
            exclude_conflicts=cfg.get("exclude_conflicts", True),
        )
        self.val_dataset = OreClassificationDataset(
            manifest_csv=cfg["manifest_csv"],
            classes=self.classes,
            subset="val",
            image_size=cfg["data"]["image_size"],
            include_sources=cfg.get("include_sources"),
            exclude_conflicts=cfg.get("exclude_conflicts", True),
        )
        self.train_loader = self._make_loader(self.train_dataset, train=True)
        self.val_loader = self._make_loader(self.val_dataset, train=False)
        self.model = ClassifierFactory.create(cfg["model"], len(self.classes)).to(self.device)
        self.criterion = LabelSmoothingCrossEntropy()
        self.optimizer = self._make_optimizer()
        self.scaler = torch.cuda.amp.GradScaler(enabled=cfg["trainer"].get("amp", True) and self.device.type == "cuda")

    def fit(self) -> dict[str, Any]:
        best_score = -1.0
        stale_epochs = 0
        history: list[dict[str, float]] = []
        epoch_iter = range(1, self.cfg["trainer"]["epochs"] + 1)
        if tqdm is not None:
            epoch_iter = tqdm(epoch_iter, desc="epochs", unit="epoch")
        for epoch in epoch_iter:
            train_metrics = self._run_epoch(epoch=epoch, train=True)
            val_metrics = self._run_epoch(epoch=epoch, train=False)
            row = {"epoch": epoch, **{f"train_{k}": v for k, v in train_metrics.items()}, **{f"val_{k}": v for k, v in val_metrics.items()}}
            history.append(row)
            self._write_history(history)
            score = val_metrics["macro_f1"]
            if score > best_score:
                best_score = score
                stale_epochs = 0
                self._save_checkpoint("best.pt", epoch, val_metrics)
                self._snapshot_best_reports()
            else:
                stale_epochs += 1
            if epoch % self.cfg["trainer"].get("save_every", 1) == 0:
                self._save_checkpoint(f"epoch_{epoch:03d}.pt", epoch, val_metrics)
            patience = self.cfg["trainer"].get("early_stopping_patience", 0)
            if patience and stale_epochs >= patience:
                break
        summary = {"best_val_macro_f1": best_score, "epochs": len(history)}
        (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    def _run_epoch(self, epoch: int, train: bool) -> dict[str, float]:
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        total_loss = 0.0
        total_correct = 0
        total_items = 0
        confusion = np.zeros((len(self.classes), len(self.classes)), dtype=np.int64)
        phase = "train" if train else "val"
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(loader, desc=f"{phase} epoch {epoch}", unit="batch", leave=False)
        for batch in iterator:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device)
            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            preds = logits.argmax(dim=1)
            total_loss += float(loss.detach().cpu()) * labels.numel()
            total_correct += int((preds == labels).sum().detach().cpu())
            total_items += labels.numel()
            for true, pred in zip(labels.detach().cpu().tolist(), preds.detach().cpu().tolist()):
                confusion[true, pred] += 1
            if tqdm is not None:
                iterator.set_postfix(
                    loss=total_loss / max(total_items, 1),
                    acc=total_correct / max(total_items, 1),
                )
        metrics = self._metrics_from_confusion(confusion)
        metrics["loss"] = total_loss / max(total_items, 1)
        metrics["accuracy"] = total_correct / max(total_items, 1)
        if not train:
            self._write_confusion(confusion)
            self._write_per_class_metrics(metrics["per_class"])
            self._write_metrics_json(metrics)
        return {k: v for k, v in metrics.items() if k != "per_class"}

    def _make_loader(self, dataset: OreClassificationDataset, train: bool) -> DataLoader:
        batch_size = self.cfg["data"]["batch_size"]
        if train and self.cfg["data"].get("use_weighted_sampler", False):
            labels = dataset.labels()
            counts = np.bincount(labels, minlength=len(self.classes))
            weights = [1.0 / max(counts[label], 1) for label in labels]
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=self.cfg["data"].get("num_workers", 0))
        return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=self.cfg["data"].get("num_workers", 0))

    def _make_optimizer(self):
        cfg = self.cfg["optimizer"]
        if cfg.get("name", "adamw").lower() == "adamw":
            return torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
        if cfg.get("name", "").lower() == "sgd":
            return torch.optim.SGD(self.model.parameters(), lr=cfg["lr"], momentum=0.9, weight_decay=cfg.get("weight_decay", 0.0))
        raise ValueError(f"Unknown optimizer: {cfg.get('name')}")

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict[str, float]) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "classes": self.classes,
                "metrics": metrics,
                "config": self.cfg,
            },
            self.run_dir / name,
        )

    def _write_history(self, history: list[dict[str, float]]) -> None:
        path = self.run_dir / "history.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    def _write_confusion(self, matrix) -> None:
        path = self.run_dir / "confusion.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["true\\pred", *self.classes])
            for label, row in zip(self.classes, matrix.tolist()):
                writer.writerow([label, *row])

    def _metrics_from_confusion(self, matrix) -> dict[str, Any]:
        per_class: list[dict[str, float | str | int]] = []
        f1_values: list[float] = []
        precision_values: list[float] = []
        recall_values: list[float] = []
        for idx, label in enumerate(self.classes):
            tp = int(matrix[idx, idx])
            fp = int(matrix[:, idx].sum() - tp)
            fn = int(matrix[idx, :].sum() - tp)
            support = int(matrix[idx, :].sum())
            precision = tp / max(tp + fp, 1)
            recall = tp / max(tp + fn, 1)
            f1 = 2 * precision * recall / max(precision + recall, 1e-12)
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
            per_class.append(
                {
                    "class": label,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                }
            )
        return {
            "macro_precision": float(np.mean(precision_values)) if precision_values else 0.0,
            "macro_recall": float(np.mean(recall_values)) if recall_values else 0.0,
            "macro_f1": float(np.mean(f1_values)) if f1_values else 0.0,
            "per_class": per_class,
        }

    def _write_per_class_metrics(self, rows: list[dict[str, Any]]) -> None:
        path = self.run_dir / "per_class_metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["class", "precision", "recall", "f1", "support"])
            writer.writeheader()
            writer.writerows(rows)

    def _write_metrics_json(self, metrics: dict[str, Any]) -> None:
        path = self.run_dir / "metrics.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)

    def _snapshot_best_reports(self) -> None:
        for source, target in [
            ("metrics.json", "best_metrics.json"),
            ("confusion.csv", "best_confusion.csv"),
            ("per_class_metrics.csv", "best_per_class_metrics.csv"),
        ]:
            source_path = self.run_dir / source
            if source_path.exists():
                shutil.copyfile(source_path, self.run_dir / target)

    @staticmethod
    def _resolve_device(device: str):
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
