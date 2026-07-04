"""
Обучение soft binary классификатора coarse vs fine (BCE + sigmoid).
"""
from __future__ import annotations

import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, WeightedRandomSampler
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        precision_score,
        recall_score,
        roc_auc_score,
    )
except ImportError:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from data.coarse_fine_dataset import CoarseFineDataset
from loggers.mlflow_utils import MlflowRun
from models.classifiers import ClassifierFactory


class CoarseFineTrainer:
    """1 logit + BCEWithLogitsLoss; метрики на clean (0/1) и ambiguous (0.5)."""

    THRESHOLD = 0.5

    def __init__(self, cfg: dict[str, Any]) -> None:
        if torch is None:
            raise ImportError("CoarseFineTrainer requires torch.")
        self.cfg = cfg
        self.run_dir = Path(cfg["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(cfg.get("device", "auto"))
        self._set_seed(cfg.get("seed", 42))

        aug_cfg = cfg.get("augmentation")
        self.train_dataset = CoarseFineDataset(
            manifest_csv=cfg["manifest_csv"],
            subset="train",
            image_size=cfg["data"]["image_size"],
            augmentation=True,
            augmentation_cfg=aug_cfg,
        )
        self.val_dataset = CoarseFineDataset(
            manifest_csv=cfg["manifest_csv"],
            subset="val",
            image_size=cfg["data"]["image_size"],
            augmentation=False,
        )
        self.train_loader = self._make_loader(self.train_dataset, train=True)
        self.val_loader = self._make_loader(self.val_dataset, train=False)
        self.model = ClassifierFactory.create(cfg["model"], num_classes=1).to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=cfg["trainer"].get("amp", True) and self.device.type == "cuda"
        )

    def fit(self) -> dict[str, Any]:
        """Полный цикл; best checkpoint по composite_score."""
        with MlflowRun(self.cfg.get("mlflow"), run_name=self.cfg.get("run_name", "coarse_fine")) as mlrun:
            mlrun.log_params_flat(self.cfg)
            best_score = -1.0
            stale_epochs = 0
            history: list[dict[str, float]] = []
            epoch_iter = range(1, self.cfg["trainer"]["epochs"] + 1)
            if tqdm is not None:
                epoch_iter = tqdm(epoch_iter, desc="epochs", unit="epoch")

            for epoch in epoch_iter:
                if self.scheduler is not None and self.cfg.get("scheduler", {}).get("name") != "onecycle":
                    if epoch > 1:
                        self.scheduler.step()

                train_metrics = self._run_epoch(epoch=epoch, train=True)
                val_metrics = self._run_epoch(epoch=epoch, train=False)
                row = {
                    "epoch": epoch,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                history.append(row)
                self._write_history(history)
                mlrun.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
                mlrun.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)

                score = val_metrics["composite_score"]
                if score > best_score:
                    best_score = score
                    stale_epochs = 0
                    self._save_checkpoint("best.pt", epoch, val_metrics)
                    self._snapshot_best_reports()
                else:
                    stale_epochs += 1

                patience = self.cfg["trainer"].get("early_stopping_patience", 0)
                if patience and stale_epochs >= patience:
                    break

            summary = {
                "best_composite_score": best_score,
                "epochs": len(history),
            }
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            mlrun.log_metrics(summary)
            for name in ["history.csv", "best_metrics.json", "summary.json"]:
                path = self.run_dir / name
                if path.exists():
                    mlrun.log_artifact(path)
            return summary

    def _run_epoch(self, epoch: int, train: bool) -> dict[str, float]:
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        total_loss = 0.0
        total_items = 0
        y_true: list[float] = []
        y_prob: list[float] = []
        buckets: list[str] = []

        phase = "train" if train else "val"
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(loader, desc=f"{phase} epoch {epoch}", unit="batch", leave=False)

        for batch in iterator:
            images = batch["image"].to(self.device)
            targets = batch["target"].to(self.device)
            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    logits = self.model(images).squeeze(-1)
                    loss = self.criterion(logits, targets)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler is not None and self.cfg.get("scheduler", {}).get("name") == "onecycle":
                        self.scheduler.step()

            probs = torch.sigmoid(logits.detach())
            total_loss += float(loss.detach().cpu()) * targets.size(0)
            total_items += targets.size(0)
            y_true.extend(targets.detach().cpu().tolist())
            y_prob.extend(probs.cpu().tolist())
            buckets.extend(batch["label_bucket"])

            if tqdm is not None:
                iterator.set_postfix(loss=total_loss / max(total_items, 1))

        metrics = self._compute_metrics(y_true, y_prob, buckets)
        metrics["loss"] = total_loss / max(total_items, 1)
        if not train:
            self._write_metrics_json(metrics)
        return {k: v for k, v in metrics.items() if k not in ("per_bucket",)}

    def _compute_metrics(
        self,
        y_true: list[float],
        y_prob: list[float],
        buckets: list[str],
    ) -> dict[str, Any]:
        """Clean F1/AUC + ambiguous MAE + composite score."""
        y_true_arr = np.array(y_true, dtype=np.float32)
        y_prob_arr = np.array(y_prob, dtype=np.float32)
        bucket_arr = np.array(buckets)

        clean_mask = np.isin(bucket_arr, ["coarse", "fine"])
        amb_mask = bucket_arr == "ambiguous"

        if clean_mask.any():
            clean_true = y_true_arr[clean_mask]
            clean_prob = y_prob_arr[clean_mask]
            clean_pred = (clean_prob >= self.THRESHOLD).astype(np.int32)
            clean_bin = clean_true.astype(np.int32)
            metrics: dict[str, Any] = {
                "clean_accuracy": float(accuracy_score(clean_bin, clean_pred)),
                "clean_f1": float(f1_score(clean_bin, clean_pred, zero_division=0)),
                "clean_precision": float(precision_score(clean_bin, clean_pred, zero_division=0)),
                "clean_recall": float(recall_score(clean_bin, clean_pred, zero_division=0)),
                "clean_mae": float(mean_absolute_error(clean_true, clean_prob)),
            }
            if len(np.unique(clean_bin)) >= 2:
                metrics["clean_auc"] = float(roc_auc_score(clean_bin, clean_prob))
            else:
                metrics["clean_auc"] = float("nan")
        else:
            metrics = {
                "clean_accuracy": 0.0,
                "clean_f1": 0.0,
                "clean_precision": 0.0,
                "clean_recall": 0.0,
                "clean_auc": float("nan"),
                "clean_mae": float("nan"),
            }

        if amb_mask.any():
            amb_prob = y_prob_arr[amb_mask]
            metrics["ambiguous_mae"] = float(mean_absolute_error(np.full(amb_prob.shape, 0.5), amb_prob))
            metrics["ambiguous_in_band"] = float(np.mean((amb_prob >= 0.35) & (amb_prob <= 0.65)))
            metrics["ambiguous_count"] = int(amb_mask.sum())
        else:
            metrics["ambiguous_mae"] = float("nan")
            metrics["ambiguous_in_band"] = float("nan")
            metrics["ambiguous_count"] = 0

        clean_f1 = metrics["clean_f1"]
        amb_mae = metrics["ambiguous_mae"]
        if not math.isnan(amb_mae):
            metrics["composite_score"] = 0.85 * clean_f1 + 0.15 * (1.0 - min(amb_mae, 0.5) / 0.5)
        else:
            metrics["composite_score"] = clean_f1

        return metrics

    def _make_loader(self, dataset: CoarseFineDataset, train: bool) -> DataLoader:
        batch_size = self.cfg["data"]["batch_size"]
        if train and self.cfg["data"].get("use_weighted_sampler", True):
            amb_w = float(self.cfg["data"].get("ambiguous_sampler_weight", 0.12))
            weights = dataset.sampler_weights(ambiguous_weight=amb_w)
            sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
            return DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=self.cfg["data"].get("num_workers", 0),
            )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=train,
            num_workers=self.cfg["data"].get("num_workers", 0),
        )

    def _make_optimizer(self):
        cfg = self.cfg["optimizer"]
        if cfg.get("name", "adamw").lower() == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=cfg["lr"],
                weight_decay=cfg.get("weight_decay", 0.0),
            )
        raise ValueError(f"Unknown optimizer: {cfg.get('name')}")

    def _make_scheduler(self):
        sched_cfg = self.cfg.get("scheduler") or {}
        name = sched_cfg.get("name", "none").lower()
        if name == "none":
            return None
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.cfg["trainer"]["epochs"],
                eta_min=sched_cfg.get("eta_min", 1e-6),
            )
        if name == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_cfg.get("step_size", 8),
                gamma=sched_cfg.get("gamma", 0.1),
            )
        if name == "onecycle":
            steps_per_epoch = max(len(self.train_loader), 1)
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.cfg["optimizer"]["lr"],
                epochs=self.cfg["trainer"]["epochs"],
                steps_per_epoch=steps_per_epoch,
                pct_start=sched_cfg.get("pct_start", 0.1),
            )
        raise ValueError(f"Unknown scheduler: {name}")

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict[str, float]) -> None:
        payload: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "metrics": metrics,
            "config": self.cfg,
            "task": "coarse_fine_soft_binary",
        }
        if self.scheduler is not None:
            payload["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(payload, self.run_dir / name)

    def _write_history(self, history: list[dict[str, float]]) -> None:
        path = self.run_dir / "history.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    def _write_metrics_json(self, metrics: dict[str, Any]) -> None:
        (self.run_dir / "metrics.json").write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _snapshot_best_reports(self) -> None:
        source = self.run_dir / "metrics.json"
        if source.exists():
            shutil.copyfile(source, self.run_dir / "best_metrics.json")

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
