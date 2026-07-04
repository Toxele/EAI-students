"""
Обучение multi-label классификатора (BCE + sigmoid).
"""
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
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        hamming_loss,
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

from data.multilabel_dataset import MultiLabelOreDataset
from loggers.mlflow_utils import MlflowRun
from models.classifiers import ClassifierFactory


class MultiLabelTrainer:
    """Backbone + BCEWithLogitsLoss, метрики per-tag, macro и ig-decision."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        if torch is None:
            raise ImportError("MultiLabelTrainer requires torch.")
        self.cfg = cfg
        self.run_dir = Path(cfg["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tag_names: list[str] = cfg["tags"]
        self.threshold = float(cfg.get("threshold", 0.5))
        self.device = self._resolve_device(cfg.get("device", "auto"))
        self._set_seed(cfg.get("seed", 42))
        aug_cfg = cfg.get("augmentation", {})

        self.train_dataset = MultiLabelOreDataset(
            manifest_csv=cfg["manifest_csv"],
            subset="train",
            tags=self.tag_names,
            image_size=cfg["data"]["image_size"],
            augmentation=True,
            augmentation_cfg=aug_cfg,
        )
        self.val_dataset = MultiLabelOreDataset(
            manifest_csv=cfg["manifest_csv"],
            subset="val",
            tags=self.tag_names,
            image_size=cfg["data"]["image_size"],
            augmentation=False,
            augmentation_cfg={},
        )
        self.train_loader = self._make_loader(self.train_dataset, train=True)
        self.val_loader = self._make_loader(self.val_dataset, train=False)
        self.model = ClassifierFactory.create(cfg["model"], len(self.tag_names)).to(self.device)
        pos_weight = self._compute_pos_weight()
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = self._make_optimizer()
        self.scheduler = self._make_scheduler()
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=cfg["trainer"].get("amp", True) and self.device.type == "cuda"
        )

    def fit(self) -> dict[str, Any]:
        """Полный цикл обучения; возвращает summary с best macro-F1."""
        with MlflowRun(self.cfg.get("mlflow"), run_name=self.cfg.get("run_name", "multilabel")) as mlrun:
            mlrun.log_params_flat(self.cfg)
            score_key = self.cfg.get("selection_metric", "macro_f1")
            best_score = -1.0
            best_ig_f1 = -1.0
            best_macro_f1 = -1.0
            stale_epochs = 0
            history: list[dict[str, float]] = []
            epoch_iter = range(1, self.cfg["trainer"]["epochs"] + 1)
            if tqdm is not None:
                epoch_iter = tqdm(epoch_iter, desc="epochs", unit="epoch")

            for epoch in epoch_iter:
                train_metrics = self._run_epoch(epoch=epoch, train=True)
                val_metrics = self._run_epoch(epoch=epoch, train=False)
                if self.scheduler is not None and self.cfg.get("scheduler", {}).get("name", "cosine") == "cosine":
                    self.scheduler.step()
                row = {
                    "epoch": epoch,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
                history.append(row)
                self._write_history(history)
                mlrun.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
                mlrun.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)

                score = val_metrics.get(score_key, val_metrics["macro_f1"])
                if score > best_score:
                    best_score = score
                    best_ig_f1 = float(val_metrics.get("ig_macro_f1", best_ig_f1))
                    best_macro_f1 = float(val_metrics.get("macro_f1", best_macro_f1))
                    stale_epochs = 0
                    self._save_checkpoint("best.pt", epoch, val_metrics)
                    self._snapshot_best_reports()
                else:
                    stale_epochs += 1

                patience = self.cfg["trainer"].get("early_stopping_patience", 0)
                if patience and stale_epochs >= patience:
                    break

            summary = {
                "best_val_macro_f1": best_macro_f1,
                f"best_val_{score_key}": best_score,
                "best_val_ig_macro_f1": best_ig_f1,
                "epochs": len(history),
            }
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            mlrun.log_metrics(summary)
            for name in ["history.csv", "best_metrics.json", "best_per_tag_metrics.csv", "summary.json"]:
                path = self.run_dir / name
                if path.exists():
                    mlrun.log_artifact(path)
            return summary

    def _run_epoch(self, epoch: int, train: bool) -> dict[str, float]:
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        total_loss = 0.0
        total_items = 0
        y_true: list[list[int]] = []
        y_prob: list[list[float]] = []

        phase = "train" if train else "val"
        iterator = loader
        if tqdm is not None:
            iterator = tqdm(loader, desc=f"{phase} epoch {epoch}", unit="batch", leave=False)

        for batch in iterator:
            images = batch["image"].to(self.device)
            labels = batch["labels"].to(self.device)
            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
                    logits = self.model(images)
                    loss = self.criterion(logits, labels)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if self.scheduler is not None and self.cfg.get("scheduler", {}).get("name") == "onecycle":
                        self.scheduler.step()

            probs = torch.sigmoid(logits.detach())
            total_loss += float(loss.detach().cpu()) * labels.size(0)
            total_items += labels.size(0)
            y_true.extend(labels.detach().cpu().numpy().astype(int).tolist())
            y_prob.extend(probs.cpu().numpy().tolist())

            if tqdm is not None:
                iterator.set_postfix(loss=total_loss / max(total_items, 1))

        metrics = self._compute_metrics(y_true, y_prob)
        metrics["loss"] = total_loss / max(total_items, 1)
        if not train:
            self._write_metrics_json(metrics)
            self._write_per_tag_csv(metrics["per_tag"])
        return {k: v for k, v in metrics.items() if k != "per_tag"}

    def _compute_metrics(self, y_true: list[list[int]], y_prob: list[list[float]]) -> dict[str, Any]:
        """Per-tag, macro и ig-decision (coarse vs fine по argmax вероятностей)."""
        y_true_arr = np.array(y_true, dtype=np.int32)
        y_prob_arr = np.array(y_prob, dtype=np.float32)
        y_pred_arr = (y_prob_arr >= self.threshold).astype(np.int32)

        per_tag: list[dict[str, Any]] = []
        f1_list: list[float] = []
        precision_list: list[float] = []
        recall_list: list[float] = []
        auc_list: list[float] = []

        for idx, name in enumerate(self.tag_names):
            true_col = y_true_arr[:, idx]
            pred_col = y_pred_arr[:, idx]
            prob_col = y_prob_arr[:, idx]
            support = int(true_col.sum())
            precision = float(precision_score(true_col, pred_col, zero_division=0))
            recall = float(recall_score(true_col, pred_col, zero_division=0))
            f1 = float(f1_score(true_col, pred_col, zero_division=0))
            if len(np.unique(true_col)) < 2:
                auc = float("nan")
            else:
                auc = float(roc_auc_score(true_col, prob_col))
            precision_list.append(precision)
            recall_list.append(recall)
            f1_list.append(f1)
            if not np.isnan(auc):
                auc_list.append(auc)
            per_tag.append(
                {
                    "tag": name,
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "auc": auc,
                    "support_pos": support,
                    "support_total": len(true_col),
                }
            )

        metrics: dict[str, Any] = {
            "macro_precision": float(np.mean(precision_list)),
            "macro_recall": float(np.mean(recall_list)),
            "macro_f1": float(np.mean(f1_list)),
            "macro_auc": float(np.mean(auc_list)) if auc_list else float("nan"),
            "micro_f1": float(f1_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
            "micro_precision": float(precision_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
            "micro_recall": float(recall_score(y_true_arr, y_pred_arr, average="micro", zero_division=0)),
            "exact_match": float(accuracy_score(y_true_arr, y_pred_arr)),
            "hamming_loss": float(hamming_loss(y_true_arr, y_pred_arr)),
            "per_tag": per_tag,
        }
        metrics.update(self._compute_ig_decision_metrics(y_true_arr, y_prob_arr))
        return metrics

    def _compute_ig_decision_metrics(
        self, y_true_arr: np.ndarray, y_prob_arr: np.ndarray
    ) -> dict[str, float]:
        """
        Бинарное решение coarse vs fine: argmax(P(coarse), P(fine)).
        Сравнимо с бинарным классификатором.
        """
        if "coarse" not in self.tag_names or "fine" not in self.tag_names:
            return {}
        coarse_idx = self.tag_names.index("coarse")
        fine_idx = self.tag_names.index("fine")
        true_ig: list[int] = []
        pred_ig: list[int] = []
        for row_true, row_prob in zip(y_true_arr, y_prob_arr):
            if row_true[coarse_idx] == 1 and row_true[fine_idx] == 0:
                true_ig.append(0)
            elif row_true[fine_idx] == 1 and row_true[coarse_idx] == 0:
                true_ig.append(1)
            else:
                continue
            pred_ig.append(0 if row_prob[coarse_idx] >= row_prob[fine_idx] else 1)
        if not true_ig:
            return {}
        return {
            "ig_accuracy": float(accuracy_score(true_ig, pred_ig)),
            "ig_macro_f1": float(f1_score(true_ig, pred_ig, average="macro", zero_division=0)),
            "ig_coarse_f1": float(f1_score(true_ig, pred_ig, labels=[0], average="macro", zero_division=0)),
            "ig_fine_f1": float(f1_score(true_ig, pred_ig, labels=[1], average="macro", zero_division=0)),
        }

    def _compute_pos_weight(self) -> torch.Tensor | None:
        """Веса для BCE: (N - pos) / pos по каждому тегу."""
        if not self.cfg.get("use_pos_weight", True):
            return None
        vectors = np.array(self.train_dataset.label_vectors(), dtype=np.float32)
        n = len(vectors)
        weights = []
        for col in range(len(self.tag_names)):
            pos = max(float(vectors[:, col].sum()), 1.0)
            weights.append((n - pos) / pos)
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _make_loader(self, dataset: MultiLabelOreDataset, train: bool) -> DataLoader:
        batch_size = self.cfg["data"]["batch_size"]
        if train and self.cfg["data"].get("use_weighted_sampler", False):
            sampler_mode = self.cfg["data"].get("sampler_mode", "combo")
            if sampler_mode == "ig_label" and dataset.ig_labels():
                labels = dataset.ig_labels()
                counts: dict[str, int] = {}
                for label in labels:
                    counts[label] = counts.get(label, 0) + 1
                weights = [1.0 / counts[label] for label in labels]
            else:
                vectors = dataset.label_vectors()
                combo_counts: dict[tuple[int, ...], int] = {}
                for vec in vectors:
                    key = tuple(vec)
                    combo_counts[key] = combo_counts.get(key, 0) + 1
                weights = [1.0 / combo_counts[tuple(vec)] for vec in vectors]
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
        cfg = self.cfg.get("scheduler")
        if not cfg:
            return None
        name = cfg.get("name", "cosine").lower()
        epochs = self.cfg["trainer"]["epochs"]
        if name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs,
                eta_min=float(cfg.get("eta_min", 1e-6)),
            )
        if name == "onecycle":
            steps = len(self.train_loader) * epochs
            return torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=float(cfg.get("max_lr", self.cfg["optimizer"]["lr"])),
                total_steps=max(steps, 1),
                pct_start=float(cfg.get("pct_start", 0.1)),
            )
        raise ValueError(f"Unknown scheduler: {name}")

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict[str, float]) -> None:
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "tags": self.tag_names,
                "threshold": self.threshold,
                "metrics": metrics,
                "config": self.cfg,
            },
            self.run_dir / name,
        )

    def _write_history(self, history: list[dict[str, float]]) -> None:
        path = self.run_dir / "history.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    def _write_per_tag_csv(self, rows: list[dict[str, Any]]) -> None:
        path = self.run_dir / "per_tag_metrics.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["tag", "precision", "recall", "f1", "auc", "support_pos", "support_total"],
            )
            writer.writeheader()
            writer.writerows(rows)

    def _write_metrics_json(self, metrics: dict[str, Any]) -> None:
        payload = {k: v for k, v in metrics.items() if k != "per_tag"}
        payload["per_tag"] = metrics.get("per_tag", [])
        (self.run_dir / "metrics.json").write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _snapshot_best_reports(self) -> None:
        for source, target in [
            ("metrics.json", "best_metrics.json"),
            ("per_tag_metrics.csv", "best_per_tag_metrics.csv"),
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
