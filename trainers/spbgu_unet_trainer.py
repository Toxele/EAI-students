from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from torch.utils.data import DataLoader
except ImportError:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from data.spbgu_segmentation import SpbguSegmentationDataset, resolve_project_path, split_rows_by_subset
from losses.segmentation import TalcSegmentationLoss
from models.segmentation import SegmentationFactory


class SpbguUNetTrainer:
    """Train and validate a binary U-Net segmenter on SPbGU AFM data."""

    def __init__(self, cfg: dict[str, Any], train_rows: list[dict[str, str]] | None = None, val_rows: list[dict[str, str]] | None = None) -> None:
        """Initialize datasets, loaders, model, loss and optimizer."""
        if torch is None:
            raise ImportError("SpbguUNetTrainer requires torch, torchvision and scikit-learn.")
        self.cfg = cfg
        self.run_dir = resolve_project_path(cfg["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(cfg.get("device", "auto"))
        self._set_seed(int(cfg.get("seed", 42)))
        train_rows = train_rows or split_rows_by_subset(cfg["dataset_csv"], "train")
        val_rows = val_rows or split_rows_by_subset(cfg["dataset_csv"], "val")
        self.train_dataset = self._make_dataset(train_rows, augment=cfg["data"].get("augment", True))
        self.val_dataset = self._make_dataset(val_rows, augment=False)
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=True,
            num_workers=cfg["data"].get("num_workers", 0),
        )
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=cfg["data"]["batch_size"],
            shuffle=False,
            num_workers=cfg["data"].get("num_workers", 0),
        )
        self.model = SegmentationFactory.create(cfg["model"]).to(self.device)
        self.criterion = TalcSegmentationLoss(**cfg["loss"])
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=cfg["optimizer"]["lr"],
            weight_decay=cfg["optimizer"].get("weight_decay", 0.0),
        )
        self.scaler = torch.amp.GradScaler(
            "cuda",
            enabled=cfg["trainer"].get("amp", True) and self.device.type == "cuda",
        )

    def fit(self) -> dict[str, Any]:
        """Run training, save best checkpoint and write metric artifacts."""
        best_score = -1.0
        stale_epochs = 0
        history: list[dict[str, float]] = []
        epochs = range(1, self.cfg["trainer"]["epochs"] + 1)
        if tqdm is not None:
            epochs = tqdm(epochs, desc="spbgu epochs", unit="epoch")
        for epoch in epochs:
            train_metrics = self._run_epoch(epoch, train=True)
            val_metrics = self._run_epoch(epoch, train=False)
            row = {
                "epoch": epoch,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(row)
            self._write_history(history)
            score = val_metrics[self.cfg["trainer"].get("monitor", "dice")]
            if score > best_score:
                best_score = score
                stale_epochs = 0
                self._save_checkpoint("best.pt", epoch, val_metrics)
                self._write_json("best_metrics.json", val_metrics)
            else:
                stale_epochs += 1
            if epoch % self.cfg["trainer"].get("save_every", 1) == 0:
                self._save_checkpoint(f"epoch_{epoch:03d}.pt", epoch, val_metrics)
            patience = int(self.cfg["trainer"].get("early_stopping_patience", 0))
            if patience and stale_epochs >= patience:
                break
        summary = {"best_val_score": best_score, "epochs": len(history)}
        self._write_json("summary.json", summary)
        self._write_json("resolved_config.json", self.cfg)
        return summary

    def _make_dataset(self, rows: list[dict[str, str]], augment: bool) -> SpbguSegmentationDataset:
        """Create an SPBgu dataset using shared config options."""
        return SpbguSegmentationDataset(
            rows,
            image_size=self.cfg["data"]["image_size"],
            augment=augment,
            augmentation_cfg=self.cfg.get("augmentation", {}),
            in_channels=self.cfg["model"].get("in_channels", 3),
        )

    def _run_epoch(self, epoch: int, train: bool) -> dict[str, float]:
        """Run one train or validation epoch and aggregate segmentation metrics."""
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        phase = "train" if train else "val"
        iterator = tqdm(loader, desc=f"{phase} spbgu {epoch}", unit="batch", leave=False) if tqdm is not None else loader
        total_loss = 0.0
        total_items = 0
        sums = {"dice": 0.0, "f1": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0, "mask_fraction_mae": 0.0}
        auc_probs: list[np.ndarray] = []
        auc_targets: list[np.ndarray] = []
        threshold = float(self.cfg["trainer"].get("threshold", 0.5))
        for batch in iterator:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            with torch.set_grad_enabled(train):
                with torch.amp.autocast("cuda", enabled=self.scaler.is_enabled()):
                    logits = self.model(images)
                    loss = self.criterion(logits, masks)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            batch_size = masks.shape[0]
            metrics = self._batch_metrics(logits.detach(), masks, threshold)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_items += batch_size
            for key in sums:
                sums[key] += metrics[key] * batch_size
            if self.cfg["trainer"].get("roc_auc", True):
                auc_probs.append(torch.sigmoid(logits.detach()).float().cpu().numpy().ravel())
                auc_targets.append(masks.detach().float().cpu().numpy().ravel())
            if tqdm is not None:
                iterator.set_postfix(loss=total_loss / max(total_items, 1), dice=sums["dice"] / max(total_items, 1))
        result = {key: value / max(total_items, 1) for key, value in sums.items()}
        result["loss"] = total_loss / max(total_items, 1)
        result["roc_auc"] = self._roc_auc(auc_targets, auc_probs) if auc_probs else float("nan")
        return result

    @staticmethod
    def _batch_metrics(logits, masks, threshold: float) -> dict[str, float]:
        """Calculate batch-mean binary segmentation metrics."""
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        dims = (1, 2, 3)
        tp = (preds * masks).sum(dims)
        fp = (preds * (1 - masks)).sum(dims)
        fn = ((1 - preds) * masks).sum(dims)
        pred_sum = preds.sum(dims)
        target_sum = masks.sum(dims)
        dice = (2 * tp + 1e-7) / (pred_sum + target_sum + 1e-7)
        iou = (tp + 1e-7) / (tp + fp + fn + 1e-7)
        precision = (tp + 1e-7) / (tp + fp + 1e-7)
        recall = (tp + 1e-7) / (tp + fn + 1e-7)
        fraction_mae = (preds.mean(dims) - masks.mean(dims)).abs()
        return {
            "dice": float(dice.mean().detach().cpu()),
            "f1": float(dice.mean().detach().cpu()),
            "iou": float(iou.mean().detach().cpu()),
            "precision": float(precision.mean().detach().cpu()),
            "recall": float(recall.mean().detach().cpu()),
            "mask_fraction_mae": float(fraction_mae.mean().detach().cpu()),
        }

    @staticmethod
    def _roc_auc(target_chunks: list[np.ndarray], prob_chunks: list[np.ndarray]) -> float:
        """Calculate pixel-level ROC-AUC when both classes are present."""
        targets = np.concatenate(target_chunks)
        probs = np.concatenate(prob_chunks)
        if len(np.unique(targets)) < 2:
            return float("nan")
        return float(roc_auc_score(targets, probs))

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict[str, float]) -> None:
        """Save a checkpoint that is directly usable for inference."""
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "metrics": metrics,
                "config": self.cfg,
            },
            self.run_dir / name,
        )

    def _write_history(self, history: list[dict[str, float]]) -> None:
        """Write the training history CSV after each epoch."""
        with (self.run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

    def _write_json(self, name: str, data: dict[str, Any]) -> None:
        """Write a JSON artifact under the current run directory."""
        (self.run_dir / name).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _resolve_device(device: str):
        """Resolve an explicit device string or choose CUDA automatically."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @staticmethod
    def _set_seed(seed: int) -> None:
        """Seed Python, NumPy and Torch RNGs for repeatable splits and training."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


class SpbguCrossValidator:
    """Run stratified k-fold validation for the SPbGU U-Net segmenter."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Store the base config and full manifest rows."""
        if torch is None:
            raise ImportError("SpbguCrossValidator requires torch and scikit-learn.")
        self.cfg = cfg
        with Path(cfg["dataset_csv"]).open("r", newline="", encoding="utf-8") as f:
            self.rows = list(csv.DictReader(f))

    def run(self) -> dict[str, Any]:
        """Train one model per fold and write aggregate metrics."""
        cv_cfg = self.cfg.get("cross_validation", {})
        folds = int(cv_cfg.get("folds", 5))
        output_dir = resolve_project_path(
            cv_cfg.get("run_dir", Path(self.cfg["run_dir"]).with_name(Path(self.cfg["run_dir"]).name + "_5fold"))
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        labels = [row.get("domain_label", "unknown") for row in self.rows]
        splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=int(self.cfg.get("seed", 42)))
        fold_summaries: list[dict[str, Any]] = []
        for fold_idx, (train_idx, val_idx) in enumerate(splitter.split(np.zeros(len(self.rows)), labels)):
            fold_dir = output_dir / f"fold_{fold_idx:02d}"
            fold_cfg = json.loads(json.dumps(self.cfg))
            fold_cfg["run_dir"] = str(fold_dir)
            train_rows = [dict(self.rows[i], subset="train") for i in train_idx]
            val_rows = [dict(self.rows[i], subset="val") for i in val_idx]
            trainer = SpbguUNetTrainer(fold_cfg, train_rows=train_rows, val_rows=val_rows)
            summary = trainer.fit()
            best_metrics = json.loads((fold_dir / "best_metrics.json").read_text(encoding="utf-8"))
            fold_summaries.append({"fold": fold_idx, **summary, **best_metrics})
        self._write_cv_outputs(output_dir, fold_summaries)
        return {"folds": folds, "run_dir": str(output_dir)}

    @staticmethod
    def _write_cv_outputs(output_dir: Path, rows: list[dict[str, Any]]) -> None:
        """Write per-fold and mean/std CV metric tables."""
        if not rows:
            return
        with (output_dir / "cv_metrics.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        metric_keys = [key for key in rows[0] if key not in {"fold"} and isinstance(rows[0][key], (int, float))]
        summary = {}
        for key in metric_keys:
            values = [float(row[key]) for row in rows if not np.isnan(float(row[key]))]
            if values:
                summary[key] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
        (output_dir / "cv_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
