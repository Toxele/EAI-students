from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Any

try:
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import roc_auc_score
except ImportError:  # pragma: no cover
    torch = None

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover
    tqdm = None

from data.segmentation_datasets import TalcSegmentationDataset
from loggers.mlflow_utils import MlflowRun
from losses.segmentation import TalcSegmentationLoss
from models.segmentation import SegmentationFactory


class TalcSegmentationTrainer:
    def __init__(self, cfg: dict[str, Any]) -> None:
        if torch is None:
            raise ImportError("TalcSegmentationTrainer requires torch.")
        self.cfg = cfg
        self.run_dir = Path(cfg["run_dir"])
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = self._resolve_device(cfg.get("device", "auto"))
        self._set_seed(cfg.get("seed", 42))
        self.train_dataset = TalcSegmentationDataset(
            cfg["dataset_csv"],
            subset="train",
            image_size=cfg["data"]["image_size"],
            augment=cfg["data"].get("augment", True),
            augmentation_cfg=cfg.get("augmentation", {}),
        )
        self.val_dataset = TalcSegmentationDataset(
            cfg["dataset_csv"],
            subset="val",
            image_size=cfg["data"]["image_size"],
            augment=False,
        )
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
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=cfg["trainer"].get("amp", True) and self.device.type == "cuda"
        )

    def fit(self) -> dict[str, Any]:
        with MlflowRun(self.cfg.get("mlflow"), run_name=self.cfg.get("run_name", "talc_segmenter")) as mlrun:
            mlrun.log_params_flat(self.cfg)
            best_dice = -1.0
            stale_epochs = 0
            history: list[dict[str, float]] = []
            epochs = range(1, self.cfg["trainer"]["epochs"] + 1)
            if tqdm is not None:
                epochs = tqdm(epochs, desc="talc epochs", unit="epoch")
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
                mlrun.log_metrics({f"train_{k}": v for k, v in train_metrics.items()}, step=epoch)
                mlrun.log_metrics({f"val_{k}": v for k, v in val_metrics.items()}, step=epoch)
                if val_metrics["dice"] > best_dice:
                    best_dice = val_metrics["dice"]
                    stale_epochs = 0
                    self._save_checkpoint("best.pt", epoch, val_metrics)
                    (self.run_dir / "best_metrics.json").write_text(
                        json.dumps(val_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
                    )
                else:
                    stale_epochs += 1
                if epoch % self.cfg["trainer"].get("save_every", 1) == 0:
                    self._save_checkpoint(f"epoch_{epoch:03d}.pt", epoch, val_metrics)
                patience = self.cfg["trainer"].get("early_stopping_patience", 0)
                if patience and stale_epochs >= patience:
                    break
            summary = {"best_val_dice": best_dice, "epochs": len(history)}
            (self.run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
            mlrun.log_metrics(summary)
            mlrun.log_artifact(self.run_dir / "history.csv")
            mlrun.log_artifact(self.run_dir / "best_metrics.json")
            mlrun.log_artifact(self.run_dir / "summary.json")
            return summary

    def _run_epoch(self, epoch: int, train: bool) -> dict[str, float]:
        loader = self.train_loader if train else self.val_loader
        self.model.train(train)
        phase = "train" if train else "val"
        iterator = tqdm(loader, desc=f"{phase} talc {epoch}", unit="batch", leave=False) if tqdm is not None else loader
        total_loss = 0.0
        total_items = 0
        sums = {"dice": 0.0, "f1": 0.0, "iou": 0.0, "fraction_mae": 0.0}
        auc_probs = []
        auc_targets = []
        threshold = self.cfg["trainer"].get("threshold", 0.5)
        for batch in iterator:
            images = batch["image"].to(self.device)
            masks = batch["mask"].to(self.device)
            with torch.set_grad_enabled(train):
                with torch.cuda.amp.autocast(enabled=self.scaler.is_enabled()):
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
                probs = torch.sigmoid(logits.detach()).float().cpu().numpy().ravel()
                targets = masks.detach().float().cpu().numpy().ravel()
                auc_probs.append(probs)
                auc_targets.append(targets)
            if tqdm is not None:
                iterator.set_postfix(loss=total_loss / max(total_items, 1), dice=sums["dice"] / max(total_items, 1))
        result = {
            "loss": total_loss / max(total_items, 1),
            "dice": sums["dice"] / max(total_items, 1),
            "f1": sums["f1"] / max(total_items, 1),
            "iou": sums["iou"] / max(total_items, 1),
            "fraction_mae": sums["fraction_mae"] / max(total_items, 1),
        }
        result["roc_auc"] = self._roc_auc(auc_targets, auc_probs) if auc_probs else float("nan")
        return result

    @staticmethod
    def _batch_metrics(logits, masks, threshold: float) -> dict[str, float]:
        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).float()
        dims = (1, 2, 3)
        intersection = (preds * masks).sum(dims)
        pred_sum = preds.sum(dims)
        target_sum = masks.sum(dims)
        union = pred_sum + target_sum - intersection
        dice = ((2 * intersection + 1e-7) / (pred_sum + target_sum + 1e-7)).mean()
        iou = ((intersection + 1e-7) / (union + 1e-7)).mean()
        fraction_mae = (preds.mean(dims) - masks.mean(dims)).abs().mean()
        return {
            "dice": float(dice.detach().cpu()),
            "f1": float(dice.detach().cpu()),
            "iou": float(iou.detach().cpu()),
            "fraction_mae": float(fraction_mae.detach().cpu()),
        }

    @staticmethod
    def _roc_auc(target_chunks: list, prob_chunks: list) -> float:
        targets = np.concatenate(target_chunks)
        probs = np.concatenate(prob_chunks)
        if len(np.unique(targets)) < 2:
            return float("nan")
        return float(roc_auc_score(targets, probs))

    def _save_checkpoint(self, name: str, epoch: int, metrics: dict[str, float]) -> None:
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
        with (self.run_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)

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
