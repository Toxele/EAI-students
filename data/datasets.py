from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageOps

try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as T
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object
    T = None


class OreClassificationDataset(Dataset):
    def __init__(
        self,
        manifest_csv: str | Path,
        classes: list[str],
        subset: str | None = None,
        image_size: int = 384,
        include_sources: list[str] | None = None,
        exclude_conflicts: bool = True,
        mask_channel: dict[str, Any] | None = None,
    ) -> None:
        if torch is None or T is None:
            raise ImportError("OreClassificationDataset requires torch and torchvision.")
        self.classes = classes
        self.class_to_idx = {label: idx for idx, label in enumerate(classes)}
        self.mask_channel = mask_channel or {}
        self.mask_by_rel_path = self._load_mask_map(self.mask_channel.get("report_csv")) if self.mask_channel.get("enabled") else {}
        self.rows = self._load_rows(
            manifest_csv=manifest_csv,
            subset=subset,
            include_sources=include_sources,
            exclude_conflicts=exclude_conflicts,
        )
        self.image_size = image_size
        self.transform = self._build_transform(subset)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        if self.mask_channel.get("enabled"):
            mask_tensor = self._load_mask_tensor(row)
            tensor = torch.cat([tensor, mask_tensor], dim=0)
        label_idx = self.class_to_idx[row["label"]]
        return {
            "image": tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "path": row["path"],
            "label_name": row["label"],
        }

    def labels(self) -> list[int]:
        return [self.class_to_idx[row["label"]] for row in self.rows]

    def _load_rows(
        self,
        manifest_csv: str | Path,
        subset: str | None,
        include_sources: list[str] | None,
        exclude_conflicts: bool,
    ) -> list[dict[str, str]]:
        include_sources = include_sources or []
        rows: list[dict[str, str]] = []
        with Path(manifest_csv).open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if subset and row.get("subset") != subset:
                    continue
                if include_sources and row.get("source") not in include_sources:
                    continue
                if exclude_conflicts and str(row.get("label_conflict", "")).lower() == "true":
                    continue
                if row.get("label") not in self.class_to_idx:
                    continue
                rows.append(row)
        return rows

    def _build_transform(self, subset: str | None):
        if self.mask_channel.get("enabled"):
            transforms = [T.Resize((self.image_size, self.image_size))]
            if subset == "train":
                transforms.append(T.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.1, hue=0.02))
            transforms.extend(
                [
                    T.ToTensor(),
                    T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ]
            )
            return T.Compose(transforms)
        return T.Compose(
            [
                T.Resize((self.image_size, self.image_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.1, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
            if subset == "train"
            else [
                T.Resize((self.image_size, self.image_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    @staticmethod
    def _load_mask_map(report_csv: str | Path | None) -> dict[str, str]:
        if not report_csv or not Path(report_csv).exists():
            return {}
        with Path(report_csv).open("r", newline="", encoding="utf-8") as f:
            return {row["rel_path"].replace("\\", "/"): row["mask_path"] for row in csv.DictReader(f) if row.get("mask_path")}

    def _load_mask_tensor(self, row: dict[str, str]):
        rel_path = row["rel_path"].replace("\\", "/")
        mask_path = self.mask_by_rel_path.get(rel_path)
        if mask_path and Path(mask_path).exists():
            mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        else:
            mask = np.zeros((int(row["height"]), int(row["width"])), dtype=np.uint8)
        mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        return torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
