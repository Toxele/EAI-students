from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

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
    ) -> None:
        if torch is None or T is None:
            raise ImportError("OreClassificationDataset requires torch and torchvision.")
        self.classes = classes
        self.class_to_idx = {label: idx for idx, label in enumerate(classes)}
        self.rows = self._load_rows(
            manifest_csv=manifest_csv,
            subset=subset,
            include_sources=include_sources,
            exclude_conflicts=exclude_conflicts,
        )
        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.1, hue=0.02),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
            if subset == "train"
            else [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        with Image.open(row["path"]) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
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

