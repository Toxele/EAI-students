"""
PyTorch Dataset для multi-label классификации (coarse / fine и др.).
"""
from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

try:
    import torch
    from torch.utils.data import Dataset
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object
    T = None
    TF = None

from data.datasets import OreClassificationDataset


class MultiLabelOreDataset(Dataset):
    """Читает manifest с колонками tag_* для выбранных тегов."""

    TAG_COLUMN_PREFIX = "tag_"

    def __init__(
        self,
        manifest_csv: str | Path,
        subset: str,
        tags: list[str] | None = None,
        image_size: int = 384,
        augmentation: bool = False,
        augmentation_cfg: dict[str, Any] | None = None,
    ) -> None:
        if torch is None or T is None:
            raise ImportError("MultiLabelOreDataset requires torch and torchvision.")
        self.tags = list(tags or ["talc", "coarse", "fine"])
        self.tag_columns = tuple(f"{self.TAG_COLUMN_PREFIX}{name}" for name in self.tags)
        self.image_size = image_size
        self.augmentation_cfg = augmentation_cfg or {}
        self.project_root = self._infer_project_root(manifest_csv)
        self.rows = self._load_rows(manifest_csv, subset)
        self.transform = self._build_transform(augmentation)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        with Image.open(self._resolve_path(row["path"])) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        labels = torch.tensor(
            [float(row[col]) for col in self.tag_columns],
            dtype=torch.float32,
        )
        return {
            "image": tensor,
            "labels": labels,
            "path": str(self._resolve_path(row["path"])),
            "tags": row.get("tags", ""),
            "md5": row["md5"],
            "ig_label": row.get("ig_label", ""),
        }

    def label_vectors(self) -> list[list[int]]:
        """Бинарные метки для WeightedRandomSampler."""
        return [[int(row[col]) for col in self.tag_columns] for row in self.rows]

    def ig_labels(self) -> list[str]:
        """coarse/fine для стратифицированного сэмплера."""
        return [row.get("ig_label", "") for row in self.rows]

    def _load_rows(self, manifest_csv: str | Path, subset: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        with Path(manifest_csv).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("subset") != subset:
                    continue
                rows.append(row)
        return rows

    def _build_transform(self, augmentation: bool):
        cfg = self.augmentation_cfg
        resize_mode = cfg.get("resize_mode", "resize")
        if augmentation and resize_mode == "random_resized_crop":
            transforms: list[Any] = [
                T.RandomResizedCrop(
                    (self.image_size, self.image_size),
                    scale=tuple(cfg.get("crop_scale", [0.72, 1.0])),
                    ratio=tuple(cfg.get("crop_ratio", [0.85, 1.18])),
                )
            ]
        else:
            transforms = [T.Resize((self.image_size, self.image_size))]

        if augmentation:
            transforms.extend(
                [
                    T.RandomHorizontalFlip(p=cfg.get("hflip_p", 0.5)),
                    T.RandomVerticalFlip(p=cfg.get("vflip_p", 0.25)),
                ]
            )
        if augmentation and cfg:
            transforms.append(T.Lambda(lambda image: self._augment_image(image)))
        elif augmentation:
            transforms.append(
                T.ColorJitter(brightness=0.15, contrast=0.2, saturation=0.1, hue=0.02)
            )

        transforms.extend(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        return T.Compose(transforms)

    def _augment_image(self, image: Image.Image) -> Image.Image:
        """Domain augmentation (переиспользует OreClassificationDataset)."""
        cfg = self.augmentation_cfg
        if not cfg:
            return image
        if cfg.get("brightness", 0.0):
            image = TF.adjust_brightness(
                image, random.uniform(1 - cfg["brightness"], 1 + cfg["brightness"])
            )
        if cfg.get("contrast", 0.0):
            image = TF.adjust_contrast(
                image, random.uniform(1 - cfg["contrast"], 1 + cfg["contrast"])
            )
        if cfg.get("saturation", 0.0):
            image = TF.adjust_saturation(
                image, random.uniform(1 - cfg["saturation"], 1 + cfg["saturation"])
            )
        if cfg.get("hue", 0.0):
            image = TF.adjust_hue(image, random.uniform(-cfg["hue"], cfg["hue"]))
        if cfg.get("gamma", 0.0):
            image = TF.adjust_gamma(
                image, random.uniform(1 - cfg["gamma"], 1 + cfg["gamma"])
            )
        if random.random() < cfg.get("gray_domain_p", 0.0):
            image = OreClassificationDataset._apply_gray_domain_style(image, cfg)
        if random.random() < cfg.get("domain_strong_p", 0.0):
            image = OreClassificationDataset._apply_domain_strong_style(image, cfg)
        if random.random() < cfg.get("blur_p", 0.0):
            image = OreClassificationDataset._apply_blur(image, cfg)
        if random.random() < cfg.get("noise_p", 0.0):
            image = OreClassificationDataset._apply_sensor_noise(image, cfg)
        if random.random() < cfg.get("jpeg_p", 0.0):
            image = OreClassificationDataset._apply_jpeg_roundtrip(image, cfg)
        return image

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute() and candidate.exists():
            return candidate
        for base in (self.project_root, Path.cwd()):
            resolved = base / candidate
            if resolved.exists():
                return resolved.resolve()
        return (self.project_root / candidate).resolve()

    @staticmethod
    def _infer_project_root(manifest_csv: str | Path) -> Path:
        manifest = Path(manifest_csv).resolve()
        for candidate in [manifest.parent, *manifest.parents]:
            if (candidate / "data").is_dir() and (candidate / "models" / "classifiers.py").exists():
                return candidate
        return Path.cwd().resolve()
