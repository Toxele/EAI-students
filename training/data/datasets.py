from __future__ import annotations

import csv
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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
        augmentation_cfg: dict[str, Any] | None = None,
    ) -> None:
        if torch is None or T is None:
            raise ImportError("OreClassificationDataset requires torch and torchvision.")
        self.classes = classes
        self.class_to_idx = {label: idx for idx, label in enumerate(classes)}
        self.mask_channel = mask_channel or {}
        self.augmentation_cfg = augmentation_cfg or {}
        self.project_root = self._infer_project_root(manifest_csv)
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
        with Image.open(self._resolve_path(row["path"])) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        if self.mask_channel.get("enabled"):
            mask_tensor = self._load_mask_tensor(row)
            tensor = torch.cat([tensor, mask_tensor], dim=0)
        label_idx = self.class_to_idx[row["label"]]
        return {
            "image": tensor,
            "label": torch.tensor(label_idx, dtype=torch.long),
            "path": str(self._resolve_path(row["path"])),
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
        train = subset == "train"
        resize_mode = self.augmentation_cfg.get("resize_mode", "resize")
        transforms = []
        if train and resize_mode == "random_resized_crop":
            transforms.append(
                T.RandomResizedCrop(
                    (self.image_size, self.image_size),
                    scale=tuple(self.augmentation_cfg.get("crop_scale", [0.72, 1.0])),
                    ratio=tuple(self.augmentation_cfg.get("crop_ratio", [0.85, 1.18])),
                )
            )
        else:
            transforms.append(T.Resize((self.image_size, self.image_size)))
        if train and not self.mask_channel.get("enabled"):
            transforms.extend(
                [
                    T.RandomHorizontalFlip(p=self.augmentation_cfg.get("hflip_p", 0.5)),
                    T.RandomVerticalFlip(p=self.augmentation_cfg.get("vflip_p", 0.2)),
                ]
            )
        if train:
            transforms.append(T.Lambda(lambda image: self._augment_image(image)))
        transforms.extend(
            [
                T.ToTensor(),
                T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        if self.augmentation_cfg:
            return T.Compose(transforms)
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

    def _augment_image(self, image: Image.Image) -> Image.Image:
        cfg = self.augmentation_cfg
        if not cfg:
            return image
        if cfg.get("brightness", 0.0):
            image = TF.adjust_brightness(image, random.uniform(1 - cfg["brightness"], 1 + cfg["brightness"]))
        if cfg.get("contrast", 0.0):
            image = TF.adjust_contrast(image, random.uniform(1 - cfg["contrast"], 1 + cfg["contrast"]))
        if cfg.get("saturation", 0.0):
            image = TF.adjust_saturation(image, random.uniform(1 - cfg["saturation"], 1 + cfg["saturation"]))
        if cfg.get("hue", 0.0):
            image = TF.adjust_hue(image, random.uniform(-cfg["hue"], cfg["hue"]))
        if cfg.get("gamma", 0.0):
            image = TF.adjust_gamma(image, random.uniform(1 - cfg["gamma"], 1 + cfg["gamma"]))
        if random.random() < cfg.get("gray_domain_p", 0.0):
            image = self._apply_gray_domain_style(image, cfg)
        if random.random() < cfg.get("domain_strong_p", 0.0):
            image = self._apply_domain_strong_style(image, cfg)
        if random.random() < cfg.get("blur_p", 0.0):
            image = self._apply_blur(image, cfg)
        if random.random() < cfg.get("noise_p", 0.0):
            image = self._apply_sensor_noise(image, cfg)
        if random.random() < cfg.get("jpeg_p", 0.0):
            image = self._apply_jpeg_roundtrip(image, cfg)
        return image

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

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        if candidate.exists():
            return candidate.resolve()
        return self.project_root / candidate

    @staticmethod
    def _infer_project_root(manifest_csv: str | Path) -> Path:
        manifest = Path(manifest_csv)
        if manifest.is_absolute():
            markers = [
                Path("configs") / "classifier" / "nornikel_classifier.json",
                Path("models") / "classifiers.py",
            ]
            for candidate in [manifest.parent, *manifest.parents]:
                if all((candidate / marker).exists() for marker in markers):
                    return candidate.resolve()
            parts = manifest.parts
            if "artifacts" in parts:
                return Path(*parts[: parts.index("artifacts")]).resolve()
            return manifest.parent.resolve()
        cwd = Path.cwd().resolve()
        markers = [
            Path("configs") / "classifier" / "nornikel_classifier.json",
            Path("models") / "classifiers.py",
        ]
        for candidate in [cwd, *cwd.parents]:
            if all((candidate / marker).exists() for marker in markers):
                return candidate.resolve()
        return cwd

    @staticmethod
    def _apply_gray_domain_style(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        image = TF.adjust_saturation(
            image,
            random.uniform(cfg.get("gray_saturation_min", 0.04), cfg.get("gray_saturation_max", 0.45)),
        )
        image = TF.adjust_brightness(
            image,
            random.uniform(cfg.get("gray_brightness_min", 0.40), cfg.get("gray_brightness_max", 0.95)),
        )
        image = TF.adjust_contrast(
            image,
            random.uniform(cfg.get("gray_contrast_min", 0.55), cfg.get("gray_contrast_max", 1.15)),
        )
        rgb = np.asarray(image).astype(np.float32)
        rgb[:, :, 2] *= random.uniform(1.0, cfg.get("gray_blue_gain_max", 1.16))
        rgb[:, :, 0] *= random.uniform(cfg.get("gray_red_gain_min", 0.84), 1.0)
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    @staticmethod
    def _apply_domain_strong_style(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        rgb = np.asarray(image).astype(np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        if random.random() < cfg.get("clahe_p", 0.55):
            clip = random.uniform(cfg.get("clahe_clip_min", 1.2), cfg.get("clahe_clip_max", 3.0))
            grid = int(random.choice(cfg.get("clahe_grid_sizes", [6, 8, 10, 12])))
            l_chan = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid)).apply(l_chan)
        l_chan = np.clip(l_chan.astype(np.float32) * random.uniform(0.68, 1.22) + random.uniform(-16, 16), 0, 255)
        chroma_scale = random.uniform(cfg.get("domain_chroma_min", 0.04), cfg.get("domain_chroma_max", 0.75))
        a_chan = 128.0 + (a_chan.astype(np.float32) - 128.0) * chroma_scale + random.uniform(-7, 7)
        b_chan = 128.0 + (b_chan.astype(np.float32) - 128.0) * chroma_scale + random.uniform(-10, 8)
        styled = cv2.merge(
            [
                np.clip(l_chan, 0, 255).astype(np.uint8),
                np.clip(a_chan, 0, 255).astype(np.uint8),
                np.clip(b_chan, 0, 255).astype(np.uint8),
            ]
        )
        rgb = cv2.cvtColor(styled, cv2.COLOR_LAB2RGB).astype(np.float32)
        gains = np.array(
            [
                random.uniform(cfg.get("red_gain_min", 0.82), cfg.get("red_gain_max", 1.12)),
                random.uniform(cfg.get("green_gain_min", 0.88), cfg.get("green_gain_max", 1.12)),
                random.uniform(cfg.get("blue_gain_min", 0.88), cfg.get("blue_gain_max", 1.20)),
            ],
            dtype=np.float32,
        )
        rgb *= gains.reshape(1, 1, 3)
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    @staticmethod
    def _apply_blur(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        rgb = np.asarray(image)
        ksize = int(random.choice(cfg.get("blur_kernel_sizes", [3, 5])))
        blurred = cv2.GaussianBlur(rgb, (ksize, ksize), random.uniform(0.2, 1.1))
        return Image.fromarray(blurred, mode="RGB")

    @staticmethod
    def _apply_sensor_noise(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        rgb = np.asarray(image).astype(np.float32)
        sigma = random.uniform(cfg.get("noise_sigma_min", 1.0), cfg.get("noise_sigma_max", 7.0))
        rgb += np.random.normal(0.0, sigma, size=rgb.shape).astype(np.float32)
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    @staticmethod
    def _apply_jpeg_roundtrip(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        quality = int(random.uniform(cfg.get("jpeg_quality_min", 58), cfg.get("jpeg_quality_max", 94)))
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), mode="RGB")
