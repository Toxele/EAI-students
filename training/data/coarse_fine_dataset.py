"""
PyTorch Dataset для soft binary coarse vs fine (target_coarse in [0, 0.5, 1]).
"""
from __future__ import annotations

import csv
import io
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
    TF = None

DEFAULT_AUGMENTATION: dict[str, Any] = {
    "resize_mode": "random_resized_crop",
    "crop_scale": [0.72, 1.0],
    "crop_ratio": [0.85, 1.18],
    "hflip_p": 0.5,
    "vflip_p": 0.25,
    "rotation_deg": 15,
    "brightness": 0.22,
    "contrast": 0.26,
    "saturation": 0.22,
    "hue": 0.025,
    "gamma": 0.15,
    "gray_domain_p": 0.45,
    "gray_saturation_min": 0.04,
    "gray_saturation_max": 0.42,
    "gray_brightness_min": 0.40,
    "gray_brightness_max": 0.98,
    "gray_contrast_min": 0.55,
    "gray_contrast_max": 1.18,
    "gray_blue_gain_max": 1.18,
    "gray_red_gain_min": 0.82,
    "domain_strong_p": 0.30,
    "domain_chroma_min": 0.04,
    "domain_chroma_max": 0.78,
    "clahe_p": 0.55,
    "clahe_clip_min": 1.2,
    "clahe_clip_max": 3.0,
    "clahe_grid_sizes": [6, 8, 10, 12],
    "red_gain_min": 0.82,
    "red_gain_max": 1.12,
    "green_gain_min": 0.88,
    "green_gain_max": 1.12,
    "blue_gain_min": 0.88,
    "blue_gain_max": 1.20,
    "blur_p": 0.14,
    "blur_kernel_sizes": [3, 5],
    "noise_p": 0.22,
    "noise_sigma_min": 1.0,
    "noise_sigma_max": 7.0,
    "jpeg_p": 0.16,
    "jpeg_quality_min": 58,
    "jpeg_quality_max": 94,
    "cutout_p": 0.12,
    "cutout_scale": [0.02, 0.08],
}


class CoarseFineDataset(Dataset):
    """Manifest: target_coarse, label_bucket, subset."""

    def __init__(
        self,
        manifest_csv: str | Path,
        subset: str,
        image_size: int = 384,
        augmentation: bool = False,
        augmentation_cfg: dict[str, Any] | None = None,
    ) -> None:
        if torch is None or T is None:
            raise ImportError("CoarseFineDataset requires torch and torchvision.")
        self.image_size = image_size
        self.augmentation = augmentation
        self.augmentation_cfg = augmentation_cfg or DEFAULT_AUGMENTATION
        self.project_root = self._infer_project_root(manifest_csv)
        self.rows = self._load_rows(manifest_csv, subset)
        self.normalize = T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        with Image.open(self._resolve_path(row["path"])) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self._transform(image)
        target = torch.tensor(float(row["target_coarse"]), dtype=torch.float32)
        return {
            "image": tensor,
            "target": target,
            "path": str(self._resolve_path(row["path"])),
            "label_bucket": row["label_bucket"],
            "md5": row["md5"],
            "has_talc": row["tag_talc"] == "1",
        }

    def sampler_weights(self, ambiguous_weight: float = 0.12) -> list[float]:
        """Баланс coarse/fine; ambiguous — редко."""
        bucket_counts: dict[str, int] = {}
        for row in self.rows:
            bucket = row["label_bucket"]
            if bucket == "ambiguous":
                continue
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        max_count = max(bucket_counts.values()) if bucket_counts else 1
        weights: list[float] = []
        for row in self.rows:
            bucket = row["label_bucket"]
            if bucket == "ambiguous":
                weights.append(ambiguous_weight)
            else:
                weights.append(max_count / max(bucket_counts.get(bucket, 1), 1))
        return weights

    def _transform(self, image: Image.Image) -> torch.Tensor:
        cfg = self.augmentation_cfg
        if self.augmentation and cfg.get("resize_mode") == "random_resized_crop":
            image = T.RandomResizedCrop(
                (self.image_size, self.image_size),
                scale=tuple(cfg.get("crop_scale", [0.72, 1.0])),
                ratio=tuple(cfg.get("crop_ratio", [0.85, 1.18])),
            )(image)
        else:
            image = T.Resize((self.image_size, self.image_size))(image)

        if self.augmentation:
            if random.random() < cfg.get("hflip_p", 0.5):
                image = TF.hflip(image)
            if random.random() < cfg.get("vflip_p", 0.25):
                image = TF.vflip(image)
            rot = cfg.get("rotation_deg", 0)
            if rot and random.random() < 0.5:
                image = TF.rotate(image, random.uniform(-rot, rot))
            image = self._augment_image(image)
            if random.random() < cfg.get("cutout_p", 0.0):
                image = self._random_cutout(image, cfg)

        tensor = T.ToTensor()(image)
        return self.normalize(tensor)

    def _augment_image(self, image: Image.Image) -> Image.Image:
        cfg = self.augmentation_cfg
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
    def _random_cutout(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        rgb = np.asarray(image).copy()
        h, w = rgb.shape[:2]
        scale = cfg.get("cutout_scale", [0.02, 0.08])
        area = h * w * random.uniform(scale[0], scale[1])
        cut_h = int(max(1, min(h, area**0.5)))
        cut_w = int(max(1, min(w, area / cut_h)))
        y0 = random.randint(0, max(0, h - cut_h))
        x0 = random.randint(0, max(0, w - cut_w))
        rgb[y0 : y0 + cut_h, x0 : x0 + cut_w] = int(random.uniform(0, 40))
        return Image.fromarray(rgb, mode="RGB")

    def _load_rows(self, manifest_csv: str | Path, subset: str) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        with Path(manifest_csv).open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("subset") != subset:
                    continue
                rows.append(row)
        return rows

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
        buf = io.BytesIO()
        image.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).convert("RGB")
