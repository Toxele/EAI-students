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
    from torchvision.transforms import InterpolationMode
    import torchvision.transforms.functional as TF
except ImportError:  # pragma: no cover
    torch = None
    Dataset = object


class TalcSegmentationDataset(Dataset):
    def __init__(
        self,
        dataset_csv: str | Path,
        subset: str,
        image_size: int = 384,
        augment: bool = False,
        augmentation_cfg: dict[str, Any] | None = None,
    ) -> None:
        if torch is None:
            raise ImportError("TalcSegmentationDataset requires torch and torchvision.")
        self.rows = self._load_rows(dataset_csv, subset)
        self.image_size = image_size
        self.augment = augment
        self.augmentation_cfg = augmentation_cfg or {}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.rows[idx]
        with Image.open(row["image_path"]) as img:
            image = ImageOps.exif_transpose(img).convert("RGB")
        mask = cv2.imdecode(np.fromfile(row["mask_path"], dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {row['mask_path']}")
        mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255, mode="L")
        if self.augment:
            image, mask_image = self._augment_pair(image, mask_image)
        image = TF.resize(image, [self.image_size, self.image_size])
        image_tensor = TF.to_tensor(image)
        image_tensor = TF.normalize(image_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        mask = np.array(TF.resize(mask_image, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST))
        mask_tensor = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "image_path": row["image_path"],
            "mask_path": row["mask_path"],
            "sample_type": row["sample_type"],
        }

    @staticmethod
    def _load_rows(dataset_csv: str | Path, subset: str) -> list[dict[str, str]]:
        with Path(dataset_csv).open("r", newline="", encoding="utf-8") as f:
            return [row for row in csv.DictReader(f) if row.get("subset") == subset]

    def _augment_pair(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        cfg = self.augmentation_cfg
        if random.random() < cfg.get("hflip_p", 0.5):
            image = TF.hflip(image)
            mask = TF.hflip(mask)
        if random.random() < cfg.get("vflip_p", 0.2):
            image = TF.vflip(image)
            mask = TF.vflip(mask)
        max_rotate = cfg.get("max_rotate_degrees", 12)
        if max_rotate:
            angle = random.uniform(-max_rotate, max_rotate)
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
        brightness = cfg.get("brightness", 0.15)
        contrast = cfg.get("contrast", 0.15)
        saturation = cfg.get("saturation", 0.12)
        gamma = cfg.get("gamma", 0.08)
        if brightness:
            image = TF.adjust_brightness(image, random.uniform(1 - brightness, 1 + brightness))
        if contrast:
            image = TF.adjust_contrast(image, random.uniform(1 - contrast, 1 + contrast))
        if saturation:
            image = TF.adjust_saturation(image, random.uniform(1 - saturation, 1 + saturation))
        if gamma:
            image = TF.adjust_gamma(image, random.uniform(1 - gamma, 1 + gamma))
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
        return image, mask

    @staticmethod
    def _apply_gray_domain_style(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        image = TF.adjust_saturation(
            image,
            random.uniform(
                cfg.get("gray_saturation_min", 0.08),
                cfg.get("gray_saturation_max", 0.45),
            ),
        )
        image = TF.adjust_brightness(
            image,
            random.uniform(
                cfg.get("gray_brightness_min", 0.45),
                cfg.get("gray_brightness_max", 0.85),
            ),
        )
        image = TF.adjust_contrast(
            image,
            random.uniform(
                cfg.get("gray_contrast_min", 0.55),
                cfg.get("gray_contrast_max", 0.95),
            ),
        )
        rgb = np.asarray(image).astype(np.float32)
        rgb[:, :, 2] *= random.uniform(1.0, cfg.get("gray_blue_gain_max", 1.12))
        rgb[:, :, 0] *= random.uniform(cfg.get("gray_red_gain_min", 0.88), 1.0)
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    @staticmethod
    def _apply_domain_strong_style(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        rgb = np.asarray(image).astype(np.uint8)
        lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        if random.random() < cfg.get("clahe_p", 0.55):
            clip = random.uniform(cfg.get("clahe_clip_min", 1.2), cfg.get("clahe_clip_max", 2.8))
            grid = int(random.choice(cfg.get("clahe_grid_sizes", [6, 8, 10, 12])))
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
            l_chan = clahe.apply(l_chan)
        l_chan = np.clip(l_chan.astype(np.float32) * random.uniform(0.65, 1.25) + random.uniform(-18, 18), 0, 255)
        chroma_scale = random.uniform(cfg.get("domain_chroma_min", 0.05), cfg.get("domain_chroma_max", 0.75))
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
        if random.random() < cfg.get("vignette_p", 0.35):
            rgb = TalcSegmentationDataset._apply_vignette(rgb, cfg)
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
    def _apply_vignette(rgb: np.ndarray, cfg: dict[str, Any]) -> np.ndarray:
        h, w = rgb.shape[:2]
        y, x = np.ogrid[:h, :w]
        cy = h * random.uniform(0.42, 0.58)
        cx = w * random.uniform(0.42, 0.58)
        dist = np.sqrt(((x - cx) / max(w, 1)) ** 2 + ((y - cy) / max(h, 1)) ** 2)
        strength = random.uniform(cfg.get("vignette_strength_min", 0.08), cfg.get("vignette_strength_max", 0.35))
        mask = 1.0 - strength * (dist / max(float(dist.max()), 1e-6)) ** 1.4
        return rgb * mask[:, :, None]

    @staticmethod
    def _apply_blur(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        rgb = np.asarray(image)
        if random.random() < 0.5:
            ksize = int(random.choice(cfg.get("blur_kernel_sizes", [3, 5])))
            blurred = cv2.GaussianBlur(rgb, (ksize, ksize), random.uniform(0.2, 1.1))
        else:
            blurred = cv2.medianBlur(rgb, int(random.choice(cfg.get("median_kernel_sizes", [3]))))
        return Image.fromarray(blurred, mode="RGB")

    @staticmethod
    def _apply_sensor_noise(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        rgb = np.asarray(image).astype(np.float32)
        sigma = random.uniform(cfg.get("noise_sigma_min", 1.5), cfg.get("noise_sigma_max", 8.0))
        rgb += np.random.normal(0.0, sigma, size=rgb.shape).astype(np.float32)
        if random.random() < cfg.get("speckle_p", 0.25):
            rgb *= 1.0 + np.random.normal(0.0, random.uniform(0.01, 0.04), size=rgb.shape).astype(np.float32)
        return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")

    @staticmethod
    def _apply_jpeg_roundtrip(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
        quality = int(random.uniform(cfg.get("jpeg_quality_min", 55), cfg.get("jpeg_quality_max", 92)))
        ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return image
        decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        return Image.fromarray(cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB), mode="RGB")
