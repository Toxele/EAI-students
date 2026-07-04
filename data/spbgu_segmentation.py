from __future__ import annotations

import csv
import random
from dataclasses import dataclass
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


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
PROJECT_MARKERS = [Path("configs") / "segmentation" / "spbgu_unet.json", Path("models") / "segmentation.py"]


@dataclass(frozen=True)
class NtMdtAsciiImage:
    """Container for an NT-MDT ASCII height map and its physical metadata."""

    data: np.ndarray
    nx: int
    ny: int
    scale_x: float | None
    scale_y: float | None
    unit_x: str | None
    unit_y: str | None
    unit_data: str | None


def load_ntmdt_ascii(path: str | Path) -> NtMdtAsciiImage:
    """Read an NT-MDT ASCII `.txt` file into a numeric 2D height map."""
    path = Path(path)
    header: dict[str, str] = {}
    values: list[float] = []
    in_data = False
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            if stripped.lower().startswith("start of data"):
                in_data = True
                continue
            if not in_data:
                if "=" in stripped:
                    key, value = stripped.split("=", 1)
                    header[key.strip().lower()] = value.strip()
                continue
            values.extend(float(x) for x in stripped.split())
    nx = int(float(header.get("nx", "0")))
    ny = int(float(header.get("ny", "0")))
    if nx <= 0 or ny <= 0:
        raise ValueError(f"Could not read NX/NY from {path}")
    data = np.asarray(values, dtype=np.float32)
    if data.size < nx * ny:
        raise ValueError(f"Expected {nx * ny} values in {path}, got {data.size}")
    data = data[: nx * ny].reshape(ny, nx)
    return NtMdtAsciiImage(
        data=data,
        nx=nx,
        ny=ny,
        scale_x=_optional_float(header.get("scale x")),
        scale_y=_optional_float(header.get("scale y")),
        unit_x=header.get("unit x"),
        unit_y=header.get("unit y"),
        unit_data=header.get("unit data"),
    )


def normalize_height_map(data: np.ndarray, lower: float = 1.0, upper: float = 99.0) -> np.ndarray:
    """Convert a height map to a robust 8-bit image using percentile clipping."""
    lo, hi = np.percentile(data[np.isfinite(data)], [lower, upper])
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(data, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def read_spbgu_image(path: str | Path, input_kind: str = "txt") -> Image.Image:
    """Read an SPbGU source image as RGB, supporting NT-MDT txt and regular images."""
    path = Path(path)
    if input_kind == "txt":
        gray = normalize_height_map(load_ntmdt_ascii(path).data)
        return Image.fromarray(gray, mode="L").convert("RGB")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width > image.height:
        image = image.crop((0, 0, image.height, image.height))
    return image


def read_spbgu_binary_mask(path: str | Path, mask_mode: str = "foreground") -> Image.Image:
    """Read a color instance BMP as a binary semantic mask."""
    image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read mask: {path}")
    if image.ndim == 3:
        foreground = np.any(image[:, :, :3] > 0, axis=2)
    else:
        foreground = image > 0
    if mask_mode == "foreground":
        mask = foreground
    elif mask_mode in {"contour", "contours", "background"}:
        mask = ~foreground
    else:
        raise ValueError(f"Unsupported SPbGU mask_mode: {mask_mode}")
    return Image.fromarray(mask.astype(np.uint8) * 255, mode="L")


class SpbguSegmentationManifestBuilder:
    """Build a segmentation manifest from SPbGU source files and BMP masks."""

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Store manifest builder settings from JSON config."""
        self.project_root = find_project_root()
        self.root = resolve_project_path(cfg.get("root", "spbgu_data"), self.project_root)
        self.output_csv = resolve_project_path(
            cfg.get("output_csv", "artifacts/manifests/spbgu_segmentation_manifest.csv"),
            self.project_root,
        )
        self.input_kind = cfg.get("input_kind", "txt")
        self.val_fraction = float(cfg.get("val_fraction", 0.25))
        self.seed = int(cfg.get("seed", 42))

    def build(self) -> list[dict[str, str]]:
        """Collect matching source/mask pairs and write a manifest CSV."""
        rows = self._collect_rows()
        rows = self._assign_holdout(rows)
        self.output_csv.parent.mkdir(parents=True, exist_ok=True)
        with self.output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        return rows

    def _collect_rows(self) -> list[dict[str, str]]:
        """Find every BMP mask and match it to txt or jpg source with the same stem."""
        rows: list[dict[str, str]] = []
        for mask_path in sorted(self.root.rglob("*.bmp")):
            source_path = self._source_for_mask(mask_path)
            if source_path is None:
                continue
            rows.append(
                {
                    "sample_id": mask_path.stem,
                    "source_path": str(source_path),
                    "mask_path": str(mask_path),
                    "input_kind": self.input_kind,
                    "domain_label": self._domain_label(mask_path),
                    "rel_path": str(mask_path.relative_to(self.root)),
                }
            )
        if not rows:
            raise FileNotFoundError(f"No SPbGU source/mask pairs found under {self.root}")
        return rows

    def _source_for_mask(self, mask_path: Path) -> Path | None:
        """Return the source path matching a BMP mask according to input kind."""
        if self.input_kind == "txt":
            candidate = mask_path.with_suffix(".txt")
            return candidate if candidate.exists() else None
        if self.input_kind == "jpg":
            candidate = mask_path.with_suffix(".jpg")
            return candidate if candidate.exists() else None
        raise ValueError(f"Unsupported input_kind: {self.input_kind}")

    @staticmethod
    def _domain_label(path: Path) -> str:
        """Infer a coarse domain label from the parent folder names."""
        parts = [part.lower() for part in path.parts]
        if any("разрыв" in part for part in parts):
            return "rupture"
        if any("поставка" in part for part in parts):
            return "delivery"
        return "unknown"

    def _assign_holdout(self, rows: list[dict[str, str]]) -> list[dict[str, str]]:
        """Assign a deterministic train/val split while preserving domain groups."""
        rng = random.Random(self.seed)
        grouped: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            grouped.setdefault(row["domain_label"], []).append(row)
        result: list[dict[str, str]] = []
        for group_rows in grouped.values():
            shuffled = group_rows[:]
            rng.shuffle(shuffled)
            val_count = max(1, int(round(len(shuffled) * self.val_fraction)))
            for idx, row in enumerate(shuffled):
                row = dict(row)
                row["subset"] = "val" if idx < val_count else "train"
                result.append(row)
        return sorted(result, key=lambda row: row["rel_path"])


class SpbguSegmentationDataset(Dataset):
    """Torch dataset for SPbGU binary foreground segmentation."""

    def __init__(
        self,
        rows: list[dict[str, str]] | str | Path,
        image_size: int = 512,
        augment: bool = False,
        augmentation_cfg: dict[str, Any] | None = None,
        in_channels: int = 3,
        patch_size: int | None = None,
        patch_stride: int | None = None,
        min_patch_foreground: float = 0.0,
        max_patch_foreground: float = 1.0,
        mask_mode: str = "foreground",
    ) -> None:
        """Create a dataset from manifest rows or a CSV path."""
        if torch is None:
            raise ImportError("SpbguSegmentationDataset requires torch and torchvision.")
        self.rows = _load_rows(rows)
        self.samples = self._build_samples(
            self.rows,
            patch_size=patch_size,
            patch_stride=patch_stride,
            min_patch_foreground=min_patch_foreground,
            max_patch_foreground=max_patch_foreground,
            mask_mode=mask_mode,
        )
        self.image_size = image_size
        self.augment = augment
        self.augmentation_cfg = augmentation_cfg or {}
        self.in_channels = in_channels
        self.mask_mode = mask_mode

    def __len__(self) -> int:
        """Return the number of available samples."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        """Return normalized image tensor, binary mask tensor and paths."""
        row = self.samples[idx]
        image = read_spbgu_image(row["source_path"], row.get("input_kind", "txt"))
        mask = read_spbgu_binary_mask(row["mask_path"], self.mask_mode)
        if "patch_x" in row:
            x = int(row["patch_x"])
            y = int(row["patch_y"])
            size = int(row["patch_size"])
            image = image.crop((x, y, x + size, y + size))
            mask = mask.crop((x, y, x + size, y + size))
        if self.augment:
            image, mask = self._augment_pair(image, mask)
        image = TF.resize(image, [self.image_size, self.image_size])
        mask = TF.resize(mask, [self.image_size, self.image_size], interpolation=InterpolationMode.NEAREST)
        image_tensor = TF.to_tensor(image)
        if self.in_channels == 1:
            image_tensor = image_tensor.mean(dim=0, keepdim=True)
            image_tensor = (image_tensor - 0.5) / 0.5
        else:
            image_tensor = TF.normalize(image_tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        mask_tensor = torch.from_numpy((np.asarray(mask) > 0).astype(np.float32)).unsqueeze(0)
        return {
            "image": image_tensor,
            "mask": mask_tensor,
            "source_path": row["source_path"],
            "mask_path": row["mask_path"],
            "sample_id": row.get("sample_id", Path(row["source_path"]).stem),
            "domain_label": row.get("domain_label", "unknown"),
            "patch_x": row.get("patch_x", ""),
            "patch_y": row.get("patch_y", ""),
        }

    @staticmethod
    def _build_samples(
        rows: list[dict[str, str]],
        patch_size: int | None,
        patch_stride: int | None,
        min_patch_foreground: float,
        max_patch_foreground: float,
        mask_mode: str,
    ) -> list[dict[str, str]]:
        """Expand image rows into deterministic patch rows when patching is enabled."""
        if not patch_size:
            return rows
        stride = patch_stride or patch_size
        samples: list[dict[str, str]] = []
        for row in rows:
            mask = read_spbgu_binary_mask(row["mask_path"], mask_mode)
            width, height = mask.size
            x_positions = _grid_positions(width, patch_size, stride)
            y_positions = _grid_positions(height, patch_size, stride)
            mask_array = np.asarray(mask) > 0
            for y in y_positions:
                for x in x_positions:
                    patch = mask_array[y : y + patch_size, x : x + patch_size]
                    foreground = float(patch.mean())
                    if foreground < min_patch_foreground or foreground > max_patch_foreground:
                        continue
                    sample = dict(row)
                    sample["patch_x"] = str(x)
                    sample["patch_y"] = str(y)
                    sample["patch_size"] = str(patch_size)
                    sample["patch_foreground"] = f"{foreground:.6f}"
                    samples.append(sample)
        if not samples:
            raise ValueError("Patch filtering removed every SPbGU sample. Relax foreground thresholds.")
        return samples

    def _augment_pair(self, image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
        """Apply paired geometry transforms and image-only acquisition noise."""
        cfg = self.augmentation_cfg
        if random.random() < cfg.get("hflip_p", 0.5):
            image, mask = TF.hflip(image), TF.hflip(mask)
        if random.random() < cfg.get("vflip_p", 0.5):
            image, mask = TF.vflip(image), TF.vflip(mask)
        if cfg.get("max_rotate_degrees", 0):
            angle = random.uniform(-cfg["max_rotate_degrees"], cfg["max_rotate_degrees"])
            image = TF.rotate(image, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST, fill=0)
        if cfg.get("brightness", 0):
            image = TF.adjust_brightness(image, random.uniform(1 - cfg["brightness"], 1 + cfg["brightness"]))
        if cfg.get("contrast", 0):
            image = TF.adjust_contrast(image, random.uniform(1 - cfg["contrast"], 1 + cfg["contrast"]))
        if random.random() < cfg.get("clahe_p", 0.0):
            image = _apply_clahe(image)
        if random.random() < cfg.get("noise_p", 0.0):
            image = _apply_noise(image, cfg)
        if random.random() < cfg.get("blur_p", 0.0):
            image = _apply_blur(image, cfg)
        return image, mask


def split_rows_by_subset(manifest_csv: str | Path, subset: str) -> list[dict[str, str]]:
    """Load manifest rows matching a subset name."""
    return [row for row in _load_rows(manifest_csv) if row.get("subset") == subset]


def _load_rows(rows: list[dict[str, str]] | str | Path) -> list[dict[str, str]]:
    """Load rows from a manifest CSV unless rows are already materialized."""
    if isinstance(rows, list):
        return rows
    path = resolve_project_path(rows)
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_project_root(start: str | Path | None = None) -> Path:
    """Find the repository root from a known local path or parent directories."""
    known_roots = [Path(r"D:/Nornikel-2026-Shlif-Case")]
    start_path = Path(start or Path.cwd()).resolve()
    for candidate in [*known_roots, start_path, *start_path.parents]:
        if all((candidate / marker).exists() for marker in PROJECT_MARKERS):
            return candidate
    return start_path


def resolve_project_path(path: str | Path, project_root: str | Path | None = None) -> Path:
    """Resolve relative project paths against the repository root."""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    root = Path(project_root) if project_root is not None else find_project_root()
    return root / candidate


def _optional_float(value: str | None) -> float | None:
    """Parse optional floating-point values from NT-MDT headers."""
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _apply_clahe(image: Image.Image) -> Image.Image:
    """Apply CLAHE to the luminance channel of an RGB image."""
    rgb = np.asarray(image.convert("RGB"))
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_chan, a_chan, b_chan = cv2.split(lab)
    l_chan = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l_chan)
    return Image.fromarray(cv2.cvtColor(cv2.merge([l_chan, a_chan, b_chan]), cv2.COLOR_LAB2RGB), mode="RGB")


def _apply_noise(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
    """Add Gaussian sensor-like noise to an image."""
    rgb = np.asarray(image).astype(np.float32)
    sigma = random.uniform(cfg.get("noise_sigma_min", 1.0), cfg.get("noise_sigma_max", 6.0))
    rgb += np.random.normal(0.0, sigma, size=rgb.shape).astype(np.float32)
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")


def _apply_blur(image: Image.Image, cfg: dict[str, Any]) -> Image.Image:
    """Apply a small Gaussian blur to mimic acquisition softness."""
    rgb = np.asarray(image)
    ksize = int(random.choice(cfg.get("blur_kernel_sizes", [3, 5])))
    if ksize % 2 == 0:
        ksize += 1
    return Image.fromarray(cv2.GaussianBlur(rgb, (ksize, ksize), 0), mode="RGB")


def _grid_positions(length: int, patch_size: int, stride: int) -> list[int]:
    """Return patch start coordinates that cover the whole axis."""
    if length <= patch_size:
        return [0]
    positions = list(range(0, length - patch_size + 1, stride))
    last = length - patch_size
    if positions[-1] != last:
        positions.append(last)
    return positions
