from __future__ import annotations

import csv
import hashlib
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from data.manifest import read_manifest, write_manifest


class TalcSegmentationDatasetBuilder:
    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        self.seed = cfg.get("seed", 42)

    def build(self) -> list[dict[str, Any]]:
        positives = self._read_positives()
        negatives = self._make_negatives(max(1, round(len(positives) * self.cfg.get("negative_to_positive_ratio", 2.0))))
        rows = positives + negatives
        rows = self._split(rows)
        write_manifest(self.cfg["output_csv"], rows)
        return rows

    def _read_positives(self) -> list[dict[str, Any]]:
        report = Path(self.cfg["weak_mask_report_csv"])
        if not report.exists():
            raise FileNotFoundError(
                f"{report} does not exist. Run `python -m scripts.extract_weak_masks` first."
            )
        with report.open("r", newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        manual_dir = Path(self.cfg.get("manual_mask_dir", ""))
        include_weak_fallback = self.cfg.get("include_weak_fallback", False)
        output: list[dict[str, Any]] = []
        for row in rows:
            image_path = row["image_path"]
            mask_path = row["mask_path"]
            sample_type = "positive_weak"
            manual_mask = self._manual_mask_path(manual_dir, image_path) if manual_dir else None
            if manual_mask and manual_mask.exists():
                mask_path = str(manual_mask)
                sample_type = "positive_manual"
            elif not include_weak_fallback:
                continue
            foreground_fraction = self._mask_fraction(mask_path)
            output.append(
                {
                    "image_path": image_path,
                    "mask_path": mask_path,
                    "sample_type": sample_type,
                    "source_label": "talc",
                    "foreground_fraction": foreground_fraction,
                    "contours": row.get("contours", ""),
                    "subset": "",
                }
            )
        if not output:
            raise ValueError(
                f"No positive talc masks found. Check manual_mask_dir={manual_dir} "
                f"or set include_weak_fallback=true."
            )
        return output

    def _make_negatives(self, count: int) -> list[dict[str, Any]]:
        manifest = read_manifest(self.cfg["manifest_csv"])
        candidates = []
        for row in manifest:
            if row.get("label") not in self.cfg.get("negative_labels", ["ordinary", "thin"]):
                continue
            if row.get("source") not in self.cfg.get("negative_sources", ["classification"]):
                continue
            if self.cfg.get("exclude_conflicts", True) and row.get("label_conflict", "").lower() == "true":
                continue
            candidates.append(row)
        rng = random.Random(self.seed)
        rng.shuffle(candidates)
        selected = candidates[: min(count, len(candidates))]
        mask_dir = Path(self.cfg["negative_mask_dir"])
        mask_dir.mkdir(parents=True, exist_ok=True)
        output: list[dict[str, Any]] = []
        for row in selected:
            mask_path = mask_dir / f"{Path(row['rel_path']).stem}_{row['content_hash'][:8]}_zero.png"
            self._write_zero_mask(row["path"], mask_path)
            output.append(
                {
                    "image_path": row["path"],
                    "mask_path": str(mask_path),
                    "sample_type": "negative_zero",
                    "source_label": row["label"],
                    "foreground_fraction": 0.0,
                    "contours": 0,
                    "subset": "",
                }
            )
        return output

    def _split(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rng = random.Random(self.seed)
        by_type: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            by_type.setdefault(row["sample_type"], []).append(row)
        output: list[dict[str, Any]] = []
        for group in by_type.values():
            rng.shuffle(group)
            n_val = max(1, round(len(group) * self.cfg.get("val_fraction", 0.2))) if len(group) > 1 else 0
            for idx, row in enumerate(group):
                row = dict(row)
                row["subset"] = "val" if idx < n_val else "train"
                output.append(row)
        return output

    @staticmethod
    def _write_zero_mask(image_path: str, mask_path: Path) -> None:
        with Image.open(image_path) as img:
            width, height = img.size
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.imencode(".png", mask)[1].tofile(str(mask_path))

    @staticmethod
    def _safe_stem(path: str) -> str:
        stem = Path(path).stem.replace(" ", "_").replace("/", "__").replace("\\", "__")
        digest = hashlib.md5(path.encode("utf-8")).hexdigest()[:8]
        return f"{stem}_{digest}"

    @classmethod
    def _manual_mask_path(cls, manual_dir: Path, image_path: str) -> Path:
        return manual_dir / f"{cls._safe_stem(image_path)}__manual_mask.png"

    @staticmethod
    def _mask_fraction(mask_path: str | Path) -> float:
        mask = cv2.imdecode(np.fromfile(str(mask_path), dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"Could not read mask: {mask_path}")
        return float((mask > 0).mean())
