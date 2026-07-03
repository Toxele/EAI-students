from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class WeakMaskResult:
    image_path: str
    mask_path: str
    overlay_path: str
    foreground_fraction: float
    contours: int


class BlueLineTalcMaskExtractor:
    def __init__(
        self,
        blue_hsv_lower: list[int],
        blue_hsv_upper: list[int],
        close_kernel: int = 17,
        dilate_iterations: int = 1,
        min_contour_area: int = 200,
        overlay_alpha: float = 0.45,
    ) -> None:
        self.lower = np.array(blue_hsv_lower, dtype=np.uint8)
        self.upper = np.array(blue_hsv_upper, dtype=np.uint8)
        self.close_kernel = close_kernel
        self.dilate_iterations = dilate_iterations
        self.min_contour_area = min_contour_area
        self.overlay_alpha = overlay_alpha

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "BlueLineTalcMaskExtractor":
        return cls(
            blue_hsv_lower=cfg["blue_hsv_lower"],
            blue_hsv_upper=cfg["blue_hsv_upper"],
            close_kernel=cfg.get("close_kernel", 17),
            dilate_iterations=cfg.get("dilate_iterations", 1),
            min_contour_area=cfg.get("min_contour_area", 200),
            overlay_alpha=cfg.get("overlay_alpha", 0.45),
        )

    def extract(self, image_bgr: np.ndarray) -> tuple[np.ndarray, int]:
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        line_mask = cv2.inRange(hsv, self.lower, self.upper)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (self.close_kernel, self.close_kernel)
        )
        closed = cv2.morphologyEx(line_mask, cv2.MORPH_CLOSE, kernel)
        if self.dilate_iterations:
            closed = cv2.dilate(closed, kernel, iterations=self.dilate_iterations)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filled = np.zeros(line_mask.shape, dtype=np.uint8)
        kept = 0
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue
            cv2.drawContours(filled, [contour], -1, color=255, thickness=-1)
            kept += 1
        return filled, kept

    def overlay(self, image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        color = np.zeros_like(image_bgr)
        color[:, :, 0] = 255
        blended = cv2.addWeighted(image_bgr, 1.0, color, self.overlay_alpha, 0)
        return np.where(mask[:, :, None] > 0, blended, image_bgr)


class WeakMaskBatchExporter:
    def __init__(self, extractor: BlueLineTalcMaskExtractor, save_overlays: bool = True) -> None:
        self.extractor = extractor
        self.save_overlays = save_overlays

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "WeakMaskBatchExporter":
        return cls(
            extractor=BlueLineTalcMaskExtractor.from_config(cfg),
            save_overlays=cfg.get("save_overlays", True),
        )

    def export(self, input_root: str | Path, output_root: str | Path) -> list[WeakMaskResult]:
        input_root = Path(input_root)
        output_root = Path(output_root)
        mask_root = output_root / "masks"
        overlay_root = output_root / "overlays"
        mask_root.mkdir(parents=True, exist_ok=True)
        overlay_root.mkdir(parents=True, exist_ok=True)

        results: list[WeakMaskResult] = []
        for path in self._iter_weak_talc_images(input_root):
            image = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                continue
            mask, contour_count = self.extractor.extract(image)
            rel = path.relative_to(input_root)
            safe_name = "__".join(rel.parts)
            mask_path = mask_root / f"{Path(safe_name).stem}.png"
            overlay_path = overlay_root / f"{Path(safe_name).stem}.jpg"
            cv2.imencode(".png", mask)[1].tofile(str(mask_path))
            if self.save_overlays:
                cv2.imencode(".jpg", self.extractor.overlay(image, mask))[1].tofile(str(overlay_path))
            results.append(
                WeakMaskResult(
                    image_path=str(path),
                    mask_path=str(mask_path),
                    overlay_path=str(overlay_path) if self.save_overlays else "",
                    foreground_fraction=float((mask > 0).mean()),
                    contours=contour_count,
                )
            )
        return results

    @staticmethod
    def _iter_weak_talc_images(root: Path):
        for dirpath, _, filenames in os.walk(root):
            if "области оталькования" not in dirpath.lower():
                continue
            for filename in filenames:
                if Path(filename).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
                    yield Path(dirpath) / filename
