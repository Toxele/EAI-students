from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass
class GoldenOreInclusion:
    """One algorithmically detected golden ore inclusion."""

    inclusion_id: int
    bbox: tuple[int, int, int, int]
    area: int
    fill_ratio: float
    mean_hue: float
    mean_saturation: float
    mean_value: float


@dataclass
class GoldenOreDetectionResult:
    """Golden ore detector output with objects, mask, and visual overlay."""

    inclusions: list[GoldenOreInclusion]
    mask: NDArray[np.uint8]
    overlay_rgb: NDArray[np.uint8]
    ore_percent: float


class GoldenOreDetector:
    """Detect golden ore inclusions on dark polished-section OM images."""

    GOLD_HSV_PROTOTYPES = np.array(
        [
            [22.0, 120.0, 190.0],
            [28.0, 90.0, 220.0],
            [34.0, 150.0, 175.0],
            [18.0, 165.0, 215.0],
            [42.0, 110.0, 165.0],
        ],
        dtype=np.float32,
    )

    def __init__(
        self,
        hue_min: int = 12,
        hue_max: int = 48,
        min_saturation: int = 35,
        value_percentile: float = 78.0,
        local_value_percentile: float = 88.0,
        min_area_fraction: float = 0.000015,
        max_area_fraction: float = 0.04,
        min_fill_ratio: float = 0.08,
        min_box_gold_ratio: float = 0.018,
        min_knn_box_ratio: float = 0.004,
        min_knn_gold_ratio: float = 0.25,
        max_knn_distance: float = 0.36,
        min_aspect_ratio: float = 0.12,
        max_aspect_ratio: float = 8.0,
        box_padding: int = 2,
        component_close_size: int = 3,
        merge_intersecting_boxes: bool = True,
        box_merge_gap: int = 0,
        min_box_side: int = 100,
    ) -> None:
        """Store color and component filters for the detector."""
        self.hue_min = hue_min
        self.hue_max = hue_max
        self.min_saturation = min_saturation
        self.value_percentile = value_percentile
        self.local_value_percentile = local_value_percentile
        self.min_area_fraction = min_area_fraction
        self.max_area_fraction = max_area_fraction
        self.min_fill_ratio = min_fill_ratio
        self.min_box_gold_ratio = min_box_gold_ratio
        self.min_knn_box_ratio = min_knn_box_ratio
        self.min_knn_gold_ratio = min_knn_gold_ratio
        self.max_knn_distance = max_knn_distance
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.box_padding = box_padding
        self.component_close_size = component_close_size
        self.merge_intersecting_boxes = merge_intersecting_boxes
        self.box_merge_gap = box_merge_gap
        # Discard inclusions whose bbox sides are both no larger than this
        # threshold — too small a candidate for reliable classification.
        self.min_box_side = min_box_side

    def detect(self, image_rgb: NDArray[np.uint8]) -> GoldenOreDetectionResult:
        """Detect ore inclusions and return components, binary mask, and overlay."""
        raw_mask = self.build_mask(image_rgb)
        inclusions, mask = self._components_to_inclusions(image_rgb, raw_mask)
        overlay = self.draw_overlay(image_rgb, inclusions)
        ore_percent = float((mask > 0).mean() * 100.0)
        return GoldenOreDetectionResult(inclusions, mask, overlay, ore_percent)

    def build_mask(self, image_rgb: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Build a binary mask for golden metallic ore candidates."""
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
        hue = hsv[:, :, 0]
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        lightness = lab[:, :, 0]

        red = image_rgb[:, :, 0].astype(np.float32)
        green = image_rgb[:, :, 1].astype(np.float32)
        blue = image_rgb[:, :, 2].astype(np.float32)

        clahe = cv2.createCLAHE(clipLimit=1.8, tileGridSize=(8, 8))
        local_value = clahe.apply(value)
        local_lightness = clahe.apply(lightness)

        value_threshold = np.percentile(value, self.value_percentile)
        local_threshold = np.percentile(local_value, self.local_value_percentile)

        hue_mask = (hue >= self.hue_min) & (hue <= self.hue_max)
        saturation_mask = saturation >= self.min_saturation
        brightness_mask = (value >= value_threshold) | (local_value >= local_threshold)

        # Golden ore is yellowish: red and green dominate blue, while pure gray phases do not.
        yellow_ratio_mask = (red >= blue * 1.08) & (green >= blue * 1.03) & (red >= green * 0.75)
        lab_yellow_mask = lab[:, :, 2] >= np.percentile(lab[:, :, 2], 65.0)

        mask = hue_mask & saturation_mask & brightness_mask & yellow_ratio_mask & lab_yellow_mask
        mask_uint8 = mask.astype(np.uint8) * 255

        # Remove isolated sensor noise and connect small fractured ore patches.
        kernel3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        kernel5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_uint8 = cv2.medianBlur(mask_uint8, 3)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel3)
        mask_uint8 = cv2.morphologyEx(mask_uint8, cv2.MORPH_CLOSE, kernel5, iterations=1)
        return mask_uint8

    def draw_overlay(
        self, image_rgb: NDArray[np.uint8], inclusions: list[GoldenOreInclusion]
    ) -> NDArray[np.uint8]:
        """Draw detected bounding boxes over the original image."""
        overlay = image_rgb.copy()
        line_width = max(2, min(image_rgb.shape[:2]) // 900)
        for inclusion in inclusions:
            x, y, width, height = inclusion.bbox
            cv2.rectangle(
                overlay,
                (x, y),
                (x + width, y + height),
                color=(255, 0, 0),
                thickness=line_width,
            )
        return overlay

    def _components_to_inclusions(
        self, image_rgb: NDArray[np.uint8], mask: NDArray[np.uint8]
    ) -> tuple[list[GoldenOreInclusion], NDArray[np.uint8]]:
        """Convert connected mask components to filtered inclusion records."""
        height, width = mask.shape[:2]
        component_mask = self._component_mask(mask)
        accepted_mask = np.zeros_like(mask)
        min_area = max(20, int(height * width * self.min_area_fraction))
        max_area = max(min_area + 1, int(height * width * self.max_area_fraction))

        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(component_mask, 8)

        inclusions: list[GoldenOreInclusion] = []
        for label_idx in range(1, num_labels):
            x = int(stats[label_idx, cv2.CC_STAT_LEFT])
            y = int(stats[label_idx, cv2.CC_STAT_TOP])
            box_width = int(stats[label_idx, cv2.CC_STAT_WIDTH])
            box_height = int(stats[label_idx, cv2.CC_STAT_HEIGHT])
            closed_area = int(stats[label_idx, cv2.CC_STAT_AREA])
            label_patch = labels[y : y + box_height, x : x + box_width]
            mask_patch = mask[y : y + box_height, x : x + box_width]
            component_pixels = label_patch == label_idx
            gold_pixels = component_pixels & (mask_patch > 0)
            area = int(gold_pixels.sum())

            if area < min_area or area > max_area:
                continue
            if max(box_width, box_height) <= self.min_box_side:
                continue

            aspect_ratio = box_width / max(box_height, 1)
            fill_ratio = area / max(closed_area, 1)
            box_gold_ratio = area / max(box_width * box_height, 1)
            if aspect_ratio < self.min_aspect_ratio or aspect_ratio > self.max_aspect_ratio:
                continue
            if fill_ratio < self.min_fill_ratio:
                continue
            if box_gold_ratio < self.min_box_gold_ratio:
                continue

            hsv_patch = hsv[y : y + box_height, x : x + box_width]
            rgb_patch = image_rgb[y : y + box_height, x : x + box_width]
            knn_box_ratio, knn_gold_ratio = self._gold_knn_ratios(hsv_patch, rgb_patch, gold_pixels)
            if knn_box_ratio < self.min_knn_box_ratio:
                continue
            if knn_gold_ratio < self.min_knn_gold_ratio:
                continue

            hue_values = hsv_patch[:, :, 0][gold_pixels]
            saturation_values = hsv_patch[:, :, 1][gold_pixels]
            value_values = hsv_patch[:, :, 2][gold_pixels]
            padded_box = self._pad_box(x, y, box_width, box_height, width, height)
            accepted_patch = accepted_mask[y : y + box_height, x : x + box_width]
            accepted_patch[gold_pixels] = 255

            inclusions.append(
                GoldenOreInclusion(
                    inclusion_id=len(inclusions),
                    bbox=padded_box,
                    area=area,
                    fill_ratio=float(fill_ratio),
                    mean_hue=float(hue_values.mean()) if hue_values.size else 0.0,
                    mean_saturation=float(saturation_values.mean()) if saturation_values.size else 0.0,
                    mean_value=float(value_values.mean()) if value_values.size else 0.0,
                )
            )
        if self.merge_intersecting_boxes:
            return self._merge_intersecting_inclusions(inclusions, width, height), accepted_mask
        return inclusions, accepted_mask

    def _component_mask(self, mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """Merge nearby ore pixels for aggregate-level bounding boxes."""
        if self.component_close_size <= 1:
            return mask
        kernel_size = self.component_close_size
        if kernel_size % 2 == 0:
            kernel_size += 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

    def _gold_knn_ratios(
        self,
        hsv_patch: NDArray[np.uint8],
        rgb_patch: NDArray[np.uint8],
        gold_pixels: NDArray[np.bool_],
    ) -> tuple[float, float]:
        """Estimate how much of a candidate box is close to golden color prototypes."""
        if hsv_patch.size == 0 or not np.any(gold_pixels):
            return 0.0, 0.0

        patch_features = self._gold_color_features(
            hsv_patch.reshape(-1, 3).astype(np.float32),
            rgb_patch.reshape(-1, 3).astype(np.float32),
        )
        prototype_rgb = np.array(
            [
                [210.0, 175.0, 80.0],
                [230.0, 215.0, 150.0],
                [180.0, 145.0, 55.0],
                [235.0, 190.0, 70.0],
                [170.0, 155.0, 95.0],
            ],
            dtype=np.float32,
        )
        prototype_features = self._gold_color_features(self.GOLD_HSV_PROTOTYPES.copy(), prototype_rgb)
        distances = np.linalg.norm(
            patch_features[:, None, :] - prototype_features[None, :, :],
            axis=2,
        )
        close_pixels = distances.min(axis=1).reshape(hsv_patch.shape[:2]) <= self.max_knn_distance
        return float(close_pixels.mean()), float(close_pixels[gold_pixels].mean())

    @staticmethod
    def _gold_color_features(
        hsv_values: NDArray[np.float32], rgb_values: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Build color features that see golden candidates from HSV and RGB directions."""
        safe_rgb = np.maximum(rgb_values, 1.0)
        red = safe_rgb[:, 0]
        green = safe_rgb[:, 1]
        blue = safe_rgb[:, 2]
        return np.column_stack(
            [
                hsv_values[:, 0] / 90.0,
                hsv_values[:, 1] / 255.0,
                hsv_values[:, 2] / 255.0,
                np.clip(red / green, 0.0, 2.5) / 2.5,
                np.clip(green / blue, 0.0, 3.0) / 3.0,
                np.clip((red + green) / (2.0 * blue), 0.0, 3.0) / 3.0,
            ]
        )

    def _merge_intersecting_inclusions(
        self,
        inclusions: list[GoldenOreInclusion],
        image_width: int,
        image_height: int,
    ) -> list[GoldenOreInclusion]:
        """Merge intersecting or almost-touching boxes into aggregate boxes."""
        merged = list(inclusions)
        changed = True
        while changed:
            changed = False
            output: list[GoldenOreInclusion] = []
            used = [False] * len(merged)
            for idx, inclusion in enumerate(merged):
                if used[idx]:
                    continue
                current = inclusion
                used[idx] = True
                for other_idx in range(idx + 1, len(merged)):
                    if used[other_idx]:
                        continue
                    candidate = merged[other_idx]
                    if self._boxes_intersect(current.bbox, candidate.bbox, self.box_merge_gap):
                        current = self._union_inclusions(current, candidate, image_width, image_height)
                        used[other_idx] = True
                        changed = True
                output.append(current)
            merged = output

        return [
            GoldenOreInclusion(
                inclusion_id=idx,
                bbox=inclusion.bbox,
                area=inclusion.area,
                fill_ratio=inclusion.fill_ratio,
                mean_hue=inclusion.mean_hue,
                mean_saturation=inclusion.mean_saturation,
                mean_value=inclusion.mean_value,
            )
            for idx, inclusion in enumerate(merged)
        ]

    @staticmethod
    def _boxes_intersect(
        first: tuple[int, int, int, int], second: tuple[int, int, int, int], gap: int
    ) -> bool:
        """Return true when two boxes overlap or are separated by at most gap pixels."""
        x1, y1, w1, h1 = first
        x2, y2, w2, h2 = second
        return not (
            x1 + w1 + gap < x2
            or x2 + w2 + gap < x1
            or y1 + h1 + gap < y2
            or y2 + h2 + gap < y1
        )

    @staticmethod
    def _union_inclusions(
        first: GoldenOreInclusion,
        second: GoldenOreInclusion,
        image_width: int,
        image_height: int,
    ) -> GoldenOreInclusion:
        """Build one inclusion record from two intersecting inclusions."""
        x1, y1, w1, h1 = first.bbox
        x2, y2, w2, h2 = second.bbox
        left = max(0, min(x1, x2))
        top = max(0, min(y1, y2))
        right = min(image_width, max(x1 + w1, x2 + w2))
        bottom = min(image_height, max(y1 + h1, y2 + h2))
        area = first.area + second.area
        total_area = max(area, 1)
        return GoldenOreInclusion(
            inclusion_id=min(first.inclusion_id, second.inclusion_id),
            bbox=(left, top, right - left, bottom - top),
            area=area,
            fill_ratio=(first.fill_ratio * first.area + second.fill_ratio * second.area) / total_area,
            mean_hue=(first.mean_hue * first.area + second.mean_hue * second.area) / total_area,
            mean_saturation=(
                first.mean_saturation * first.area + second.mean_saturation * second.area
            )
            / total_area,
            mean_value=(first.mean_value * first.area + second.mean_value * second.area) / total_area,
        )

    def _pad_box(
        self, x: int, y: int, width: int, height: int, image_width: int, image_height: int
    ) -> tuple[int, int, int, int]:
        """Expand a component box by a small padding while staying inside the image."""
        padded_x = max(0, x - self.box_padding)
        padded_y = max(0, y - self.box_padding)
        x2 = min(image_width, x + width + self.box_padding)
        y2 = min(image_height, y + height + self.box_padding)
        return padded_x, padded_y, x2 - padded_x, y2 - padded_y
