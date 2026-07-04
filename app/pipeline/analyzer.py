"""
Главный оркестратор анализа — связывает модели, rule_engine и отчёт.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.models.fixed_output_stub import (
    FixedOreClassifier,
    FixedPanoramaGrainDetector,
    FixedPanoramaTalcDetector,
)
from app.models.panorama_grain_detector import Grain
from app.models.talc_segmenter import TalcSegmenter
from app.pipeline.metrics import enrich_grain, grain_confidence
from app.pipeline.mode_detector import detect_mode
from app.pipeline.overlay import draw_grain_overlay, draw_talc_layer
from app.pipeline.report import format_conclusion
from app.pipeline.rule_engine import RuleInput, apply_rules


@dataclass
class AnalysisReport:
    """Полный результат анализа одного изображения."""

    mode: str
    sort_code: str
    sort_label_ru: str
    explanation: str
    conclusion: str
    talc_percent: float | None
    talc_available: bool
    sulfide_percent: float
    ordinary_percent: float
    thin_percent: float
    grain_count: int
    grains: list[dict[str, Any]] = field(default_factory=list)
    classifier_match: str | None = None
    overlay_rgb: NDArray[np.uint8] | None = None
    talc_mask: NDArray[np.uint8] | None = None
    overview_rgb: NDArray[np.uint8] | None = None
    processed_width: int = 0
    processed_height: int = 0


class Analyzer:
    """Точка входа pipeline. Три модели + talc detector для панорамы."""

    def __init__(self) -> None:
        # STUB: зёрна/классификатор — fixed_output_stub; сегментатор — обученный Unet++.
        self.grain_detector = FixedPanoramaGrainDetector()
        self.talc_detector = FixedPanoramaTalcDetector()
        self.classifier = FixedOreClassifier()
        self.segmenter = TalcSegmenter()

    def analyze(
        self,
        image_rgb: NDArray[np.uint8],
        original_width: int,
        original_height: int,
        mode_hint: str | None = None,
    ) -> AnalysisReport:
        """
        Анализирует изображение.

        :param mode_hint: 'panorama' | 'detail' | None (auto по размеру)
        """
        if mode_hint in ("panorama", "detail"):
            mode = mode_hint
        else:
            mode = detect_mode(original_width, original_height)

        ph, pw = image_rgb.shape[:2]
        if mode == "panorama":
            return self._analyze_panorama(image_rgb, mode, pw, ph)
        return self._analyze_detail(image_rgb, mode, pw, ph)

    def _analyze_panorama(
        self, image_rgb: NDArray[np.uint8], mode: str, pw: int, ph: int
    ) -> AnalysisReport:
        """Панорама: зёрна + тальк по тёмным областям."""
        grains, sulfide_mask = self.grain_detector.detect(image_rgb)
        talc_mask, talc_percent = self.talc_detector.predict(image_rgb, sulfide_mask)
        metrics = self._metrics_from_grains(grains, ph * pw)

        rule = apply_rules(
            RuleInput(
                talc_percent=talc_percent,
                ordinary_percent=metrics["ordinary_percent"],
                thin_percent=metrics["thin_percent"],
                talc_available=True,
            )
        )

        overlay = draw_grain_overlay(image_rgb, grains, talc_mask=talc_mask)
        conclusion = format_conclusion(
            sort_label_ru=rule.sort_label_ru,
            talc_percent=talc_percent,
            talc_available=True,
            ordinary_percent=metrics["ordinary_percent"],
            thin_percent=metrics["thin_percent"],
        )

        return AnalysisReport(
            mode=mode,
            sort_code=rule.sort_code,
            sort_label_ru=rule.sort_label_ru,
            explanation=rule.explanation,
            conclusion=conclusion,
            talc_percent=talc_percent,
            talc_available=True,
            sulfide_percent=metrics["sulfide_percent"],
            ordinary_percent=metrics["ordinary_percent"],
            thin_percent=metrics["thin_percent"],
            grain_count=len(grains),
            grains=[self._grain_to_dict(g) for g in grains],
            overlay_rgb=overlay,
            talc_mask=talc_mask,
            overview_rgb=image_rgb.copy(),
            processed_width=pw,
            processed_height=ph,
        )

    def _analyze_detail(
        self, image_rgb: NDArray[np.uint8], mode: str, pw: int, ph: int
    ) -> AnalysisReport:
        """Детальный OM: сегментатор + talc fallback + классификатор."""
        seg = self.segmenter.predict(image_rgb)
        clf = self.classifier.predict(image_rgb)
        grains, sulfide_mask = self.grain_detector.detect(image_rgb)

        talc_mask = seg.talc_mask
        talc_percent = seg.talc_percent
        if talc_percent <= 0:
            talc_mask, talc_percent = self.talc_detector.predict(image_rgb, sulfide_mask)

        metrics = self._metrics_from_grains(grains, ph * pw)

        rule = apply_rules(
            RuleInput(
                talc_percent=talc_percent,
                ordinary_percent=metrics["ordinary_percent"],
                thin_percent=metrics["thin_percent"],
                talc_available=True,
            )
        )

        overlay = draw_grain_overlay(image_rgb, grains, talc_mask=talc_mask)
        conclusion = format_conclusion(
            sort_label_ru=rule.sort_label_ru,
            talc_percent=talc_percent,
            talc_available=True,
            ordinary_percent=metrics["ordinary_percent"],
            thin_percent=metrics["thin_percent"],
        )

        explanation = rule.explanation
        if clf.matched_reference:
            explanation += f" Stub-классификатор: {clf.sort_label_ru} (~{clf.matched_reference})."

        return AnalysisReport(
            mode=mode,
            sort_code=rule.sort_code,
            sort_label_ru=rule.sort_label_ru,
            explanation=explanation,
            conclusion=conclusion,
            talc_percent=talc_percent,
            talc_available=True,
            sulfide_percent=metrics["sulfide_percent"],
            ordinary_percent=metrics["ordinary_percent"],
            thin_percent=metrics["thin_percent"],
            grain_count=len(grains),
            grains=[self._grain_to_dict(g) for g in grains],
            classifier_match=clf.matched_reference,
            overlay_rgb=overlay,
            talc_mask=talc_mask,
            overview_rgb=image_rgb.copy(),
            processed_width=pw,
            processed_height=ph,
        )

    def _metrics_from_grains(self, grains: list[Grain], total_pixels: int) -> dict[str, float]:
        if not grains:
            return {"sulfide_percent": 0.0, "ordinary_percent": 50.0, "thin_percent": 50.0}

        total_grain_area = sum(g.area for g in grains)
        ordinary_area = sum(g.area for g in grains if g.intergrowth_type == "ordinary")
        thin_area = sum(g.area for g in grains if g.intergrowth_type == "thin")

        sulfide_percent = 100.0 * total_grain_area / max(total_pixels, 1)
        ordinary_percent = 100.0 * ordinary_area / max(total_grain_area, 1)
        thin_percent = 100.0 * thin_area / max(total_grain_area, 1)

        return {
            "sulfide_percent": sulfide_percent,
            "ordinary_percent": ordinary_percent,
            "thin_percent": thin_percent,
        }

    @staticmethod
    def _grain_to_dict(grain: Grain) -> dict[str, Any]:
        conf_o, conf_t = grain_confidence(grain.gray_ratio)
        return enrich_grain(
            {
                "id": grain.grain_id,
                "bbox": list(grain.bbox),
                "area": grain.area,
                "intergrowth_type": grain.intergrowth_type,
                "gray_ratio": round(grain.gray_ratio, 3),
                "conf_ordinary": conf_o,
                "conf_thin": conf_t,
            }
        )
