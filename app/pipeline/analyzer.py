"""
Главный оркестратор анализа — связывает модели, rule_engine и отчёт.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.config import PANORAMA_TILE_MARGIN, PANORAMA_TILE_SIZE
from app.models.fixed_output_stub import (
    FixedOreClassifier,
    FixedPanoramaGrainDetector,
    FixedPanoramaTalcDetector,
)
from app.models.panorama_grain_detector import Grain, PanoramaGrainDetector
from app.models.talc_segmenter import TalcSegmenter
from app.pipeline.metrics import enrich_grain, grain_confidence
from app.pipeline.mode_detector import detect_mode
from app.pipeline.overlay import draw_grain_overlay
from app.pipeline.report import format_conclusion
from app.pipeline.rule_engine import RuleInput, apply_rules
from app.pipeline.tiling import iter_tiles


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
    processed_width: int = 0
    processed_height: int = 0


class Analyzer:
    """Точка входа pipeline. Три модели + talc detector для панорамы."""

    def __init__(self) -> None:
        # STUB: детальный режим/классификатор — fixed_output_stub (без обученной
        # модели детекции зёрен под detail-снимки); сегментатор талька — Unet++.
        self.grain_detector = FixedPanoramaGrainDetector()
        # Панорама: реальный детектор зёрен (порог яркости + connected components),
        # гоняется по тайлам — см. _analyze_panorama.
        self.panorama_grain_detector = PanoramaGrainDetector()
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
        """
        Панорама режется на тайлы ≤PANORAMA_TILE_SIZE и обрабатывается поштучно:
        цельная обработка гигантского снимка (одним проходом, с downscale под
        модель) либо виснет, либо схлопывает тальк/зёрна до суб-пикселя и
        ничего не находит. Каждый тайл берётся с контекстным полем
        PANORAMA_TILE_MARGIN — без него модель на границе тайла не видит
        соседних пикселей, и предсказания смежных тайлов расходятся на стыке
        (видимые швы по сетке). В маски/список зёрен идёт только core тайла.
        """
        sulfide_mask = np.zeros((ph, pw), dtype=np.uint8)
        talc_mask = np.zeros((ph, pw), dtype=np.uint8)
        grains: list[Grain] = []

        for tile in iter_tiles(image_rgb, PANORAMA_TILE_SIZE, margin=PANORAMA_TILE_MARGIN):
            origin_x = tile.core_x - tile.offset_x
            origin_y = tile.core_y - tile.offset_y

            tile_grains, tile_sulfide = self.panorama_grain_detector.detect(tile.image)
            core_sulfide = tile.crop_to_core(tile_sulfide)
            sulfide_mask[
                tile.core_y : tile.core_y + tile.core_h, tile.core_x : tile.core_x + tile.core_w
            ] = core_sulfide

            for g in tile_grains:
                gx, gy, gw, gh = g.bbox
                global_x, global_y = gx + origin_x, gy + origin_y
                # Центр зерна должен лежать в core тайла — иначе оно попало
                # только в margin и будет целиком найдено соседним тайлом,
                # где эта же область — его core (без этого зёрна на стыке
                # задваивались бы).
                cx, cy = global_x + gw / 2, global_y + gh / 2
                if not (
                    tile.core_x <= cx < tile.core_x + tile.core_w
                    and tile.core_y <= cy < tile.core_y + tile.core_h
                ):
                    continue
                grains.append(
                    Grain(
                        grain_id=len(grains),
                        bbox=(global_x, global_y, gw, gh),
                        area=g.area,
                        intergrowth_type=g.intergrowth_type,
                        gray_ratio=g.gray_ratio,
                    )
                )

            if self.segmenter.ready:
                tile_talc = self.segmenter.predict(tile.image).talc_mask
            else:
                tile_talc, _ = self.talc_detector.predict(tile.image, tile_sulfide)
            core_talc = tile.crop_to_core(tile_talc)
            talc_mask[
                tile.core_y : tile.core_y + tile.core_h, tile.core_x : tile.core_x + tile.core_w
            ] = core_talc

        talc_percent = 100.0 * float(np.count_nonzero(talc_mask)) / max(ph * pw, 1)
        metrics = self._metrics_from_grains(grains, ph * pw)

        rule = apply_rules(
            RuleInput(
                talc_percent=talc_percent,
                ordinary_percent=metrics["ordinary_percent"],
                thin_percent=metrics["thin_percent"],
                talc_available=True,
            )
        )

        # Используется только Streamlit-фронтендом (frontend/app.py) для
        # предпросмотра overlay — при желании можно тоже перевести на тайлы,
        # но там снимки не панорамного размера.
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
        if not self.segmenter.ready:
            # Веса не загружены — используем заглушку, чтобы UI не падал.
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
