"""
Main analysis orchestrator — ties together the models, rule_engine, and report.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from app.config import PANORAMA_TILE_MARGIN, PANORAMA_TILE_SIZE, TALC_PERCENT_THRESHOLD
from app.models.fixed_output_stub import FixedPanoramaGrainDetector, FixedPanoramaTalcDetector
from app.models.golden_ore_detector import GoldenOreDetector, GoldenOreInclusion
from app.models.ore_inclusion_classifier import OreInclusionClassifier
from app.models.panorama_grain_detector import Grain
from app.models.talc_segmenter import TalcSegmenter
from app.pipeline.metrics import enrich_grain, grain_confidence
from app.pipeline.mode_detector import detect_mode
from app.pipeline.report import format_conclusion
from app.pipeline.rule_engine import RuleInput, apply_rules
from app.pipeline.tiling import iter_tiles


@dataclass
class AnalysisReport:
    """Full analysis result for a single image."""

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
    talc_mask: NDArray[np.uint8] | None = None
    talc_confidence: NDArray[np.uint8] | None = None
    processed_width: int = 0
    processed_height: int = 0


class Analyzer:
    """Pipeline entry point. Three models + talc detector for the panorama."""

    def __init__(self) -> None:
        # STUB: detail mode — fixed grain bboxes (no trained model for
        # sulfide detection on detail shots); talc segmenter is Unet++.
        self.grain_detector = FixedPanoramaGrainDetector()
        # Panorama: algorithmic golden ore inclusion detector (HSV/LAB +
        # color KNN, see app/models/golden_ore_detector.py), run tile by
        # tile — see _analyze_panorama.
        self.golden_ore_detector = GoldenOreDetector()
        self.talc_detector = FixedPanoramaTalcDetector()
        # Real coarse/fine classifier (ordinary/thin intergrowth) — used in
        # both modes whenever talc is <= the talc-ore threshold.
        self.ore_classifier = OreInclusionClassifier()
        self.segmenter = TalcSegmenter()

    def analyze(
        self,
        image_rgb: NDArray[np.uint8],
        original_width: int,
        original_height: int,
        mode_hint: str | None = None,
    ) -> AnalysisReport:
        """
        Analyzes an image.

        :param mode_hint: 'panorama' | 'detail' | None (auto by size)
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
        The panorama is cut into tiles of at most PANORAMA_TILE_SIZE and
        processed tile by tile: processing the giant image whole (one pass,
        downscaled to fit the model) either hangs or collapses talc/grains
        to sub-pixel size and finds nothing. Each tile is taken with a
        PANORAMA_TILE_MARGIN context field — without it the model can't see
        neighboring pixels at the tile border, and predictions from adjacent
        tiles disagree at the seam (visible grid-line seams). Only the
        tile's core goes into the masks/grain list.
        """
        sulfide_mask = np.zeros((ph, pw), dtype=np.uint8)
        talc_mask = np.zeros((ph, pw), dtype=np.uint8)
        talc_confidence = np.zeros((ph, pw), dtype=np.uint8)
        grains: list[Grain] = []

        for tile in iter_tiles(image_rgb, PANORAMA_TILE_SIZE, margin=PANORAMA_TILE_MARGIN):
            origin_x = tile.core_x - tile.offset_x
            origin_y = tile.core_y - tile.offset_y

            detection = self.golden_ore_detector.detect(tile.image)
            core_sulfide = tile.crop_to_core(detection.mask)
            sulfide_mask[
                tile.core_y : tile.core_y + tile.core_h, tile.core_x : tile.core_x + tile.core_w
            ] = core_sulfide

            for inclusion in detection.inclusions:
                ix, iy, iw, ih = inclusion.bbox
                global_x, global_y = ix + origin_x, iy + origin_y
                # The inclusion's center must lie within the tile's core —
                # otherwise it only landed in the margin and will be found
                # in full by the neighboring tile, where this same area is
                # its core (without this check, inclusions at the seam
                # would be duplicated).
                cx, cy = global_x + iw / 2, global_y + ih / 2
                if not (
                    tile.core_x <= cx < tile.core_x + tile.core_w
                    and tile.core_y <= cy < tile.core_y + tile.core_h
                ):
                    continue
                patch = image_rgb[global_y : global_y + ih, global_x : global_x + iw]
                intergrowth_type, gray_ratio = self._classify_inclusion(patch, inclusion)
                grains.append(
                    Grain(
                        grain_id=len(grains),
                        bbox=(global_x, global_y, iw, ih),
                        area=inclusion.area,
                        intergrowth_type=intergrowth_type,
                        gray_ratio=gray_ratio,
                    )
                )

            if self.segmenter.ready:
                seg_result = self.segmenter.predict(tile.image)
                tile_talc = seg_result.talc_mask
                tile_confidence = seg_result.talc_confidence
            else:
                tile_talc, _ = self.talc_detector.predict(tile.image, detection.mask)
                # The stub doesn't estimate confidence — treat it as maximal
                # wherever the stub found anything at all.
                tile_confidence = tile_talc
            core_talc = tile.crop_to_core(tile_talc)
            core_confidence = tile.crop_to_core(tile_confidence)
            talc_mask[
                tile.core_y : tile.core_y + tile.core_h, tile.core_x : tile.core_x + tile.core_w
            ] = core_talc
            talc_confidence[
                tile.core_y : tile.core_y + tile.core_h, tile.core_x : tile.core_x + tile.core_w
            ] = core_confidence

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

        conclusion = format_conclusion(
            sort_label_ru=rule.sort_label_ru,
            talc_percent=talc_percent,
            talc_available=True,
            ordinary_percent=metrics["ordinary_percent"],
            thin_percent=metrics["thin_percent"],
            mode=mode,
        )

        classifier_match = None
        if grains:
            ordinary_n = sum(1 for g in grains if g.intergrowth_type == "ordinary")
            thin_n = len(grains) - ordinary_n
            source = "классификатор" if self.ore_classifier.ready else "эвристика (веса не загружены)"
            classifier_match = f"{source} вкраплений: ordinary={ordinary_n}, thin={thin_n}"

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
            classifier_match=classifier_match,
            talc_mask=talc_mask,
            talc_confidence=talc_confidence,
            processed_width=pw,
            processed_height=ph,
        )

    def _classify_inclusion(
        self, patch_rgb: NDArray[np.uint8], inclusion: GoldenOreInclusion
    ) -> tuple[str, float]:
        """Classifies a single inclusion as ordinary or thin."""
        if self.ore_classifier.ready and patch_rgb.size:
            result = self.ore_classifier.predict(patch_rgb)
            gray_ratio = (1.0 - result.prob_ordinary) * 0.5
            return result.intergrowth_type, gray_ratio

        # Fallback without weights: compact, barely-replaced blobs (high
        # fill_ratio) are ordinary; heavily fragmented ones are thin.
        if inclusion.fill_ratio >= 0.5:
            return "ordinary", 0.1
        return "thin", 0.4

    def _analyze_detail(
        self, image_rgb: NDArray[np.uint8], mode: str, pw: int, ph: int
    ) -> AnalysisReport:
        """Detail OM: talc segmenter + coarse/fine classifier."""
        seg = self.segmenter.predict(image_rgb)
        grains, sulfide_mask = self.grain_detector.detect(image_rgb)

        talc_mask = seg.talc_mask
        talc_percent = seg.talc_percent
        talc_confidence = seg.talc_confidence
        if not self.segmenter.ready:
            # Weights not loaded — fall back to the stub so the UI doesn't crash.
            talc_mask, talc_percent = self.talc_detector.predict(image_rgb, sulfide_mask)
            # The stub doesn't estimate confidence — treat it as maximal
            # wherever the stub found anything at all.
            talc_confidence = talc_mask

        # Talc above the threshold means the ore is already talc ore, no
        # classifier needed (see app/pipeline/rule_engine.py). Otherwise we
        # decide ordinary/hard-to-beneficiate with the real classifier — its
        # verdict is applied to every grain found.
        classifier_match = None
        if talc_percent <= TALC_PERCENT_THRESHOLD and self.ore_classifier.ready:
            result = self.ore_classifier.predict(image_rgb)
            for grain in grains:
                grain.intergrowth_type = result.intergrowth_type
                grain.gray_ratio = (1.0 - result.prob_ordinary) * 0.5
            classifier_match = f"классификатор: {result.intergrowth_type} (p_ordinary={result.prob_ordinary:.2f})"

        metrics = self._metrics_from_grains(grains, ph * pw)

        rule = apply_rules(
            RuleInput(
                talc_percent=talc_percent,
                ordinary_percent=metrics["ordinary_percent"],
                thin_percent=metrics["thin_percent"],
                talc_available=True,
            )
        )

        conclusion = format_conclusion(
            sort_label_ru=rule.sort_label_ru,
            talc_percent=talc_percent,
            talc_available=True,
            ordinary_percent=metrics["ordinary_percent"],
            thin_percent=metrics["thin_percent"],
            mode=mode,
        )

        explanation = rule.explanation
        if classifier_match:
            explanation += f" Реальный {classifier_match}."

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
            classifier_match=classifier_match,
            talc_mask=talc_mask,
            talc_confidence=talc_confidence,
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
