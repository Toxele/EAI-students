"""Заглушки моделей — заменяются на реальный ML на фазе 2."""

from app.models.classifier_stub import ClassifierStub, ClassificationResult
from app.models.golden_ore_detector import GoldenOreDetectionResult, GoldenOreDetector, GoldenOreInclusion
from app.models.panorama_grain_detector import Grain, PanoramaGrainDetector
from app.models.segmentation_stub import SegmentationResult, SegmentationStub
from app.models.talc_segmenter import TalcSegmenter

__all__ = [
    "GoldenOreDetector",
    "GoldenOreDetectionResult",
    "GoldenOreInclusion",
    "PanoramaGrainDetector",
    "Grain",
    "ClassifierStub",
    "ClassificationResult",
    "SegmentationStub",
    "SegmentationResult",
    "TalcSegmenter",
]
