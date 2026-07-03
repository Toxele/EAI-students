"""Быстрый smoke-test pipeline."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import CH1_DATA_DIR, DATA_DIR
from app.pipeline.analyzer import Analyzer
from app.pipeline.loader import imread_unicode, load_image

a = Analyzer()
print("references:", len(a.classifier.references))

detail = CH1_DATA_DIR / "Рядовые руды" / "2539590-1.JPG"
bgr = imread_unicode(detail)
h, w = bgr.shape[:2]
r = a.analyze(load_image(detail), w, h)
print("detail:", r.mode, r.sort_label_ru, "grains:", r.grain_count)

pan = DATA_DIR / "Панорамы" / "4.jpg"
bgr2 = imread_unicode(pan)
h2, w2 = bgr2.shape[:2]
r2 = a.analyze(load_image(pan), w2, h2)
print("panorama:", r2.mode, r2.sort_label_ru, "grains:", r2.grain_count)
print("OK")
