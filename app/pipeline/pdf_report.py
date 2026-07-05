"""
Генерация PDF-отчёта по результатам анализа.

Простой одностраничный PDF: сорт, заключение, таблица метрик, опционально overlay.
Шрифт с поддержкой кириллицы — DejaVu (bundled) или системный Arial/DejaVu.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from fpdf import FPDF
from numpy.typing import NDArray
from PIL import Image

from app.config import PROJECT_ROOT, TALC_PERCENT_THRESHOLD
from app.pipeline.report import ReportMetrics

# Имя семейства шрифта после add_font
_FONT_FAMILY = "ReportFont"

# Поиск TTF с кириллицей: bundled → Windows → Linux
_FONT_CANDIDATES = [
    PROJECT_ROOT / "app" / "fonts" / "DejaVuSans.ttf",
    Path("C:/Windows/Fonts/arial.ttf"),
    Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"),
]


def _find_cyrillic_font() -> Path:
    """Возвращает путь к первому доступному Unicode-шрифту."""
    for path in _FONT_CANDIDATES:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Не найден TTF-шрифт с кириллицей. Положите DejaVuSans.ttf в app/fonts/."
    )


def _strip_markdown(text: str) -> str:
    """Убирает ** из markdown-заключения для plain-text PDF."""
    return re.sub(r"\*\*", "", text)


def _setup_pdf() -> FPDF:
    """Создаёт FPDF с зарегистрированным Unicode-шрифтом."""
    pdf = FPDF()
    font_path = _find_cyrillic_font()
    pdf.add_font(family=_FONT_FAMILY, fname=str(font_path))
    pdf.set_font(_FONT_FAMILY, size=11)
    return pdf


def build_pdf_bytes(
    metrics: ReportMetrics,
    conclusion: str,
    explanation: str = "",
    overlay_rgb: NDArray[np.uint8] | None = None,
    overview_rgb: NDArray[np.uint8] | None = None,
    talc_layer_rgb: NDArray[np.uint8] | None = None,
    type_layer_rgb: NDArray[np.uint8] | None = None,
    counts: dict | None = None,
) -> bytes:
    """
    Собирает PDF-отчёт в память.

    :param metrics: табличные метрики анализа
    :param conclusion: краткое заключение (русский текст)
    :param explanation: пояснение rule engine (опционально)
    :param overlay_rgb: RGB-изображение overlay для вставки в PDF
    :returns: bytes PDF-файла
    """
    pdf = _setup_pdf()
    pdf.add_page()

    # --- Заголовок ---
    pdf.set_font(_FONT_FAMILY, size=16)
    pdf.cell(0, 10, "Nornickel — отчёт по анализу шлифа", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Сорт и режим ---
    pdf.set_font(_FONT_FAMILY, size=12)
    pdf.cell(0, 8, f"Сорт: {metrics.sort_label_ru}", new_x="LMARGIN", new_y="NEXT")
    mode_ru = "панорама" if metrics.mode == "panorama" else "детальный OM"
    pdf.cell(0, 8, f"Режим: {mode_ru}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Заключение ---
    pdf.set_font(_FONT_FAMILY, size=11)
    pdf.multi_cell(pdf.epw, 6, "Заключение:", new_x="LMARGIN", new_y="NEXT")
    pdf.multi_cell(pdf.epw, 6, _strip_markdown(conclusion), new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    # --- Таблица метрик ---
    pdf.set_font(_FONT_FAMILY, size=11)
    pdf.cell(0, 8, "Метрики:", new_x="LMARGIN", new_y="NEXT")

    talc_str = "н/д" if not metrics.talc_available else f"{metrics.talc_percent or 0:.2f}%"
    k_str = str(counts["total_k"]) if counts else str(metrics.grain_count)
    l_str = str(counts.get("ordinary_l", "—")) if counts else "—"
    j_str = str(counts.get("thin_j", "—")) if counts else "—"
    table_rows = [
        ("Показатель", "Значение"),
        ("Сульфиды", f"{metrics.sulfide_percent:.2f}%"),
        ("Рядовые срастания", f"{metrics.ordinary_percent:.2f}%"),
        ("Тонкие срастания", f"{metrics.thin_percent:.2f}%"),
        ("Тальк", talc_str),
        ("k — вкр. всего", k_str),
        ("l — рядовых", l_str),
        ("j — тонких", j_str),
        ("Порог оталькованности", f"{TALC_PERCENT_THRESHOLD:.0f}%"),
    ]

    # Простая таблица без bold-заголовков (не нужен отдельный add_font для Bold)
    col_w = pdf.epw / 2
    for label, value in table_rows:
        pdf.cell(col_w, 7, label, border=1)
        pdf.cell(col_w, 7, value, border=1, new_x="LMARGIN", new_y="NEXT")

    pdf.ln(6)
    pdf.set_x(pdf.l_margin)

    # --- Пояснение rule engine (кратко) ---
    if explanation:
        pdf.set_font(_FONT_FAMILY, size=10)
        pdf.multi_cell(pdf.epw, 5, f"Правило: {explanation[:500]}", new_x="LMARGIN", new_y="NEXT")

    # --- Слои (обзор, тальк, тип) ---
    layers = [
        ("Обзор", overview_rgb if overview_rgb is not None else overlay_rgb),
        ("Тальк", talc_layer_rgb),
        ("Тип срастаний", type_layer_rgb),
    ]
    for title, layer_rgb in layers:
        if layer_rgb is None:
            continue
        pdf.add_page()
        pdf.set_font(_FONT_FAMILY, size=12)
        pdf.cell(0, 8, title, new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

        img = Image.fromarray(layer_rgb)
        max_w_mm = 190
        aspect = img.height / img.width
        w_mm = max_w_mm
        h_mm = w_mm * aspect
        if h_mm > 250:
            h_mm = 250
            w_mm = h_mm / aspect
        pdf.image(img, w=w_mm, h=h_mm)

    # Возвращаем bytes
    out = pdf.output()
    return bytes(out) if isinstance(out, (bytes, bytearray)) else out.encode("latin-1")
