"""
Формирование текстового отчёта и CSV-метрик для UI и API.

Отдельный модуль — не модель и не rule_engine.
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass

from app.config import TALC_PERCENT_THRESHOLD


@dataclass
class ReportMetrics:
    """Табличные метрики для экспорта."""

    sulfide_percent: float
    ordinary_percent: float
    thin_percent: float
    talc_percent: float | None
    talc_available: bool
    grain_count: int
    sort_label_ru: str
    sort_code: str
    mode: str


def format_conclusion(
    sort_label_ru: str,
    talc_percent: float | None,
    talc_available: bool,
    ordinary_percent: float,
    thin_percent: float,
    mode: str,
) -> str:
    """
    Краткое заключение в стиле постановки задачи.

    Пример (панорама): «Руда классифицирована как оталькованная: содержание
    талька — 14%, преобладание тонких срастаний — 62%.»

    В режиме "detail" (близкое фото) сорт рядовая/труднообогатимая решает
    classifier на всём кадре (см. app/pipeline/analyzer.py), а не измеренная
    площадь срастаний — поэтому про их "преобладание" не пишем, оно там
    тривиально 100/0 и ничего не сообщает пользователю.
    """
    talc_dominant = talc_available and talc_percent is not None and talc_percent > TALC_PERCENT_THRESHOLD

    if mode == "detail":
        if talc_dominant:
            return (
                f"Руда классифицирована как **{sort_label_ru}**: "
                f"содержание талька — {talc_percent:.1f}% (>{TALC_PERCENT_THRESHOLD:.0f}%)."
            )
        talc_str = f"{talc_percent:.1f}%" if talc_available and talc_percent is not None else "0%"
        return (
            f"Руда классифицирована как **{sort_label_ru}**: "
            f"содержание талька — {talc_str} (≤{TALC_PERCENT_THRESHOLD:.0f}%)."
        )

    if talc_dominant:
        dominant = "тонких" if thin_percent >= ordinary_percent else "обычных"
        dominant_pct = max(thin_percent, ordinary_percent)
        return (
            f"Руда классифицирована как **{sort_label_ru}**: "
            f"содержание талька — {talc_percent:.1f}%, "
            f"преобладание {dominant} срастаний — {dominant_pct:.1f}%."
        )

    if not talc_available:
        dominant = "тонких" if thin_percent > ordinary_percent else "обычных"
        dominant_pct = max(thin_percent, ordinary_percent)
        return (
            f"Руда классифицирована как **{sort_label_ru}** "
            f"(режим панорамы, тальк не оценивался): "
            f"преобладание {dominant} срастаний — {dominant_pct:.1f}%, "
            f"рядовые {ordinary_percent:.1f}%, тонкие {thin_percent:.1f}%."
        )

    dominant = "тонких" if thin_percent > ordinary_percent else "обычных"
    dominant_pct = max(thin_percent, ordinary_percent)
    talc_str = f"{talc_percent:.1f}%" if talc_percent is not None else "0%"
    return (
        f"Руда классифицирована как **{sort_label_ru}**: "
        f"содержание талька — {talc_str} (≤{TALC_PERCENT_THRESHOLD:.0f}%), "
        f"преобладание {dominant} срастаний — {dominant_pct:.1f}%."
    )


def metrics_to_csv(metrics: ReportMetrics) -> str:
    """Возвращает CSV-строку с одной строкой метрик."""
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(
        [
            "sort",
            "mode",
            "sulfide_percent",
            "ordinary_percent",
            "thin_percent",
            "talc_percent",
            "grain_count",
        ]
    )
    talc_val = "" if not metrics.talc_available else f"{metrics.talc_percent or 0:.2f}"
    writer.writerow(
        [
            metrics.sort_label_ru,
            metrics.mode,
            f"{metrics.sulfide_percent:.2f}",
            f"{metrics.ordinary_percent:.2f}",
            f"{metrics.thin_percent:.2f}",
            talc_val,
            metrics.grain_count,
        ]
    )
    return output.getvalue()
