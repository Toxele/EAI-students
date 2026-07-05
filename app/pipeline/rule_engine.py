"""
Expert rules for classifying the ore sort (from the task specification).

This is NOT a model — fixed logic on top of the metrics.
"""
from __future__ import annotations

from dataclasses import dataclass

from app.config import TALC_PERCENT_THRESHOLD


@dataclass
class RuleInput:
    """Metrics used for the decision."""

    talc_percent: float | None
    ordinary_percent: float
    thin_percent: float
    talc_available: bool


@dataclass
class RuleOutput:
    sort_code: str
    sort_label_ru: str
    explanation: str


def apply_rules(data: RuleInput) -> RuleOutput:
    """
    Applies the rule from the task specification:

    - talc > TALC_PERCENT_THRESHOLD → talc ore
    - otherwise: ordinary dominates → ordinary ore, else → hard-to-beneficiate ore
    """
    # If talc was measured and exceeds the config threshold
    if (
        data.talc_available
        and data.talc_percent is not None
        and data.talc_percent > TALC_PERCENT_THRESHOLD
    ):
        return RuleOutput(
            sort_code="otalkovannaya",
            sort_label_ru="оталькованная",
            explanation=(
                f"Содержание талька {data.talc_percent:.1f}% (>{TALC_PERCENT_THRESHOLD:.0f}%). "
                f"Рядовые срастания {data.ordinary_percent:.1f}%, "
                f"тонкие {data.thin_percent:.1f}%."
            ),
        )

    # No talc or talc <= 10% — look at intergrowths
    if data.ordinary_percent >= data.thin_percent:
        extra = ""
        if not data.talc_available:
            extra = " Тальк не оценивался (режим панорамы)."
        return RuleOutput(
            sort_code="ryadovaya",
            sort_label_ru="рядовая",
            explanation=(
                f"Преобладают обычные срастания ({data.ordinary_percent:.1f}% vs "
                f"тонкие {data.thin_percent:.1f}%).{extra}"
            ),
        )

    extra = ""
    if not data.talc_available:
        extra = " Тальк не оценивался (режим панорамы)."
    return RuleOutput(
        sort_code="trudnoobogatimaya",
        sort_label_ru="труднообогатимая",
        explanation=(
            f"Преобладают тонкие срастания ({data.thin_percent:.1f}% vs "
            f"рядовые {data.ordinary_percent:.1f}%).{extra}"
        ),
    )
