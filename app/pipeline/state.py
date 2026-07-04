"""
Сохранение и пересчёт полного состояния анализа (зёрна, метрики, правки).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import RESULTS_DIR
from app.pipeline.metrics import (
    STATUS_FALSE_POSITIVE,
    STATUS_ORDINARY,
    STATUS_THIN,
    STATUS_UNCERTAIN,
    compute_grain_counts,
    compute_intergrowth_percent,
    enrich_grain,
    sulfide_area_percent,
)
from app.pipeline.report import format_conclusion
from app.pipeline.rule_engine import RuleInput, apply_rules


def result_json_path(result_id: str) -> Path:
    return RESULTS_DIR / f"{result_id}.json"


def load_state(result_id: str) -> dict[str, Any] | None:
    path = result_json_path(result_id)
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_state(state: dict[str, Any]) -> None:
    path = result_json_path(state["result_id"])
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def scale_grains_to_original(
    grains: list[dict[str, Any]], scale_x: float, scale_y: float
) -> list[dict[str, Any]]:
    """Переводит bbox и area из processed → original координаты."""
    scaled: list[dict[str, Any]] = []
    for g in grains:
        x, y, w, h = g["bbox"]
        sx = int(round(x * scale_x))
        sy = int(round(y * scale_y))
        sw = int(round(w * scale_x))
        sh = int(round(h * scale_y))
        area = int(g["area"] * scale_x * scale_y)
        item = dict(g)
        item["bbox"] = [sx, sy, sw, sh]
        item["bbox_processed"] = [x, y, w, h]
        item["area"] = area
        scaled.append(item)
    return scaled


def recalculate_state(state: dict[str, Any]) -> dict[str, Any]:
    """
    Пересчитывает метрики, rule engine и conclusion после правок зёрен.

    Использует processed размер для % сульфидов (area в original coords / original pixels).
    """
    grains = [enrich_grain(g) for g in state.get("grains", [])]
    img = state.get("image", {})
    orig_w = int(img.get("original_width", 1))
    orig_h = int(img.get("original_height", 1))
    total_pixels = orig_w * orig_h

    counts = compute_grain_counts(grains)
    ordinary_pct, thin_pct = compute_intergrowth_percent(grains)
    sulfide_pct = sulfide_area_percent(grains, total_pixels)

    talc_available = bool(state.get("talc_available", False))
    talc_percent = state.get("talc_percent")

    rule = apply_rules(
        RuleInput(
            talc_percent=talc_percent,
            ordinary_percent=ordinary_pct,
            thin_percent=thin_pct,
            talc_available=talc_available,
        )
    )

    conclusion = format_conclusion(
        sort_label_ru=rule.sort_label_ru,
        talc_percent=talc_percent,
        talc_available=talc_available,
        ordinary_percent=ordinary_pct,
        thin_percent=thin_pct,
    )

    state["grains"] = grains
    state["sort_code"] = rule.sort_code
    state["sort_label_ru"] = rule.sort_label_ru
    state["explanation"] = rule.explanation
    state["conclusion"] = conclusion
    state["counts"] = counts
    state["metrics"] = {
        "sulfide_percent": sulfide_pct,
        "ordinary_percent": ordinary_pct,
        "thin_percent": thin_pct,
        "talc_percent": talc_percent,
        "talc_available": talc_available,
        "grain_count": counts["total_k"],
        "ordinary_count": counts["ordinary_l"],
        "thin_count": counts["thin_j"],
        "uncertain_count": counts["uncertain"],
        "false_positive_count": counts["false_positive"],
    }
    return state


def apply_corrections(state: dict[str, Any], updates: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Применяет правки пользователя к зёрнам по id.

    update: {id, status?, bbox?}
    status: ordinary | thin | uncertain | false_positive
    """
    by_id = {u["id"]: u for u in updates}
    new_grains: list[dict[str, Any]] = []

    for g in state.get("grains", []):
        item = dict(g)
        if item["id"] in by_id:
            patch = by_id[item["id"]]
            if "status" in patch:
                item["status"] = patch["status"]
                if patch["status"] in (STATUS_ORDINARY, STATUS_THIN):
                    item["intergrowth_type"] = patch["status"]
            if "bbox" in patch and patch["bbox"]:
                bx, by, bw, bh = patch["bbox"]
                item["bbox"] = [int(bx), int(by), int(bw), int(bh)]
                item["area"] = int(bw) * int(bh)
        new_grains.append(item)

    state["grains"] = new_grains
    return recalculate_state(state)


def save_talc_layer_png(result_id: str, talc_mask: np.ndarray) -> str:
    """Сохраняет маску талька как PNG. Возвращает относительный URL."""
    path = RESULTS_DIR / f"{result_id}_talc.png"
    cv2.imwrite(str(path), talc_mask)
    return f"/result/{result_id}/layer/talc"
