"""HTTP routes FastAPI — загрузка, анализ, сохранение результатов."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.api.schemas import (
    AnalysisResponse,
    CorrectionsResponse,
    CountsSchema,
    GrainSchema,
    MetricsSchema,
)
from app.config import RESULTS_DIR, UPLOAD_DIR
from app.pipeline.analyzer import Analyzer
from app.pipeline.loader import load_image_from_bytes
from app.pipeline.overlay import draw_talc_layer, draw_type_layer, save_overlay
from app.pipeline.pdf_report import build_pdf_bytes
from app.pipeline.report import ReportMetrics, metrics_to_csv
from app.pipeline.state import (
    apply_corrections,
    load_state,
    recalculate_state,
    save_state,
    save_talc_layer_png,
    scale_grains_to_original,
)

analyzer = Analyzer()

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def _safe_filename(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _upscale_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape[1] == width and mask.shape[0] == height:
        return mask
    return cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)


def _state_to_response(state: dict[str, Any]) -> dict[str, Any]:
    """Общие поля для AnalysisResponse / CorrectionsResponse."""
    rid = state["result_id"]
    m = state["metrics"]
    counts = state["counts"]
    img = state.get("image", {})
    return {
        "result_id": rid,
        "mode": state["mode"],
        "sort_label_ru": state["sort_label_ru"],
        "sort_code": state["sort_code"],
        "conclusion": state["conclusion"],
        "explanation": state["explanation"],
        "talc_percent": m.get("talc_percent"),
        "talc_available": m.get("talc_available", False),
        "sulfide_percent": m["sulfide_percent"],
        "ordinary_percent": m["ordinary_percent"],
        "thin_percent": m["thin_percent"],
        "grain_count": counts["total_k"],
        "grains": [GrainSchema(**g) for g in state["grains"]],
        "counts": CountsSchema(**counts),
        "metrics": MetricsSchema(
            sulfide_percent=m["sulfide_percent"],
            ordinary_percent=m["ordinary_percent"],
            thin_percent=m["thin_percent"],
            talc_percent=m.get("talc_percent"),
            talc_available=m.get("talc_available", False),
            grain_count=counts["total_k"],
            ordinary_count=counts["ordinary_l"],
            thin_count=counts["thin_j"],
            uncertain_count=counts.get("uncertain", 0),
            false_positive_count=counts.get("false_positive", 0),
        ),
        "classifier_match": state.get("classifier_match"),
        "overlay_url": f"/overlay/{rid}",
        "image_url": f"/result/{rid}/image/original",
        "talc_layer_url": f"/result/{rid}/layer/talc",
        "talc_display_url": f"/result/{rid}/layer/talc-colored",
        "type_layer_url": f"/result/{rid}/layer/type",
        "labels_url": f"/result/{rid}/labels.json",
        "csv_url": f"/result/{rid}/csv",
        "pdf_url": f"/result/{rid}/pdf",
        "original_width": img.get("original_width", 0),
        "original_height": img.get("original_height", 0),
    }


def analyze_upload(file_bytes: bytes, filename: str, mode_hint: str | None = None) -> AnalysisResponse:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Некорректный файл изображения")

    original_height, original_width = bgr.shape[:2]
    result_id = str(uuid.uuid4())[:8]

    upload_path = UPLOAD_DIR / f"{result_id}_{_safe_filename(filename)}"
    upload_path.write_bytes(file_bytes)

    image_rgb = load_image_from_bytes(file_bytes)
    ph, pw = image_rgb.shape[:2]
    scale_x = original_width / pw
    scale_y = original_height / ph

    # Downscaled overview cached for fast redraws on the interactive
    # corrections path — avoids re-decoding/re-encoding the full-res
    # original (up to 10000x10000) on every grain edit.
    _save_view_jpg(image_rgb, RESULTS_DIR / f"{result_id}_overview_view.jpg")

    report = analyzer.analyze(image_rgb, original_width, original_height, mode_hint=mode_hint)

    grains_orig = scale_grains_to_original(report.grains, scale_x, scale_y)

    talc_mask_orig: np.ndarray | None = None
    if report.talc_mask is not None:
        talc_mask_orig = _upscale_mask(report.talc_mask, original_width, original_height)
        save_talc_layer_png(result_id, talc_mask_orig)

    # Слои для PDF и превью (на original размере)
    overview_bgr = bgr
    talc_layer_rgb = None
    type_layer_rgb = None
    if talc_mask_orig is not None:
        overview_rgb = cv2.cvtColor(overview_bgr, cv2.COLOR_BGR2RGB)
        talc_layer_rgb = draw_talc_layer(overview_rgb, talc_mask_orig)
        type_layer_rgb = draw_type_layer(overview_rgb, grains_orig)
        save_overlay(talc_layer_rgb, str(RESULTS_DIR / f"{result_id}_talc_layer.jpg"))
        save_overlay(type_layer_rgb, str(RESULTS_DIR / f"{result_id}_type_layer.jpg"))
        _save_view_jpg(talc_layer_rgb, RESULTS_DIR / f"{result_id}_talc_view.jpg")
        _save_view_jpg(type_layer_rgb, RESULTS_DIR / f"{result_id}_type_view.jpg")

    if report.overlay_rgb is not None:
        overlay_orig = draw_type_layer(
            cv2.cvtColor(overview_bgr, cv2.COLOR_BGR2RGB),
            grains_orig,
        )
        if talc_mask_orig is not None:
            overlay_orig = draw_talc_layer(overlay_orig, talc_mask_orig, alpha=0.35)
        save_overlay(overlay_orig, str(RESULTS_DIR / f"{result_id}_overlay.jpg"))

    state: dict[str, Any] = {
        "result_id": result_id,
        "filename": filename,
        "upload_path": str(upload_path),
        "mode": report.mode,
        "classifier_match": report.classifier_match,
        "talc_available": report.talc_available,
        "talc_percent": report.talc_percent,
        "grains": grains_orig,
        "image": {
            "original_width": original_width,
            "original_height": original_height,
            "processed_width": pw,
            "processed_height": ph,
            "scale_x": scale_x,
            "scale_y": scale_y,
        },
    }
    state = recalculate_state(state)
    save_state(state)

    report_metrics = ReportMetrics(
        sulfide_percent=state["metrics"]["sulfide_percent"],
        ordinary_percent=state["metrics"]["ordinary_percent"],
        thin_percent=state["metrics"]["thin_percent"],
        talc_percent=state["metrics"].get("talc_percent"),
        talc_available=state["metrics"]["talc_available"],
        grain_count=state["counts"]["total_k"],
        sort_label_ru=state["sort_label_ru"],
        sort_code=state["sort_code"],
        mode=state["mode"],
    )
    csv_path = RESULTS_DIR / f"{result_id}.csv"
    csv_path.write_text(metrics_to_csv(report_metrics), encoding="utf-8")

    labels_path = RESULTS_DIR / f"{result_id}_labels.json"
    labels_path.write_text(
        json.dumps({"grains": state["grains"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return AnalysisResponse(**_state_to_response(state))


def _scaled_bbox_for_view(bbox: list[int], scale_x: float, scale_y: float) -> list[int]:
    x, y, w, h = bbox
    return [
        int(round(x / scale_x)),
        int(round(y / scale_y)),
        max(1, int(round(w / scale_x))),
        max(1, int(round(h / scale_y))),
    ]


def _refresh_type_view(state: dict[str, Any]) -> None:
    """
    Быстрая перерисовка превью (interactive-путь).

    Рисует bbox на закэшированном downscaled overview вместо полноразмерного
    оригинала — иначе правка одного зерна на панораме 10000x10000 занимает
    секунды на decode/redraw/encode.
    """
    result_id = state["result_id"]
    view_path = RESULTS_DIR / f"{result_id}_overview_view.jpg"
    if not view_path.is_file():
        return
    bgr = cv2.imread(str(view_path))
    if bgr is None:
        return
    overview_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    img = state.get("image", {})
    scale_x = float(img.get("scale_x") or 1.0)
    scale_y = float(img.get("scale_y") or 1.0)
    view_grains = [
        {
            "status": g.get("status"),
            "intergrowth_type": g.get("intergrowth_type"),
            "bbox": _scaled_bbox_for_view(g["bbox"], scale_x, scale_y),
        }
        for g in state["grains"]
    ]
    type_layer = draw_type_layer(overview_rgb, view_grains)
    save_overlay(type_layer, str(RESULTS_DIR / f"{result_id}_type_view.jpg"))


def apply_grain_corrections(result_id: str, updates: list[dict[str, Any]]) -> CorrectionsResponse | None:
    state = load_state(result_id)
    if state is None:
        return None

    state = apply_corrections(state, updates)
    save_state(state)
    _refresh_type_view(state)

    # Полноразмерные слои (для PDF/`/overlay`) перерисовываются лениво, при
    # запросе экспорта — не здесь, чтобы кнопка "Сохранить" не ждала обработку
    # оригинала в полном разрешении.

    labels_path = RESULTS_DIR / f"{result_id}_labels.json"
    labels_path.write_text(
        json.dumps({"grains": state["grains"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    resp = _state_to_response(state)
    return CorrectionsResponse(
        result_id=resp["result_id"],
        sort_label_ru=resp["sort_label_ru"],
        sort_code=resp["sort_code"],
        conclusion=resp["conclusion"],
        explanation=resp["explanation"],
        counts=resp["counts"],
        metrics=resp["metrics"],
        grains=resp["grains"],
    )


def _load_original_rgb(state: dict[str, Any]) -> np.ndarray | None:
    upload_path = Path(state["upload_path"])
    if not upload_path.is_file():
        return None
    bgr = cv2.imdecode(np.fromfile(str(upload_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None


def get_overlay_path(result_id: str) -> Path | None:
    """
    Полноразмерный overlay (тип + тальк) — рендерится по требованию из
    актуального state, а не на каждой правке зерна (см. apply_grain_corrections).
    """
    path = RESULTS_DIR / f"{result_id}_overlay.jpg"
    state = load_state(result_id)
    if state is None:
        return path if path.is_file() else None

    overview_rgb = _load_original_rgb(state)
    if overview_rgb is None:
        return path if path.is_file() else None

    overlay = draw_type_layer(overview_rgb, state["grains"])
    talc_path = RESULTS_DIR / f"{result_id}_talc.png"
    if talc_path.is_file():
        talc_mask = cv2.imread(str(talc_path), cv2.IMREAD_GRAYSCALE)
        if talc_mask is not None:
            overlay = draw_talc_layer(overlay, talc_mask, alpha=0.35)

    save_overlay(overlay, str(path))
    return path


def get_original_image_path(result_id: str) -> Path | None:
    state = load_state(result_id)
    if state is None:
        return None
    path = Path(state["upload_path"])
    return path if path.is_file() else None


def get_talc_layer_path(result_id: str) -> Path | None:
    png = RESULTS_DIR / f"{result_id}_talc.png"
    if png.is_file():
        return png
    jpg = RESULTS_DIR / f"{result_id}_talc_layer.jpg"
    return jpg if jpg.is_file() else None


def _save_view_jpg(image_rgb: np.ndarray, path: Path, max_side: int = 4096) -> None:
    """Сохраняет JPEG для web-viewer (downscale если огромный)."""
    h, w = image_rgb.shape[:2]
    out = image_rgb
    if max(h, w) > max_side:
        scale = max_side / max(h, w)
        out = cv2.resize(image_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    save_overlay(out, str(path))


def get_talc_colored_path(result_id: str) -> Path | None:
    view = RESULTS_DIR / f"{result_id}_talc_view.jpg"
    if view.is_file():
        return view
    path = RESULTS_DIR / f"{result_id}_talc_layer.jpg"
    return path if path.is_file() else None


def get_type_layer_path(result_id: str) -> Path | None:
    view = RESULTS_DIR / f"{result_id}_type_view.jpg"
    if view.is_file():
        return view
    path = RESULTS_DIR / f"{result_id}_type_layer.jpg"
    return path if path.is_file() else None


def get_labels_json(result_id: str) -> str | None:
    path = RESULTS_DIR / f"{result_id}_labels.json"
    if path.is_file():
        return path.read_text(encoding="utf-8")
    state = load_state(result_id)
    if state is None:
        return None
    return json.dumps({"grains": state["grains"]}, ensure_ascii=False, indent=2)


def get_csv_content(result_id: str) -> str | None:
    path = RESULTS_DIR / f"{result_id}.csv"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8")


def get_pdf_bytes(result_id: str) -> bytes | None:
    state = load_state(result_id)
    if state is None:
        return None

    m = state["metrics"]
    report_metrics = ReportMetrics(
        sulfide_percent=m["sulfide_percent"],
        ordinary_percent=m["ordinary_percent"],
        thin_percent=m["thin_percent"],
        talc_percent=m.get("talc_percent"),
        talc_available=m.get("talc_available", False),
        grain_count=state["counts"]["total_k"],
        sort_label_ru=state["sort_label_ru"],
        sort_code=state["sort_code"],
        mode=state["mode"],
    )

    def _load_jpg(name: str) -> np.ndarray | None:
        p = RESULTS_DIR / f"{result_id}_{name}.jpg"
        if not p.is_file():
            return None
        bgr = cv2.imread(str(p))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None

    overview_rgb = _load_original_rgb(state)
    # type_layer.jpg на диске — снимок на момент анализа; тип-слой рисуем
    # заново, чтобы PDF отражал правки зёрен, внесённые после анализа.
    type_layer_rgb = draw_type_layer(overview_rgb, state["grains"]) if overview_rgb is not None else None

    return build_pdf_bytes(
        metrics=report_metrics,
        conclusion=state["conclusion"],
        explanation=state.get("explanation", ""),
        overview_rgb=overview_rgb,
        talc_layer_rgb=_load_jpg("talc_layer"),
        type_layer_rgb=type_layer_rgb,
        counts=state.get("counts"),
    )
