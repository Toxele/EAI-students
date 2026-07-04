"""
Построение маски талька из синей экспертной разметки.

STUB / baseline для фазы 2.

Алгоритм:
1. Синие штрихи разметчика.
2. Концы линий у края кадра → продление до рамки (та же толщина, что у синего).
3. Единый барьер (dilate синего + красного).
4. Замкнутые области внутри контура → тальк, если внутри темнее снаружи.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from numpy.typing import NDArray

BLUE_MIN_B = 150
BLUE_MAX_R = 120
BLUE_MAX_G = 120

# Толщина штриха до dilate (одинакова для синего и красного)
STROKE_WIDTH = 3
BARRIER_DILATE = 3

# Конец линии считаем «у края», если ближе этого (px)
BORDER_NEAR_PX = 80

# Мин. площадь области-кандидата (px)
MIN_REGION_AREA = 400

# Макс. доля кадра — отсечь фоновый контур
MAX_REGION_FRACTION = 0.85


@dataclass
class TalcMaskResult:
    """Результат построения маски и слоёв для валидации."""

    talc_mask: NDArray[np.uint8]
    blue_strokes: NDArray[np.uint8]
    closure_strokes: NDArray[np.uint8]
    barrier: NDArray[np.uint8]
    talc_percent: float


def extract_blue_strokes(bgr: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Выделяет синие штрихи экспертной разметки (0/255)."""
    b, g, r = cv2.split(bgr)
    blue = ((b > BLUE_MIN_B) & (r < BLUE_MAX_R) & (g < BLUE_MAX_G)).astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    return cv2.morphologyEx(blue, cv2.MORPH_OPEN, kernel, iterations=1)


def _stroke_layer(mask: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Утолщает бинарную линию до STROKE_WIDTH."""
    if not np.any(mask):
        return mask
    layer = mask.copy()
    radius = max(1, STROKE_WIDTH // 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1))
    return cv2.dilate(layer, kernel, iterations=1)


def _skeleton(binary: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Морфологический скелет бинарной маски."""
    img = (binary > 0).astype(np.uint8)
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img) > 0:
        eroded = cv2.erode(img, element)
        temp = cv2.subtract(img, cv2.dilate(eroded, element))
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
    return skel * 255


def _neighbors8(skeleton: NDArray[np.uint8], x: int, y: int) -> list[tuple[int, int]]:
    """8-соседи точки на скелете."""
    out: list[tuple[int, int]] = []
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nx, ny = x + dx, y + dy
            if 0 <= nx < skeleton.shape[1] and 0 <= ny < skeleton.shape[0]:
                if skeleton[ny, nx] > 0:
                    out.append((nx, ny))
    return out


def _dist_to_border(x: int, y: int, w: int, h: int) -> tuple[float, tuple[int, int]]:
    """Расстояние до ближайшего края и точка на рамке."""
    candidates = [
        (float(x), (0, y)),
        (float(w - 1 - x), (w - 1, y)),
        (float(y), (x, 0)),
        (float(h - 1 - y), (x, h - 1)),
    ]
    dist, pt = min(candidates, key=lambda item: item[0])
    return dist, pt


def _points_toward_border(
    endpoint: tuple[int, int],
    neighbor: tuple[int, int],
    border_pt: tuple[int, int],
) -> bool:
    """True, если продолжение линии endpoint смотрит в сторону border_pt."""
    ex, ey = endpoint
    nx, ny = neighbor
    bx, by = border_pt
    # Направление вдоль линии: от соседа через endpoint наружу
    vx, vy = float(ex - nx), float(ey - ny)
    wx, wy = float(bx - ex), float(by - ey)
    v_len = np.hypot(vx, vy)
    w_len = np.hypot(wx, wy)
    if v_len < 1e-3 or w_len < 1e-3:
        return False
    cos_angle = (vx * wx + vy * wy) / (v_len * w_len)
    return cos_angle > 0.35


def extend_endpoints_to_border(blue: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """
    Продлевает только «висящие» концы у края кадра до рамки.

    Красный штрих — та же толщина STROKE_WIDTH, что и синий.
    """
    h, w = blue.shape[:2]
    closure = np.zeros((h, w), dtype=np.uint8)
    if not np.any(blue):
        return closure

    skel = _skeleton(blue)
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    neighbor_count = cv2.filter2D((skel > 0).astype(np.uint8), cv2.CV_8U, kernel)
    endpoints = np.argwhere((skel > 0) & (neighbor_count == 1))

    for y, x in endpoints:
        x, y = int(x), int(y)
        nbs = _neighbors8(skel, x, y)
        if len(nbs) != 1:
            continue

        dist_edge, border_pt = _dist_to_border(x, y, w, h)
        if dist_edge > BORDER_NEAR_PX:
            continue

        if not _points_toward_border((x, y), nbs[0], border_pt):
            continue

        cv2.line(closure, (x, y), border_pt, 255, thickness=STROKE_WIDTH, lineType=cv2.LINE_AA)

    return closure


def _build_barrier(blue: NDArray[np.uint8], closure: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Единый барьер: синий + красный с одинаковым dilate + рамка кадра."""
    h, w = blue.shape[:2]
    combined = cv2.bitwise_or(_stroke_layer(blue), _stroke_layer(closure))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    barrier = cv2.dilate(combined, kernel, iterations=BARRIER_DILATE)
    barrier[0, :] = 255
    barrier[-1, :] = 255
    barrier[:, 0] = 255
    barrier[:, -1] = 255
    return barrier


def _reference_brightness(gray: NDArray[np.uint8], labels: NDArray[np.int32], n_labels: int) -> float:
    """
    Опорная яркость фона: крупнейшая компонента, касающаяся края кадра.
    """
    h, w = gray.shape
    best_area = 0
    best_mean = float(np.percentile(gray, 72))
    for label in range(1, n_labels):
        mask = labels == label
        area = int(mask.sum())
        if area < 1000:
            continue
        touches = mask[0, :].any() or mask[-1, :].any() or mask[:, 0].any() or mask[:, -1].any()
        if touches and area > best_area:
            best_area = area
            best_mean = float(gray[mask].mean())
    return best_mean


def _fill_talc_from_contours(
    barrier: NDArray[np.uint8],
    gray: NDArray[np.uint8],
    annotation_strokes: NDArray[np.uint8],
) -> NDArray[np.uint8]:
    """Заливка по связным областям, примыкающим к разметке."""
    h, w = gray.shape[:2]
    total = h * w
    talc = np.zeros((h, w), dtype=np.uint8)

    free = np.where(barrier > 0, 0, 255).astype(np.uint8)
    n_labels, labels = cv2.connectedComponents(free)
    ref_l = _reference_brightness(gray, labels, n_labels)
    margin = 12.0

    stroke_touch = cv2.dilate(annotation_strokes, np.ones((5, 5), np.uint8), iterations=1)

    for label in range(1, n_labels):
        mask = labels == label
        area = int(mask.sum())
        if area < MIN_REGION_AREA or area > total * 0.55:
            continue

        dilated = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        if not np.any(dilated.astype(bool) & (stroke_touch > 0)):
            continue

        mean_l = float(gray[mask].mean())
        if mean_l < ref_l - margin:
            talc[mask] = 255

    return talc


def build_talc_mask(bgr: NDArray[np.uint8]) -> TalcMaskResult:
    """
    Строит маску талька из кадра с синей разметкой.

    :param bgr: BGR изображение (annotated)
    :return: TalcMaskResult
    """
    blue = extract_blue_strokes(bgr)
    closure = extend_endpoints_to_border(blue)
    blue_layer = _stroke_layer(blue)
    closure_layer = _stroke_layer(closure)
    annotation = cv2.bitwise_or(blue_layer, closure_layer)
    barrier = _build_barrier(blue, closure)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    talc_mask = _fill_talc_from_contours(barrier, gray, annotation)

    total = talc_mask.shape[0] * talc_mask.shape[1]
    talc_percent = 100.0 * np.count_nonzero(talc_mask) / max(total, 1)

    return TalcMaskResult(
        talc_mask=talc_mask,
        blue_strokes=blue_layer,
        closure_strokes=closure_layer,
        barrier=barrier,
        talc_percent=round(talc_percent, 2),
    )


def _add_label(panel: NDArray[np.uint8], text: str) -> NDArray[np.uint8]:
    """Подпись панели."""
    out = panel.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), thickness=-1)
    cv2.putText(out, text, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render_strokes_validation(bgr: NDArray[np.uint8], result: TalcMaskResult) -> NDArray[np.uint8]:
    """Синяя разметка + красные дорисовки (одинаковая толщина)."""
    out = bgr.copy()
    blue_px = result.blue_strokes > 0
    out[blue_px, 0] = np.clip(out[blue_px, 0].astype(np.int16) + 100, 0, 255)
    out[blue_px, 1] = (out[blue_px, 1].astype(np.int16) * 0.4).astype(np.uint8)
    out[blue_px, 2] = (out[blue_px, 2].astype(np.int16) * 0.4).astype(np.uint8)
    closure_px = result.closure_strokes > 0
    out[closure_px] = [0, 0, 255]
    return out


def render_overlay_validation(bgr: NDArray[np.uint8], result: TalcMaskResult, alpha: float = 0.45) -> NDArray[np.uint8]:
    """Полупрозрачная маска талька."""
    overlay = bgr.copy().astype(np.float32)
    talc_px = result.talc_mask > 0
    color = np.array([200, 120, 40], dtype=np.float32)
    overlay[talc_px] = (1 - alpha) * overlay[talc_px] + alpha * color
    return overlay.astype(np.uint8)


def render_binary_validation(result: TalcMaskResult) -> NDArray[np.uint8]:
    """Ч/б маска (3 канала для склейки)."""
    return cv2.cvtColor(result.talc_mask, cv2.COLOR_GRAY2BGR)


def render_combined_validation(bgr: NDArray[np.uint8], result: TalcMaskResult) -> NDArray[np.uint8]:
    """Один JPEG: strokes | overlay | binary."""
    p1 = _add_label(render_strokes_validation(bgr, result), "1 blue + red extensions")
    p2 = _add_label(render_overlay_validation(bgr, result), f"2 mask overlay {result.talc_percent}%")
    p3 = _add_label(render_binary_validation(result), "3 binary mask")
    h = max(p1.shape[0], p2.shape[0], p3.shape[0])

    def pad(img: NDArray[np.uint8]) -> NDArray[np.uint8]:
        if img.shape[0] == h:
            return img
        return cv2.copyMakeBorder(img, 0, h - img.shape[0], 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

    return np.hstack([pad(p1), pad(p2), pad(p3)])
