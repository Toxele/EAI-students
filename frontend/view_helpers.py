"""
Вспомогательные функции UI Streamlit: zoom/pan MVP и карточка зерна.

Zoom: без тяжёлых зависимостей — numpy crop + слайдеры масштаба и центра.
Карточка зерна: selectbox по id (топ-N по площади для панорам с тысячами blob).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st
from numpy.typing import NDArray

# Доступные уровни увеличения (1 = весь кадр)
ZOOM_LEVELS = [1, 2, 4, 8]

# Сколько крупнейших зёрен показывать в selectbox (панорама может иметь 2000+)
TOP_GRAINS_LIMIT = 200

# Перевод типа сraстания для UI
INTERGROWTH_RU = {
    "ordinary": "рядовое",
    "thin": "тонкое",
}


def crop_zoom_view(
    image_rgb: NDArray[np.uint8],
    zoom: int,
    center_x_pct: float,
    center_y_pct: float,
) -> NDArray[np.uint8]:
    """
    Вырезает прямоугольную область вокруг центра с «увеличением».

    zoom=1 — весь кадр; zoom=4 — видна 1/4 ширины и высоты (центральный crop).

    :param image_rgb: исходное RGB-изображение
    :param zoom: уровень увеличения из ZOOM_LEVELS
    :param center_x_pct: горизонтальный центр crop, 0–100%
    :param center_y_pct: вертикальный центр crop, 0–100%
    :returns: вырезанный фрагмент
    """
    if zoom <= 1:
        return image_rgb

    h, w = image_rgb.shape[:2]
    # Доля кадра, видимая при данном zoom
    frac = 1.0 / zoom
    view_w = max(int(w * frac), 1)
    view_h = max(int(h * frac), 1)

    cx = int(w * center_x_pct / 100.0)
    cy = int(h * center_y_pct / 100.0)
    x0 = max(0, min(cx - view_w // 2, w - view_w))
    y0 = max(0, min(cy - view_h // 2, h - view_h))

    return image_rgb[y0 : y0 + view_h, x0 : x0 + view_w]


def top_grains_by_area(grains: list[dict[str, Any]], limit: int = TOP_GRAINS_LIMIT) -> list[dict[str, Any]]:
    """Возвращает top-N зёрен по площади (для selectbox на панораме)."""
    sorted_grains = sorted(grains, key=lambda g: g["area"], reverse=True)
    return sorted_grains[:limit]


def intergrowth_label_ru(intergrowth_type: str) -> str:
    """Переводит ordinary/thin в русские подписи."""
    return INTERGROWTH_RU.get(intergrowth_type, intergrowth_type)


def render_zoom_controls(image_rgb: NDArray[np.uint8], label: str) -> None:
    """
    Рисует блок zoom/pan: слайдеры + увеличенный crop.

    MVP без drag на canvas — пользователь двигает центр слайдерами X/Y.
    """
    st.markdown(f"**{label}** — zoom/pan (MVP)")

    c1, c2, c3 = st.columns(3)
    with c1:
        zoom = st.selectbox(
            "Масштаб",
            options=ZOOM_LEVELS,
            format_func=lambda z: f"{z}×" if z > 1 else "1× (весь кадр)",
            key=f"zoom_{label}",
        )
    with c2:
        center_x = st.slider("Центр X, %", 0, 100, 50, key=f"cx_{label}")
    with c3:
        center_y = st.slider("Центр Y, %", 0, 100, 50, key=f"cy_{label}")

    view = crop_zoom_view(image_rgb, zoom, center_x, center_y)
    st.image(view, use_container_width=True, caption=f"{label} — {zoom}×")


def render_grain_selector(grains: list[dict[str, Any]], total_count: int) -> None:
    """
    Selectbox по id зерна + карточка с bbox, area, тип сraстания, gray_ratio.

    Для панорам с большим числом blob — только top-N по площади.
    """
    if not grains:
        st.info("Зёра не найдены.")
        return

    candidates = top_grains_by_area(grains)
    if total_count > len(candidates):
        st.caption(
            f"Показаны {len(candidates)} крупнейших из {total_count} зёрен "
            f"(лимит selectbox — {TOP_GRAINS_LIMIT})."
        )

    # Подписи для selectbox: id + площадь + тип
    def _option_label(g: dict[str, Any]) -> str:
        ig = intergrowth_label_ru(g["intergrowth_type"])
        return f"#{g['id']} — area={g['area']} px, {ig}"

    selected = st.selectbox(
        "Выберите зерно",
        options=candidates,
        format_func=_option_label,
        key="grain_select",
    )

    if selected is None:
        return

    x, y, w, h = selected["bbox"]
    ig_ru = intergrowth_label_ru(selected["intergrowth_type"])

    st.markdown("#### Карточка зерна")
    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("ID", selected["id"])
        st.metric("Площадь", f"{selected['area']} px")
    with col_b:
        st.metric("Тип сraстания", ig_ru)
        st.metric("gray_ratio", f"{selected['gray_ratio']:.3f}")

    st.text(f"bbox (x, y, w, h): ({x}, {y}, {w}, {h})")
