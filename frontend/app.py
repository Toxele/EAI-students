"""

Streamlit UI — загрузка фото и показ результата.



Запуск: streamlit run frontend/app.py



Вызывает pipeline напрямую (без обязательного FastAPI).

"""

from __future__ import annotations



import sys

from pathlib import Path



import cv2

import numpy as np

import streamlit as st



# Корень проекта в PYTHONPATH

ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT))



from app.config import MAX_PROCESS_SIDE, TALC_PERCENT_THRESHOLD

from app.pipeline.analyzer import Analyzer

from app.pipeline.loader import load_image_from_bytes

from app.pipeline.pdf_report import build_pdf_bytes

from app.pipeline.report import ReportMetrics, metrics_to_csv

from frontend.view_helpers import render_grain_selector, render_zoom_controls



st.set_page_config(page_title="Nornickel Ore Analyzer", layout="wide")

st.title("Nornickel — классификация руд по шлифам")

st.caption("Stub-версия: 3 симуляции моделей. Панорамы >50 Mpx обрабатываются с downscale до MAX_PROCESS_SIDE.")



uploaded = st.file_uploader(

    "Загрузите OM-фото или панораму",

    type=["jpg", "jpeg", "png", "tif", "tiff"],

)



if uploaded is None:

    st.info("Загрузите изображение для анализа.")

    st.stop()



file_bytes = uploaded.read()



# Оригинальный размер (до downscale)

arr = np.frombuffer(file_bytes, dtype=np.uint8)

bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)

if bgr is None:

    st.error("Не удалось прочитать изображение.")

    st.stop()



orig_h, orig_w = bgr.shape[:2]

image_rgb = load_image_from_bytes(file_bytes)



if "analyzer" not in st.session_state:

    with st.spinner("Инициализация моделей (индекс эталонов ch1)..."):

        st.session_state.analyzer = Analyzer()



with st.spinner("Анализ..."):

    report = st.session_state.analyzer.analyze(image_rgb, orig_w, orig_h)



# --- Режим и disclaimer ---

mode_label = "панорама" if report.mode == "panorama" else "детальный OM"

st.markdown(f"**Режим:** {mode_label} ({orig_w}×{orig_h} px)")



if report.mode == "panorama":

    st.warning(

        f"Режим панорамы: изображение уменьшено до max {MAX_PROCESS_SIDE}px для обработки. "

        f"Тальк **не оценивается** на панораме (н/д). Для оценки талька используйте детальный OM-снимок."

    )



col1, col2 = st.columns(2)



with col1:

    st.subheader("Исходное (после downscale)")

    st.image(image_rgb, use_container_width=True)



with col2:

    st.subheader("Overlay")

    if report.overlay_rgb is not None:

        st.image(report.overlay_rgb, use_container_width=True)

    st.caption("🟢 рядовое сraстание | 🔴 тонкое | 🔵 тальк (если доступен)")



# --- Zoom/pan MVP (numpy crop + слайдеры, без тяжёлых deps) ---

with st.expander("🔍 Zoom / pan на overlay", expanded=report.mode == "panorama"):

    if report.overlay_rgb is not None:

        render_zoom_controls(report.overlay_rgb, label="Overlay")

    else:

        st.info("Overlay недоступен.")



# --- Заключение ---

st.subheader("Заключение")

st.markdown(report.conclusion)

with st.expander("Подробности rule engine"):

    st.markdown(report.explanation)



# --- Метрики ---

st.subheader("Метрики")

m1, m2, m3, m4 = st.columns(4)

m1.metric("Сульфиды %", f"{report.sulfide_percent:.2f}")

m2.metric("Рядовые %", f"{report.ordinary_percent:.1f}")

m3.metric("Тонкие %", f"{report.thin_percent:.1f}")

if report.talc_available:

    m4.metric("Тальк %", f"{report.talc_percent or 0:.1f}")

else:

    m4.metric("Тальк", "н/д")



st.caption(f"Порог оталькованности: {TALC_PERCENT_THRESHOLD:.0f}%")



if report.classifier_match:

    st.text(f"Stub-классификатор: ближайший эталон — {report.classifier_match}")



# --- Экспорт CSV и PDF ---

report_metrics = ReportMetrics(

    sulfide_percent=round(report.sulfide_percent, 2),

    ordinary_percent=round(report.ordinary_percent, 2),

    thin_percent=round(report.thin_percent, 2),

    talc_percent=round(report.talc_percent, 2) if report.talc_percent is not None else None,

    talc_available=report.talc_available,

    grain_count=report.grain_count,

    sort_label_ru=report.sort_label_ru,

    sort_code=report.sort_code,

    mode=report.mode,

)

csv_data = metrics_to_csv(report_metrics)

pdf_data = build_pdf_bytes(

    metrics=report_metrics,

    conclusion=report.conclusion,

    explanation=report.explanation,

    overlay_rgb=report.overlay_rgb,

)



dl_col1, dl_col2 = st.columns(2)

with dl_col1:

    st.download_button(

        label="Скачать метрики (CSV)",

        data=csv_data.encode("utf-8"),

        file_name=f"analysis_{report.sort_code}.csv",

        mime="text/csv",

    )

with dl_col2:

    st.download_button(

        label="Скачать PDF",

        data=pdf_data,

        file_name=f"analysis_{report.sort_code}.pdf",

        mime="application/pdf",

    )



# --- Карточка зерна (selectbox по id, top-N по площади) ---

st.subheader("Зёра")

render_grain_selector(report.grains, total_count=report.grain_count)



with st.expander(f"JSON: первые 20 из {report.grain_count} зёрен"):

    st.json(report.grains[:20])

