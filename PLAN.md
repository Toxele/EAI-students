# PLAN.md — план реализации сайта



## Фаза 0 — сейчас (скелет)



- [x] Очистка временных файлов

- [x] AGENTS.md + правила Cursor

- [x] 3 stub-модели

- [x] Pipeline + FastAPI + Streamlit

- [x] Прогнать demo на 1 панораме + 1 ch1 фото



## Фаза 1 — backend



- [x] Загрузка TIFF/PNG/JPEG, сохранение в `uploads/`

- [x] Tiler для панорам (sliding window) — пока downscale целиком (документировано в README)

- [x] API: `POST /analyze`, `GET /overlay/{id}`, `GET /result/{id}/csv`

- [x] CSV export

- [x] PDF export



## Фаза 2 — stub → baseline ML



- [x] Парсер синих контуров → маски талька (`scripts/extract_talc_masks.py`)

- [ ] Сегментатор: U-Net на масках

- [ ] Классификатор: CNN на ch1/ch2

- [ ] Детектор зёрен: улучшить эвристику или YOLO на blob



## Фаза 3 — frontend polish



- [x] Zoom/pan на панораме

- [x] Клик по blob → карточка

- [x] Disclaimer когда talc=n/a

- [ ] Demo video для жюри



## Архитектура потока данных



```

Upload image

    ↓

mode_detector (panorama | detail)

    ↓

┌─────────────────┬──────────────────────┐

│ PANORAMA        │ DETAIL               │

│ grain_detector  │ segmentation_stub    │

│                 │ classifier_stub      │

└────────┬────────┴──────────┬───────────┘

         ↓                   ↓

         rule_engine (talc threshold 10%)

         ↓

         report (metrics + text + overlay paths)

         ↓

         frontend

```



## Экспертное правило (не менять без согласования)



```

if talc_percent > 10:

    sort = "оталькованная"

elif ordinary_intergrowth_percent > thin_intergrowth_percent:

    sort = "рядовая"

else:

    sort = "труднообогатимая"

```



На панораме без talc: сорт только по сraстаниям + флаг `talc_available=false`.



## Заметки по архитектуре (ответ на «ничего не упустил?»)



- **rule_engine** — отдельный модуль, не модель

- **mode_detector** — авто по размеру картинки

- **Серые пятна** на панораме — часть grain_detector (замещение), не сегментатор

- **Классификатор по data/** — только для detail; для панорамы — grain_detector

- Позже: patch-voting для talc на панораме (фаза 2)

