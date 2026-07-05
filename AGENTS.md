# AGENTS.md — Nornickel Ore Analyzer

Инструкции для AI-агентов, работающих над проектом.

## Что это за проект

Автоматическая классификация руд по OM-фото полированных шлифов (кейс Nornickel).
Три сорта: **рядовая**, **труднообогатимая**, **оталькованная**.

## Правила кода (обязательно)

1. **Простота** — без лишних fallback-ов и абстракций. Один понятный путь выполнения.
2. **Комментарии** — у каждой функции docstring + комментарии к нетривиальным шагам.
3. **Модульность** — один файл = одна ответственность. Не монолит.
4. **Заглушки явные** — модели в `app/models/` помечены Stub/Simulation. Не притворяться, что это production ML.
5. **Не трогать** `data/` при runtime — для обучения используй `dataset/` (см. `py scripts/build_dataset.py`).

## Структура проекта

```
app/
  config.py              # пути, пороги
  main.py                # FastAPI entry
  models/                # 3 заглушки моделей
  pipeline/              # loader → analyze → rule_engine → report
  api/                   # HTTP routes + schemas
frontend/
  app.py                 # Streamlit UI
scripts/                 # build_dataset.py, validate_talc_masks.py
data/                    # исходник (можно удалить после build_dataset)
dataset/                 # зеркало + classification + index
task/                    # постановка задачи
```

## Три модели (текущий план)

| Модель | Файл | Режим | Сейчас делает |
|--------|------|-------|---------------|
| Детектор зёрен | `panorama_grain_detector.py` | Панорама | Порог яркости + connected components |
| Классификатор | `classifier_stub.py` | Детальный OM | Сравнение с фото из `data/` по хешу |
| Сегментатор | `segmentation_stub.py` | Детальный OM | Возвращает вход как «маску» |

## Что НЕ модель, но обязательно

- `mode_detector.py` — панорама (>50 Mpx) vs детальный снимок
- `rule_engine.py` — экспертное правило: talc>10% → оталькованная, иначе по срастаниям
- `report.py` — текст, метрики, overlay для UI

## Два режима анализа

**Панорама:** зёрна (сульфиды) + рядовое/тонкое по форме blob. Тальк — н/д или позже patch-classifier.

**Детальный OM:** сегментатор + классификатор + rule_engine с % талька.

## Как запускать

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
streamlit run frontend/app.py
```

## Что делать агенту дальше

См. `PLAN.md` — пошаговый план. Текущая фаза: **скелет + stub-модели + UI**.
