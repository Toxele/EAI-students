# Nornickel Ore Analyzer (Web App)

Автоматическая классификация руд по OM-фото полированных шлифов.

Три сорта: **рядовая**, **труднообогатимая**, **оталькованная**.

## Быстрый старт

```bash
pip install -r requirements.txt

# Backend
uvicorn app.main:app --reload --port 8000

# Frontend (React + OpenSeadragon)
cd web && npm install && npm run dev
# → http://127.0.0.1:5173

# Альтернатива: Streamlit
streamlit run frontend/app.py
```

Windows: `.\scripts\start.ps1` — поднимает API и UI.

## Smoke-test

```bash
py scripts/smoke_test.py
```

## API

| Метод | Путь | Описание |
|-------|------|----------|
| GET | `/health` | Проверка сервера |
| POST | `/analyze` | Загрузка (`file`, `mode=panorama\|detail`) |
| POST | `/result/{id}/corrections` | Правки зёрен → пересчёт k/l/j |
| GET | `/result/{id}/image/original` | Полное изображение |
| GET | `/result/{id}/layer/talc-colored` | Слой талька |
| GET | `/result/{id}/labels.json` | Метки зёрен |
| GET | `/result/{id}/pdf` | PDF-отчёт |

## Структура app/

```
app/           FastAPI + pipeline + stub-модели
web/           React UI (OpenSeadragon, Mantine)
frontend/      Streamlit MVP
scripts/       smoke_test, start.ps1, extract_talc_masks, …
```

См. `AGENTS.md`, `PLAN.md`.
