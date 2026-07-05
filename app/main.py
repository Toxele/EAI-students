"""
FastAPI entry point.

Запуск: uvicorn app.main:app --reload --port 8000
"""
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse, Response

from app.api.routes import (
    analyze_upload,
    apply_grain_corrections,
    apply_talc_mask_edit,
    get_csv_content,
    get_labels_json,
    get_original_image_path,
    get_overlay_path,
    get_pdf_bytes,
    get_talc_colored_path,
    get_talc_confidence_path,
    get_talc_layer_path,
    get_type_layer_path,
)
from app.api.schemas import AnalysisResponse, CorrectionsRequest, CorrectionsResponse

app = FastAPI(title="Nornickel Ore Analyzer", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    file: UploadFile = File(...),
    mode: str | None = Form(default=None),
) -> AnalysisResponse:
    """
    Загрузить OM-фото или панораму и получить классификацию.

    mode: 'panorama' | 'detail' | None (auto)
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Нужен файл")

    mode_hint = mode if mode in ("panorama", "detail") else None
    data = await file.read()
    try:
        return analyze_upload(data, file.filename, mode_hint=mode_hint)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/result/{result_id}/corrections", response_model=CorrectionsResponse)
def post_corrections(result_id: str, body: CorrectionsRequest) -> CorrectionsResponse:
    """Применить правки зёрен (класс / bbox / ложная детекция) и пересчитать метрики."""
    updates = [u.model_dump(exclude_none=True) for u in body.grains]
    result = apply_grain_corrections(result_id, updates)
    if result is None:
        raise HTTPException(status_code=404, detail="Результат не найден")
    return result


@app.post("/result/{result_id}/talc-mask", response_model=CorrectionsResponse)
async def post_talc_mask(result_id: str, mask: UploadFile = File(...)) -> CorrectionsResponse:
    """Сохранить отредактированную маску талька (карандаш/ластик/заливка) и пересчитать метрики."""
    data = await mask.read()
    try:
        result = apply_talc_mask_edit(result_id, data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Результат не найден")
    return result


@app.get("/overlay/{result_id}")
def get_overlay(result_id: str) -> FileResponse:
    path = get_overlay_path(result_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Overlay не найден")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/result/{result_id}/image/original")
def get_original_image(result_id: str) -> FileResponse:
    path = get_original_image_path(result_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Изображение не найдено")
    return FileResponse(path)


@app.get("/result/{result_id}/layer/talc-colored")
def get_talc_colored_layer(result_id: str) -> FileResponse:
    path = get_talc_colored_path(result_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Цветной слой талька не найден")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/result/{result_id}/layer/talc")
def get_talc_layer(result_id: str) -> FileResponse:
    path = get_talc_layer_path(result_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Слой талька не найден")
    media = "image/png" if path.suffix == ".png" else "image/jpeg"
    return FileResponse(path, media_type=media)


@app.get("/result/{result_id}/layer/talc-confidence")
def get_talc_confidence(result_id: str) -> FileResponse:
    path = get_talc_confidence_path(result_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Карта уверенности талька не найдена")
    return FileResponse(path, media_type="image/png")


@app.get("/result/{result_id}/layer/type")
def get_type_layer(result_id: str) -> FileResponse:
    path = get_type_layer_path(result_id)
    if path is None:
        raise HTTPException(status_code=404, detail="Слой типов не найден")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/result/{result_id}/labels.json")
def get_labels(result_id: str) -> PlainTextResponse:
    content = get_labels_json(result_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Метки не найдены")
    return PlainTextResponse(content, media_type="application/json")


@app.get("/result/{result_id}/csv")
def get_result_csv(result_id: str) -> PlainTextResponse:
    content = get_csv_content(result_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Результат не найден")
    return PlainTextResponse(content, media_type="text/csv")


@app.get("/result/{result_id}/pdf")
def get_result_pdf(result_id: str) -> Response:
    pdf_data = get_pdf_bytes(result_id)
    if pdf_data is None:
        raise HTTPException(status_code=404, detail="Результат не найден")
    return Response(
        content=pdf_data,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="report_{result_id}.pdf"'},
    )
