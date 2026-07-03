"""Pydantic-схемы ответов API."""

from pydantic import BaseModel, Field


class GrainSchema(BaseModel):
    id: int
    bbox: list[int]
    area: int
    intergrowth_type: str
    gray_ratio: float
    status: str = "ordinary"
    conf_ordinary: float = 0.5
    conf_thin: float = 0.5


class GrainUpdateSchema(BaseModel):
    id: int
    status: str | None = None
    bbox: list[int] | None = None


class CountsSchema(BaseModel):
    total_k: int
    ordinary_l: int
    thin_j: int
    uncertain: int = 0
    false_positive: int = 0


class MetricsSchema(BaseModel):
    """Таблица метрик из постановки задачи."""

    sulfide_percent: float
    ordinary_percent: float
    thin_percent: float
    talc_percent: float | None = None
    talc_available: bool
    grain_count: int
    ordinary_count: int = 0
    thin_count: int = 0
    uncertain_count: int = 0
    false_positive_count: int = 0


class AnalysisResponse(BaseModel):
    result_id: str
    mode: str
    sort_label_ru: str
    sort_code: str
    conclusion: str = Field(description="Краткое заключение на русском")
    explanation: str
    talc_percent: float | None
    talc_available: bool
    sulfide_percent: float
    ordinary_percent: float
    thin_percent: float
    grain_count: int
    grains: list[GrainSchema]
    counts: CountsSchema
    metrics: MetricsSchema
    classifier_match: str | None = None
    overlay_url: str | None = None
    image_url: str | None = None
    talc_layer_url: str | None = None
    talc_display_url: str | None = None
    type_layer_url: str | None = None
    labels_url: str | None = None
    csv_url: str | None = None
    pdf_url: str | None = None
    original_width: int = 0
    original_height: int = 0


class CorrectionsRequest(BaseModel):
    grains: list[GrainUpdateSchema]


class CorrectionsResponse(BaseModel):
    result_id: str
    sort_label_ru: str
    sort_code: str
    conclusion: str
    explanation: str
    counts: CountsSchema
    metrics: MetricsSchema
    grains: list[GrainSchema]
