"""Pydantic API schemas matching the backend wire contract."""

from __future__ import annotations

from typing import Literal
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
    device_name: str
    torch_version: str
    mps_fallback: bool
    weights_available: bool


class EmbeddingSetSummary(BaseModel):
    id: str
    label: str
    has_patch_embeddings: bool
    has_offset_summary: bool
    patch_grid_shape: list[int]
    patch_size: int
    il_patch_start: list[int]
    xl_patch_start: list[int]
    t_patch_start: list[int]
    inline_center: list[float]
    xline_center: list[float]
    time_center: list[float]
    embedded_offsets: list[float]
    available_modes: list[str]
    available_attributes: list[str]


class SurveySummary(BaseModel):
    id: str
    name: str
    sgz_path: str
    is_4d: bool
    shape: list[int]
    ilines: list[int]
    xlines: list[int]
    samples: list[float]
    offsets: list[float]
    default_offset_index: int
    amplitude_clip_default: float
    embedding_sets: list[EmbeddingSetSummary] = Field(default_factory=list)


class SimilarityRequest(BaseModel):
    embedding_set_id: str
    il_idx: int
    xl_idx: int
    sample_idx: int
    offset_idx: int = 0
    mode: Literal[
        "active_offset",
        "zero_offset",
        "offset_direction",
        "offset_curvature",
        "combined_trajectory",
    ] = "active_offset"
    whiten: bool = False


class PatchCenter(BaseModel):
    il: float
    xl: float
    sample: float


class PickInfo(BaseModel):
    il_idx: int
    xl_idx: int
    sample_idx: int
    offset_idx: int
    patch_idx: list[int]
    patch_center: PatchCenter
    embedded_offset_used: float | None = None


class TopMatch(BaseModel):
    patch_idx: list[int]
    il_idx: int
    xl_idx: int
    sample_idx: int
    il: int
    xl: int
    sample: float
    score: float


class TrajectoryInfo(BaseModel):
    available: bool
    offsets: list[float] = Field(default_factory=list)
    c1: list[float] = Field(default_factory=list)
    c2: list[float] = Field(default_factory=list)
    slope: float | None = None
    r2: float | None = None
    axis_var: list[float] | None = None


class SimilarityResponse(BaseModel):
    pick: PickInfo
    mode: str
    whiten: bool
    grid_shape: list[int]
    patch_size: int
    il_patch_start: list[int]
    xl_patch_start: list[int]
    t_patch_start: list[int]
    similarity_min: float
    similarity_max: float
    similarity_b64: str
    top_matches: list[TopMatch]
    trajectory: TrajectoryInfo


class AttributeResponse(BaseModel):
    attribute: str
    grid_shape: list[int]
    patch_size: int
    il_patch_start: list[int]
    xl_patch_start: list[int]
    t_patch_start: list[int]
    value_min: float
    value_max: float
    values_b64: str


class JobStatus(BaseModel):
    id: str
    job_type: str
    survey_id: str
    status: Literal["queued", "running", "completed", "failed", "cancelled", "unsupported"]
    progress_pct: float = 0.0
    message: str = ""
    logs_tail: list[str] = Field(default_factory=list)
    created_at: str
    finished_at: str | None = None
    output_path: str | None = None


class DemoEmbeddingsJobRequest(BaseModel):
    survey_id: str
    tag: str = "demo"
    offset_stride: int = Field(default=5, ge=1, le=50)


class EmbedJobRequest(BaseModel):
    survey_id: str
    tag: str = "ncs3d"
    offsets: str | None = "0:100:10"
    num_overlap_patches: int = Field(default=0, ge=0, le=13)
    batch_size: int = Field(default=4, ge=1, le=64)
    auto_summarize: bool = True


class SummarizeJobRequest(BaseModel):
    survey_id: str
    tag: str = "ncs3d"
    exclude_zero_offset: bool = False


class PreprocessJobRequest(BaseModel):
    survey_id: str
    params: dict[str, object] = Field(default_factory=dict)
