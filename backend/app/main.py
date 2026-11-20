"""FastAPI application entrypoint for pre-stack seismic foundation model exploration."""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from .config import Settings, settings as default_settings  # noqa: E402
from .device import get_device_info  # noqa: E402
from .horizon_store import HorizonStore  # noqa: E402
from .jobs import JobConflictError, JobManager  # noqa: E402
from .latent_store import LatentStore  # noqa: E402
from .models import (  # noqa: E402
    AttributeResponse,
    DemoEmbeddingsJobRequest,
    EmbedJobRequest,
    EmbeddingSetSummary,
    FocusTrajectoryResponse,
    HealthResponse,
    HorizonsResponse,
    JobStatus,
    PreprocessJobRequest,
    SimilarityRequest,
    SimilarityResponse,
    SummarizeJobRequest,
    SurveySummary,
)
from .seismic_store import SeismicStore  # noqa: E402


class AppState:
    def __init__(self, cfg: Settings) -> None:
        self.settings = cfg
        self.seismic_store = SeismicStore(cfg.data_dir, pool_size=cfg.sgz_pool_size)
        self.latent_store = LatentStore(cfg.data_dir, cfg.embeddings_dir)
        self.horizon_store = HorizonStore(cfg.data_dir)
        self.job_manager = JobManager(cfg, self.seismic_store, self.latent_store, self.rescan)

    def rescan(self) -> list[SurveySummary]:
        surveys = self.seismic_store.scan()
        self.latent_store.scan(list(surveys.keys()))
        self.horizon_store.scan(list(surveys.keys()))
        return self.list_surveys()

    def list_surveys(self) -> list[SurveySummary]:
        result: list[SurveySummary] = []
        for s in self.seismic_store.list_surveys():
            emb_sets = [
                EmbeddingSetSummary(
                    id=e.id,
                    label=e.label,
                    has_patch_embeddings=e.patch_path is not None,
                    has_offset_summary=e.summary_path is not None,
                    patch_grid_shape=e.patch_grid_shape,
                    patch_size=e.patch_size,
                    il_patch_start=[int(x) for x in e.il_patch_start],
                    xl_patch_start=[int(x) for x in e.xl_patch_start],
                    t_patch_start=[int(x) for x in e.t_patch_start],
                    inline_center=[float(x) for x in e.inline_center],
                    xline_center=[float(x) for x in e.xline_center],
                    time_center=[float(x) for x in e.time_center],
                    embedded_offsets=[float(x) for x in e.embedded_offsets],
                    available_modes=e.available_modes,
                    available_attributes=e.available_attributes,
                )
                for e in self.latent_store.list_for_survey(s.survey_id)
            ]
            result.append(
                SurveySummary(
                    id=s.survey_id,
                    name=s.name,
                    sgz_path=str(s.sgz_path),
                    is_4d=s.is_4d,
                    shape=s.shape,
                    ilines=s.ilines,
                    xlines=s.xlines,
                    samples=s.samples,
                    offsets=s.offsets,
                    default_offset_index=s.default_offset_index,
                    amplitude_clip_default=s.amplitude_clip_default,
                    embedding_sets=emb_sets,
                    horizon_ids=self.horizon_store.horizon_ids(s.survey_id),
                )
            )
        return result

    def close(self) -> None:
        self.job_manager.shutdown()
        self.latent_store.close()
        self.seismic_store.close()


def create_app(cfg: Settings | None = None) -> FastAPI:
    active_settings = cfg or default_settings
    state = AppState(active_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        state.rescan()
        try:
            yield
        finally:
            state.close()

    app = FastAPI(
        title="Pre-Stack Seismic Foundation Model Backend",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.backend = state

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Shape", "X-Axis", "X-Header-Value", "X-Offset-Value", "X-Locations"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        info = get_device_info()
        return HealthResponse(
            status="ok",
            device=str(info["device"]),
            device_name=str(info["device_name"]),
            torch_version=str(info["torch_version"]),
            mps_fallback=bool(info["mps_fallback"]),
            weights_available=state.settings.weights_available(),
        )

    @app.get("/api/surveys", response_model=list[SurveySummary])
    def get_surveys() -> list[SurveySummary]:
        return state.list_surveys()

    @app.post("/api/surveys/rescan", response_model=list[SurveySummary])
    def rescan_surveys() -> list[SurveySummary]:
        return state.rescan()

    @app.get("/api/surveys/{survey_id}/section")
    def get_section(
        survey_id: str,
        axis: Literal["inline", "xline", "sample"] = Query(...),
        index: int = Query(..., ge=0),
        offset_index: int = Query(0, ge=0),
    ) -> Response:
        try:
            arr, header_val, offset_val = state.seismic_store.read_section(
                survey_id, axis=axis, index=index, offset_index=offset_index
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (IndexError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return Response(
            content=arr.astype("<f4", copy=False).tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Shape": f"{arr.shape[0]},{arr.shape[1]}",
                "X-Axis": axis,
                "X-Header-Value": str(header_val),
                "X-Offset-Value": str(offset_val),
            },
        )

    @app.get("/api/surveys/{survey_id}/gather-panel")
    def get_gather_panel(
        survey_id: str,
        line_axis: Literal["xline", "inline"] = Query(...),
        il_start_idx: int = Query(..., ge=0),
        xl_start_idx: int = Query(..., ge=0),
        increment: int = Query(1),
        count: int = Query(16, ge=1, le=64),
    ) -> Response:
        try:
            gathers, locations = state.seismic_store.read_gather_panel(
                survey_id=survey_id,
                line_axis=line_axis,
                il_start_idx=il_start_idx,
                xl_start_idx=xl_start_idx,
                increment=increment,
                count=count,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (IndexError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return Response(
            content=gathers.astype("<f4", copy=False).tobytes(),
            media_type="application/octet-stream",
            headers={
                "X-Shape": f"{gathers.shape[0]},{gathers.shape[1]},{gathers.shape[2]}",
                "X-Locations": json.dumps(locations),
            },
        )

    @app.post("/api/surveys/{survey_id}/similarity", response_model=SimilarityResponse)
    def post_similarity(survey_id: str, body: SimilarityRequest) -> SimilarityResponse:
        try:
            survey = state.seismic_store.get_survey(survey_id)
            res = state.latent_store.compute_similarity(
                survey=survey,
                embedding_set_id=body.embedding_set_id,
                il_idx=body.il_idx,
                xl_idx=body.xl_idx,
                sample_idx=body.sample_idx,
                offset_idx=body.offset_idx,
                mode=body.mode,
                whiten=body.whiten,
                features=body.features,
                exclusion_weight=body.exclusion_weight,
                focus_pick=body.focus_pick,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (IndexError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return SimilarityResponse(**res)

    @app.get("/api/surveys/{survey_id}/trajectory", response_model=FocusTrajectoryResponse)
    def get_trajectory(
        survey_id: str,
        embedding_set_id: str = Query(...),
        il_idx: int = Query(..., ge=0),
        xl_idx: int = Query(..., ge=0),
        sample_idx: int = Query(..., ge=0),
        offset_idx: int = Query(0, ge=0),
    ) -> FocusTrajectoryResponse:
        try:
            survey = state.seismic_store.get_survey(survey_id)
            res = state.latent_store.compute_trajectory(
                survey=survey,
                embedding_set_id=embedding_set_id,
                il_idx=il_idx,
                xl_idx=xl_idx,
                sample_idx=sample_idx,
                offset_idx=offset_idx,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (IndexError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return FocusTrajectoryResponse(**res)

    @app.get("/api/surveys/{survey_id}/attribute", response_model=AttributeResponse)
    def get_attribute(
        survey_id: str,
        embedding_set_id: str = Query(...),
        attribute: str = Query(...),
    ) -> AttributeResponse:
        try:
            state.seismic_store.get_survey(survey_id)
            res = state.latent_store.get_attribute(
                survey_id=survey_id,
                embedding_set_id=embedding_set_id,
                attribute=attribute,
            )
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return AttributeResponse(**res)

    @app.get("/api/surveys/{survey_id}/horizons", response_model=HorizonsResponse)
    def get_horizons(survey_id: str) -> HorizonsResponse:
        try:
            survey = state.seismic_store.get_survey(survey_id)
            payload = state.horizon_store.get_payload(survey_id, survey.shape)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return HorizonsResponse(**payload)

    @app.get("/api/jobs", response_model=list[JobStatus])
    def list_jobs() -> list[JobStatus]:
        return state.job_manager.list_jobs()

    @app.post("/api/jobs/demo-embeddings", response_model=JobStatus)
    def trigger_demo_embeddings(body: DemoEmbeddingsJobRequest) -> JobStatus:
        try:
            return state.job_manager.start_demo_embeddings(
                survey_id=body.survey_id,
                tag=body.tag,
                offset_stride=body.offset_stride,
            )
        except JobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/embed", response_model=JobStatus)
    def trigger_embed(body: EmbedJobRequest) -> JobStatus:
        try:
            return state.job_manager.start_embed(
                survey_id=body.survey_id,
                tag=body.tag,
                offsets=body.offsets,
                num_overlap_patches=body.num_overlap_patches,
                batch_size=body.batch_size,
                auto_summarize=body.auto_summarize,
            )
        except JobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/summarize", response_model=JobStatus)
    def trigger_summarize(body: SummarizeJobRequest) -> JobStatus:
        try:
            return state.job_manager.start_summarize(
                survey_id=body.survey_id,
                tag=body.tag,
                exclude_zero_offset=body.exclude_zero_offset,
            )
        except JobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/preprocess", response_model=JobStatus)
    def trigger_preprocess(body: PreprocessJobRequest) -> JobStatus:
        try:
            return state.job_manager.start_preprocess(
                survey_id=body.survey_id,
                params=body.params,
            )
        except JobConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except (ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @app.post("/api/jobs/{job_id}/cancel", response_model=JobStatus)
    def cancel_job(job_id: str) -> JobStatus:
        try:
            return state.job_manager.cancel_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    return app


app = create_app()
