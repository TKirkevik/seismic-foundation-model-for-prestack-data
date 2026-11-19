"""Background job orchestration for demo embeddings, NCS inference, offset summarization, and preprocessing."""

from __future__ import annotations

import logging
import os
import re
import subprocess
import sys
import threading
import uuid
import zipfile
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
from scripts.generate_demo_embeddings import generate_demo_embeddings

from .config import Settings
from .device import get_device
from .latent_store import LatentStore
from .models import JobStatus
from .seismic_store import SeismicStore

log = logging.getLogger(__name__)

_PCT_RE = re.compile(r"(\d{1,3})%")


class JobConflictError(RuntimeError):
    """Raised when a conflicting active job is already queued or running."""


class _JobCancelled(Exception):
    """Internal sentinel raised when an in-process job observes cancellation."""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_tag(tag: str | None) -> str | None:
    if tag is None:
        return None
    return "default" if not tag or tag == "default" else tag


def _has_distinct_abs_offsets(patch_npz: Path) -> bool:
    with zipfile.ZipFile(patch_npz, "r") as zf:
        if "offsets.npy" not in zf.namelist():
            return False
        with zf.open("offsets.npy", "r") as fh:
            offsets = np.asarray(np.load(fh, allow_pickle=False), dtype=np.float64).reshape(-1)
    return int(np.unique(np.abs(offsets)).size) >= 2


@dataclass
class _JobRecord:
    id: str
    job_type: str
    survey_id: str
    tag: str | None = None
    status: str = "queued"
    progress_pct: float = 0.0
    message: str = ""
    logs_tail: deque[str] = field(default_factory=lambda: deque(maxlen=40))
    created_at: str = field(default_factory=_now_iso)
    finished_at: str | None = None
    output_path: str | None = None
    process: subprocess.Popen[str] | None = None
    cancel_requested: bool = False

    def to_model(self) -> JobStatus:
        return JobStatus(
            id=self.id,
            job_type=self.job_type,
            survey_id=self.survey_id,
            status=self.status,  # type: ignore[arg-type]
            progress_pct=round(float(self.progress_pct), 2),
            message=self.message,
            logs_tail=list(self.logs_tail),
            created_at=self.created_at,
            finished_at=self.finished_at,
            output_path=self.output_path,
        )


class PreprocessingAdapter:
    """Adapter stub for optional external pre-stack preprocessing scripts."""

    def __init__(self, script_path: Path) -> None:
        self.script_path = Path(script_path)

    def is_available(self) -> bool:
        return self.script_path.is_file()

    def build_command(self, survey_path: Path, params: dict[str, object]) -> list[str]:
        cmd = [sys.executable, str(self.script_path), "--input", str(survey_path)]
        for k, v in params.items():
            flag = f"--{k.replace('_', '-')}"
            if isinstance(v, bool):
                if v:
                    cmd.append(flag)
            elif v is not None:
                cmd.extend([flag, str(v)])
        return cmd


class JobManager:
    def __init__(
        self,
        settings: Settings,
        seismic_store: SeismicStore,
        latent_store: LatentStore,
        on_rescan: Callable[[], None],
    ) -> None:
        self.settings = settings
        self.seismic_store = seismic_store
        self.latent_store = latent_store
        self.on_rescan = on_rescan
        self.preprocessing = PreprocessingAdapter(settings.preprocessing_script)
        self._lock = threading.Lock()
        self._jobs: dict[str, _JobRecord] = {}

    def list_jobs(self) -> list[JobStatus]:
        with self._lock:
            records = list(self._jobs.values())
        records.sort(key=lambda r: r.created_at, reverse=True)
        return [r.to_model() for r in records]

    def get_job(self, job_id: str) -> JobStatus:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job '{job_id}'")
            return self._jobs[job_id].to_model()

    def _register_job(
        self, job_type: str, survey_id: str, message: str, tag: str | None = None
    ) -> _JobRecord:
        norm_tag = _normalize_tag(tag)
        with self._lock:
            for existing in self._jobs.values():
                if existing.status not in ("queued", "running"):
                    continue
                if job_type == "embed" and existing.job_type == "embed":
                    raise JobConflictError(
                        f"Active embed job '{existing.id}' is already running on GPU/MPS"
                    )
                if (
                    norm_tag is not None
                    and existing.survey_id == survey_id
                    and existing.tag == norm_tag
                ):
                    raise JobConflictError(
                        f"Active job '{existing.id}' ({existing.job_type}) already targets "
                        f"survey '{survey_id}' tag '{norm_tag}'"
                    )
                if (
                    job_type == "preprocess"
                    and existing.job_type == "preprocess"
                    and existing.survey_id == survey_id
                ):
                    raise JobConflictError(
                        f"Active preprocess job '{existing.id}' already targets survey '{survey_id}'"
                    )

            rec = _JobRecord(
                id=uuid.uuid4().hex[:12],
                job_type=job_type,
                survey_id=survey_id,
                tag=norm_tag,
                status="queued",
                message=message,
            )
            self._jobs[rec.id] = rec
            return rec

    def _run_subprocess(
        self,
        rec: _JobRecord,
        cmd: list[str],
        pct_start: float,
        pct_end: float,
    ) -> int:
        env = os.environ.copy()
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
        env["PYTHONUNBUFFERED"] = "1"

        with self._lock:
            if rec.cancel_requested:
                return -1

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            cwd=str(self.settings.repo_root),
        )
        cancelled_during_spawn = False
        with self._lock:
            if rec.cancel_requested:
                cancelled_during_spawn = True
            else:
                rec.process = proc

        if cancelled_during_spawn:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            return -1

        assert proc.stdout is not None
        buf: list[str] = []
        while True:
            ch = proc.stdout.read(1)
            if ch == "" and proc.poll() is not None:
                break
            if ch in ("\r", "\n"):
                line = "".join(buf).strip()
                buf.clear()
                if line:
                    with self._lock:
                        rec.logs_tail.append(line)
                        if "Cube statistics" not in line:
                            m = _PCT_RE.search(line)
                            if m:
                                pct = min(100, max(0, int(m.group(1))))
                                rec.progress_pct = pct_start + (pct_end - pct_start) * (pct / 100.0)
                        rec.message = line[:160]
            elif ch:
                buf.append(ch)
        tail = "".join(buf).strip()
        if tail:
            with self._lock:
                rec.logs_tail.append(tail)

        rc = proc.wait()
        with self._lock:
            rec.process = None
        return rc

    def start_demo_embeddings(self, survey_id: str, tag: str = "demo", offset_stride: int = 5) -> JobStatus:
        survey = self.seismic_store.get_survey(survey_id)
        rec = self._register_job(
            "demo-embeddings", survey_id, f"Generating demo embeddings ({tag})", tag=tag
        )
        self.latent_store.release_set(survey_id, tag)

        def _worker() -> None:
            with self._lock:
                if rec.cancel_requested:
                    rec.status = "cancelled"
                    rec.message = "Cancelled by user"
                    rec.finished_at = _now_iso()
                    return
                rec.status = "running"
                rec.progress_pct = 2.0
            try:
                def _cb(pct: float, msg: str) -> None:
                    with self._lock:
                        if rec.cancel_requested:
                            raise _JobCancelled("Cancelled by user")
                        rec.progress_pct = pct
                        rec.message = msg
                        rec.logs_tail.append(msg)

                patch_npz, summary_npz = generate_demo_embeddings(
                    sgz_path=survey.sgz_path,
                    output_dir=self.settings.data_dir,
                    tag=tag,
                    offset_stride=offset_stride,
                    force=True,
                    summarizer_script=self.settings.embedders_dir / "offset_direction_summarizer.py",
                    progress_callback=_cb,
                )
                with self._lock:
                    if rec.cancel_requested:
                        rec.status = "cancelled"
                        rec.message = "Cancelled by user"
                        if rec.finished_at is None:
                            rec.finished_at = _now_iso()
                        return
                self.on_rescan()
                with self._lock:
                    if rec.cancel_requested:
                        rec.status = "cancelled"
                        rec.message = "Cancelled by user"
                        if rec.finished_at is None:
                            rec.finished_at = _now_iso()
                        return
                    rec.status = "completed"
                    rec.progress_pct = 100.0
                    rec.message = f"Generated {patch_npz.name} and {summary_npz.name}"
                    rec.output_path = str(patch_npz)
                    rec.finished_at = _now_iso()
            except _JobCancelled:
                with self._lock:
                    rec.status = "cancelled"
                    rec.message = "Cancelled by user"
                    if rec.finished_at is None:
                        rec.finished_at = _now_iso()
            except Exception as exc:
                log.exception("Demo embeddings job %s failed", rec.id)
                with self._lock:
                    if rec.cancel_requested:
                        rec.status = "cancelled"
                        rec.message = "Cancelled by user"
                    else:
                        rec.status = "failed"
                        rec.message = str(exc)
                        rec.logs_tail.append(f"ERROR: {exc}")
                    if rec.finished_at is None:
                        rec.finished_at = _now_iso()

        threading.Thread(target=_worker, daemon=True, name=f"job-{rec.id}").start()
        return rec.to_model()

    def start_embed(
        self,
        survey_id: str,
        tag: str = "ncs3d",
        offsets: str | None = "0:100:10",
        num_overlap_patches: int = 0,
        batch_size: int = 4,
        auto_summarize: bool = True,
    ) -> JobStatus:
        survey = self.seismic_store.get_survey(survey_id)
        if not survey.is_4d:
            raise ValueError(f"Survey '{survey_id}' is not 4D; prestack_patch_embedder requires 4D SGZ")
        if not self.settings.weights_available():
            raise FileNotFoundError(
                f"Model weights missing in {self.settings.weights_dir} (need config.json & model.safetensors)"
            )

        suffix = f"_{tag}" if tag and tag != "default" else ""
        patch_out = self.settings.data_dir / f"{survey_id}{suffix}_patch_embeddings.npz"
        summary_out = self.settings.data_dir / f"{survey_id}{suffix}_offset_summary.npz"

        rec = self._register_job(
            "embed", survey_id, f"Running prestack_patch_embedder ({tag})", tag=tag
        )
        self.latent_store.release_set(survey_id, tag)

        def _worker() -> None:
            with self._lock:
                if rec.cancel_requested:
                    rec.status = "cancelled"
                    rec.message = "Cancelled by user"
                    rec.finished_at = _now_iso()
                    return
                rec.status = "running"
                rec.progress_pct = 1.0
            try:
                embed_script = self.settings.embedders_dir / "prestack_patch_embedder.py"
                device_str = get_device().type
                cmd = [
                    sys.executable,
                    str(embed_script),
                    "--input",
                    str(survey.sgz_path),
                    "--output",
                    str(patch_out),
                    "--weights",
                    str(self.settings.weights_dir),
                    "--num-overlap-patches",
                    str(num_overlap_patches),
                    "--batch-size",
                    str(batch_size),
                    "--device",
                    device_str,
                    "--force",
                ]
                if offsets:
                    cmd.extend(["--offsets", offsets])

                rc = self._run_subprocess(rec, cmd, 0.0, 85.0 if auto_summarize else 100.0)
                with self._lock:
                    if rec.cancel_requested:
                        rec.status = "cancelled"
                        rec.message = "Cancelled by user"
                        if rec.finished_at is None:
                            rec.finished_at = _now_iso()
                        return
                if rc != 0:
                    with self._lock:
                        rec.status = "failed"
                        rec.message = f"prestack_patch_embedder exited with code {rc}"
                        rec.finished_at = _now_iso()
                    return

                skipped_summary_msg: str | None = None
                if auto_summarize:
                    if not _has_distinct_abs_offsets(patch_out):
                        skipped_summary_msg = (
                            f"Completed embedding ({tag}); skipped offset summary "
                            "(need >= 2 distinct |offset| values)"
                        )
                        with self._lock:
                            rec.logs_tail.append(skipped_summary_msg)
                            rec.message = skipped_summary_msg
                    else:
                        with self._lock:
                            rec.message = "Running offset_direction_summarizer"
                        sum_script = self.settings.embedders_dir / "offset_direction_summarizer.py"
                        sum_cmd = [
                            sys.executable,
                            str(sum_script),
                            "--input",
                            str(patch_out),
                            "--output",
                            str(summary_out),
                            "--force",
                        ]
                        rc2 = self._run_subprocess(rec, sum_cmd, 85.0, 100.0)
                        with self._lock:
                            if rec.cancel_requested:
                                rec.status = "cancelled"
                                rec.message = "Cancelled by user"
                                if rec.finished_at is None:
                                    rec.finished_at = _now_iso()
                                return
                        if rc2 != 0:
                            with self._lock:
                                rec.status = "failed"
                                rec.message = f"offset_direction_summarizer exited with code {rc2}"
                                rec.finished_at = _now_iso()
                            return

                self.on_rescan()
                with self._lock:
                    if rec.cancel_requested:
                        rec.status = "cancelled"
                        rec.message = "Cancelled by user"
                        if rec.finished_at is None:
                            rec.finished_at = _now_iso()
                        return
                    rec.status = "completed"
                    rec.progress_pct = 100.0
                    rec.message = skipped_summary_msg or f"Completed embedding ({tag})"
                    rec.output_path = str(patch_out)
                    rec.finished_at = _now_iso()
            except Exception as exc:
                log.exception("Embed job %s failed", rec.id)
                with self._lock:
                    if rec.cancel_requested:
                        rec.status = "cancelled"
                        rec.message = "Cancelled by user"
                    else:
                        rec.status = "failed"
                        rec.message = str(exc)
                        rec.logs_tail.append(f"ERROR: {exc}")
                    if rec.finished_at is None:
                        rec.finished_at = _now_iso()

        threading.Thread(target=_worker, daemon=True, name=f"job-{rec.id}").start()
        return rec.to_model()

    def start_summarize(
        self, survey_id: str, tag: str = "ncs3d", exclude_zero_offset: bool = False
    ) -> JobStatus:
        self.seismic_store.get_survey(survey_id)
        handle = self.latent_store.get_set(survey_id, tag)
        if handle.patch_path is None:
            raise ValueError(f"Embedding set '{tag}' has no patch_embeddings file to summarize")

        patch_in = handle.patch_path
        suffix = f"_{tag}" if tag and tag != "default" else ""
        summary_out = patch_in.with_name(f"{survey_id}{suffix}_offset_summary.npz")

        rec = self._register_job(
            "summarize", survey_id, f"Running offset_direction_summarizer ({tag})", tag=tag
        )
        self.latent_store.release_set(survey_id, tag)

        def _worker() -> None:
            with self._lock:
                if rec.cancel_requested:
                    rec.status = "cancelled"
                    rec.message = "Cancelled by user"
                    rec.finished_at = _now_iso()
                    return
                rec.status = "running"
                rec.progress_pct = 5.0
            try:
                sum_script = self.settings.embedders_dir / "offset_direction_summarizer.py"
                cmd = [
                    sys.executable,
                    str(sum_script),
                    "--input",
                    str(patch_in),
                    "--output",
                    str(summary_out),
                    "--force",
                ]
                if exclude_zero_offset:
                    cmd.append("--exclude-zero-offset")

                rc = self._run_subprocess(rec, cmd, 5.0, 100.0)
                with self._lock:
                    if rec.cancel_requested:
                        rec.status = "cancelled"
                        rec.message = "Cancelled by user"
                        if rec.finished_at is None:
                            rec.finished_at = _now_iso()
                        return
                if rc != 0:
                    with self._lock:
                        rec.status = "failed"
                        rec.message = f"offset_direction_summarizer exited with code {rc}"
                        rec.finished_at = _now_iso()
                    return

                self.on_rescan()
                with self._lock:
                    if rec.cancel_requested:
                        rec.status = "cancelled"
                        rec.message = "Cancelled by user"
                        if rec.finished_at is None:
                            rec.finished_at = _now_iso()
                        return
                    rec.status = "completed"
                    rec.progress_pct = 100.0
                    rec.message = f"Completed offset summary ({tag})"
                    rec.output_path = str(summary_out)
                    rec.finished_at = _now_iso()
            except Exception as exc:
                log.exception("Summarize job %s failed", rec.id)
                with self._lock:
                    if rec.cancel_requested:
                        rec.status = "cancelled"
                        rec.message = "Cancelled by user"
                    else:
                        rec.status = "failed"
                        rec.message = str(exc)
                        rec.logs_tail.append(f"ERROR: {exc}")
                    if rec.finished_at is None:
                        rec.finished_at = _now_iso()

        threading.Thread(target=_worker, daemon=True, name=f"job-{rec.id}").start()
        return rec.to_model()

    def start_preprocess(self, survey_id: str, params: dict[str, object] | None = None) -> JobStatus:
        survey = self.seismic_store.get_survey(survey_id)
        if not self.preprocessing.is_available():
            rec = self._register_job(
                "preprocess",
                survey_id,
                f"Preprocessing adapter stub: {self.preprocessing.script_path} not installed yet",
            )
            with self._lock:
                rec.status = "unsupported"
                rec.finished_at = _now_iso()
            return rec.to_model()

        rec = self._register_job("preprocess", survey_id, "Running preprocessing pipeline")
        cmd = self.preprocessing.build_command(survey.sgz_path, params or {})

        def _worker() -> None:
            with self._lock:
                if rec.cancel_requested:
                    rec.status = "cancelled"
                    rec.message = "Cancelled by user"
                    rec.finished_at = _now_iso()
                    return
                rec.status = "running"
            try:
                rc = self._run_subprocess(rec, cmd, 0.0, 100.0)
                with self._lock:
                    if rec.cancel_requested:
                        rec.status = "cancelled"
                        rec.message = "Cancelled by user"
                        if rec.finished_at is None:
                            rec.finished_at = _now_iso()
                        return
                if rc != 0:
                    with self._lock:
                        rec.status = "failed"
                        rec.message = f"Preprocessing exited with code {rc}"
                        rec.finished_at = _now_iso()
                    return
                self.on_rescan()
                with self._lock:
                    if rec.cancel_requested:
                        rec.status = "cancelled"
                        rec.message = "Cancelled by user"
                        if rec.finished_at is None:
                            rec.finished_at = _now_iso()
                        return
                    rec.status = "completed"
                    rec.progress_pct = 100.0
                    rec.message = "Preprocessing completed"
                    rec.finished_at = _now_iso()
            except Exception as exc:
                with self._lock:
                    if rec.cancel_requested:
                        rec.status = "cancelled"
                        rec.message = "Cancelled by user"
                    else:
                        rec.status = "failed"
                        rec.message = str(exc)
                    if rec.finished_at is None:
                        rec.finished_at = _now_iso()

        threading.Thread(target=_worker, daemon=True, name=f"job-{rec.id}").start()
        return rec.to_model()

    def cancel_job(self, job_id: str) -> JobStatus:
        with self._lock:
            if job_id not in self._jobs:
                raise KeyError(f"Unknown job '{job_id}'")
            rec = self._jobs[job_id]
            rec.cancel_requested = True
            proc = rec.process
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        with self._lock:
            if rec.status in ("queued", "running"):
                rec.status = "cancelled"
                rec.message = "Cancelled by user"
                rec.finished_at = _now_iso()
            return rec.to_model()

    def shutdown(self) -> None:
        with self._lock:
            procs = [r.process for r in self._jobs.values() if r.process is not None and r.process.poll() is None]
        for proc in procs:
            try:
                proc.terminate()
            except Exception:
                pass
