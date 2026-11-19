"""Unit tests for JobManager cancellation (no SGZ data or weights needed)."""

from __future__ import annotations

import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import jobs as jobs_mod
from app.config import Settings
from app.jobs import JobManager


class _SeismicStub:
    def __init__(self, sgz_path: Path) -> None:
        self.survey = SimpleNamespace(sgz_path=sgz_path, is_4d=True)

    def get_survey(self, survey_id: str):
        return self.survey


class _LatentStub:
    def __init__(self, patch_path: Path) -> None:
        self.patch_path = patch_path

    def release_set(self, survey_id: str, tag: str) -> None:
        pass

    def get_set(self, survey_id: str, tag: str):
        return SimpleNamespace(patch_path=self.patch_path)


@pytest.fixture
def manager(tmp_path: Path):
    (tmp_path / "embedders").mkdir()
    settings = Settings(
        data_dir=tmp_path,
        embeddings_dir=tmp_path,
        weights_dir=tmp_path,
        embedders_dir=tmp_path / "embedders",
        preprocessing_script=tmp_path / "preprocess.py",
    )
    rescans: list[int] = []
    mgr = JobManager(
        settings,
        _SeismicStub(tmp_path / "survey.sgz"),
        _LatentStub(tmp_path / "s_ncs3d_patch_embeddings.npz"),
        on_rescan=lambda: rescans.append(1),
    )
    mgr.rescans = rescans  # type: ignore[attr-defined]
    return mgr


def _wait_terminal(mgr: JobManager, job_id: str, timeout: float = 10.0) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        status = mgr.get_job(job_id).status
        if status in ("completed", "failed", "cancelled", "unsupported"):
            return status
        time.sleep(0.02)
    raise AssertionError(f"job {job_id} still {mgr.get_job(job_id).status} after {timeout}s")


@pytest.fixture
def deferred_threads(monkeypatch):
    """Capture worker threads instead of starting them, so a test can cancel first."""
    pending: list[threading.Thread] = []
    monkeypatch.setattr(jobs_mod.threading.Thread, "start", lambda self: pending.append(self))
    return pending


@pytest.fixture
def popen_calls(monkeypatch):
    calls: list[list[str]] = []
    real_popen = jobs_mod.subprocess.Popen

    def _spy(cmd, *args, **kwargs):
        calls.append(list(cmd))
        return real_popen(cmd, *args, **kwargs)

    monkeypatch.setattr(jobs_mod.subprocess, "Popen", _spy)
    return calls


def test_cancel_stops_running_demo_job_and_is_not_overwritten(manager, monkeypatch):
    started = threading.Event()
    steps_after_cancel: list[int] = []
    cancelled = threading.Event()

    def fake_generate(progress_callback, **kwargs):
        started.set()
        for i in range(200):
            progress_callback(float(i) / 2, f"step {i}")
            if cancelled.is_set():
                steps_after_cancel.append(i)
            time.sleep(0.01)
        return Path("patch.npz"), Path("summary.npz")

    monkeypatch.setattr(jobs_mod, "generate_demo_embeddings", fake_generate)
    job = manager.start_demo_embeddings("s")
    assert started.wait(5)
    manager.cancel_job(job.id)
    cancelled.set()

    assert _wait_terminal(manager, job.id) == "cancelled"
    time.sleep(0.2)  # give a non-stopping worker time to flip the status back
    assert manager.get_job(job.id).status == "cancelled"
    assert len(steps_after_cancel) <= 1, "generator kept running after cancel"
    assert manager.rescans == []


def test_cancel_before_worker_starts_never_launches_preprocess(manager, deferred_threads, popen_calls):
    manager.settings.preprocessing_script.write_text("import time; time.sleep(30)\n")
    job = manager.start_preprocess("s")
    manager.cancel_job(job.id)
    deferred_threads[0].run()  # worker runs only now, after the cancel

    assert manager.get_job(job.id).status == "cancelled"
    assert popen_calls == []


def test_cancel_before_worker_starts_never_launches_summarizer(manager, deferred_threads, popen_calls):
    job = manager.start_summarize("s")
    manager.cancel_job(job.id)
    deferred_threads[0].run()

    assert manager.get_job(job.id).status == "cancelled"
    assert popen_calls == []


def test_cancel_before_worker_starts_never_launches_embedder(manager, deferred_threads, popen_calls, monkeypatch):
    monkeypatch.setattr(Settings, "weights_available", lambda self: True)
    job = manager.start_embed("s")
    manager.cancel_job(job.id)
    deferred_threads[0].run()

    assert manager.get_job(job.id).status == "cancelled"
    assert popen_calls == []


def test_cancel_terminates_running_subprocess(manager, popen_calls):
    manager.settings.preprocessing_script.write_text("import time; time.sleep(30)\n")
    job = manager.start_preprocess("s")
    deadline = time.monotonic() + 5
    while not popen_calls and time.monotonic() < deadline:
        time.sleep(0.02)
    assert popen_calls, "subprocess never started"

    t0 = time.monotonic()
    manager.cancel_job(job.id)
    assert _wait_terminal(manager, job.id) == "cancelled"
    assert time.monotonic() - t0 < 5
    assert manager.rescans == []


def test_uncancelled_demo_job_still_completes(manager, monkeypatch):
    def fake_generate(progress_callback, **kwargs):
        progress_callback(50.0, "half")
        return Path("patch.npz"), Path("summary.npz")

    monkeypatch.setattr(jobs_mod, "generate_demo_embeddings", fake_generate)
    job = manager.start_demo_embeddings("s")
    assert _wait_terminal(manager, job.id) == "completed"
    assert manager.rescans == [1]


def test_python_executable_is_used(manager):
    # guard: preprocess command runs with the backend's interpreter
    cmd = manager.preprocessing.build_command(Path("x.sgz"), {})
    assert cmd[0] == sys.executable
