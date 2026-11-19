"""Thread-safe SGZ survey reader pool and survey metadata catalog."""

from __future__ import annotations

import logging
import queue
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from seismic_zfp.read import SgzReader

log = logging.getLogger(__name__)


@dataclass
class SurveyHandle:
    survey_id: str
    name: str
    sgz_path: Path
    is_4d: bool
    shape: list[int]
    ilines: list[int]
    xlines: list[int]
    samples: list[float]
    offsets: list[float]
    default_offset_index: int
    amplitude_clip_default: float
    pool_size: int = 4
    _pool: queue.LifoQueue[SgzReader] = field(init=False, repr=False)
    _sem: threading.BoundedSemaphore = field(init=False, repr=False)
    _lock: threading.Lock = field(init=False, repr=False)
    _all_readers: list[SgzReader] = field(default_factory=list, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def __post_init__(self) -> None:
        self._pool = queue.LifoQueue(maxsize=self.pool_size)
        self._sem = threading.BoundedSemaphore(self.pool_size)
        self._lock = threading.Lock()

    def seed_reader(self, reader: SgzReader) -> None:
        with self._lock:
            self._all_readers.append(reader)
        self._pool.put_nowait(reader)

    @contextmanager
    def acquire(self) -> Iterator[SgzReader]:
        if self._closed:
            raise RuntimeError(f"SurveyHandle {self.survey_id} is closed")
        self._sem.acquire()
        reader: SgzReader | None = None
        try:
            try:
                reader = self._pool.get_nowait()
            except queue.Empty:
                reader = SgzReader(str(self.sgz_path))
                with self._lock:
                    if self._closed:
                        reader.close()
                        raise RuntimeError(f"SurveyHandle {self.survey_id} is closed")
                    self._all_readers.append(reader)
            yield reader
        finally:
            if reader is not None:
                if self._closed:
                    try:
                        reader.close()
                    except Exception:
                        pass
                else:
                    self._pool.put_nowait(reader)
            self._sem.release()

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            readers = list(self._all_readers)
            self._all_readers.clear()
        while not self._pool.empty():
            try:
                self._pool.get_nowait()
            except queue.Empty:
                break
        for reader in readers:
            try:
                reader.close()
            except Exception as exc:
                log.warning("Error closing SgzReader for %s: %s", self.survey_id, exc)


def _pick_default_offset_index(offsets: np.ndarray) -> int:
    if len(offsets) == 0:
        return 0
    exact_zero = np.flatnonzero(offsets == 0)
    if exact_zero.size:
        return int(exact_zero[0])
    # Break +/- ties toward the non-negative side (e.g. +50 over -50 -> index 50 on Volve).
    abs_vals = np.abs(offsets).astype(np.float64)
    min_abs = abs_vals.min()
    candidates = np.flatnonzero(abs_vals == min_abs)
    pos_candidates = [int(i) for i in candidates if offsets[i] >= 0]
    return pos_candidates[0] if pos_candidates else int(candidates[0])


def _estimate_amplitude_clip(reader: SgzReader, default_offset_idx: int) -> float:
    mid_il = reader.n_ilines // 2
    if reader.is_4d:
        section = reader.read_subvolume_4d(
            mid_il,
            mid_il + 1,
            0,
            reader.n_xlines,
            default_offset_idx,
            default_offset_idx + 1,
            0,
            reader.n_samples,
        )[0, :, 0, :]
    else:
        section = reader.read_inline(mid_il)
    live = section[section != 0]
    if live.size == 0:
        return 1.0
    clip = float(np.percentile(np.abs(live), 99.0))
    return clip if clip > 0.0 and np.isfinite(clip) else 1.0


class SeismicStore:
    """Catalog and thread-safe slice/gather access for all .sgz surveys in DATA_DIR."""

    def __init__(self, data_dir: Path, pool_size: int = 4) -> None:
        self.data_dir = Path(data_dir)
        self.pool_size = max(1, pool_size)
        self._lock = threading.RLock()
        self._surveys: dict[str, SurveyHandle] = {}

    def scan(self) -> dict[str, SurveyHandle]:
        new_surveys: dict[str, SurveyHandle] = {}
        if self.data_dir.is_dir():
            sgz_files = sorted(self.data_dir.glob("*.sgz"), key=lambda p: p.name.lower())
        else:
            sgz_files = []

        for sgz_path in sgz_files:
            survey_id = sgz_path.stem
            try:
                reader = SgzReader(str(sgz_path))
                if not (reader.is_4d or reader.is_3d):
                    reader.close()
                    log.info("Skipping 2D SGZ file %s", sgz_path)
                    continue
                offsets_arr = np.asarray(reader.offsets, dtype=np.float64) if reader.is_4d else np.empty(0)
                default_offset_idx = _pick_default_offset_index(offsets_arr) if reader.is_4d else 0
                clip = _estimate_amplitude_clip(reader, default_offset_idx)
                shape = (
                    [int(reader.n_ilines), int(reader.n_xlines), int(reader.n_samples), int(reader.n_offsets)]
                    if reader.is_4d
                    else [int(reader.n_ilines), int(reader.n_xlines), int(reader.n_samples)]
                )
                handle = SurveyHandle(
                    survey_id=survey_id,
                    name=survey_id,
                    sgz_path=sgz_path.resolve(),
                    is_4d=bool(reader.is_4d),
                    shape=shape,
                    ilines=[int(x) for x in reader.ilines],
                    xlines=[int(x) for x in reader.xlines],
                    samples=[float(x) for x in reader.zslices],
                    offsets=[float(x) for x in offsets_arr],
                    default_offset_index=default_offset_idx,
                    amplitude_clip_default=clip,
                    pool_size=self.pool_size,
                )
                handle.seed_reader(reader)
                new_surveys[survey_id] = handle
            except Exception as exc:
                log.exception("Failed to index SGZ file %s: %s", sgz_path, exc)

        with self._lock:
            old_surveys = self._surveys
            self._surveys = new_surveys
        for old in old_surveys.values():
            old.close()
        return new_surveys

    def list_surveys(self) -> list[SurveyHandle]:
        with self._lock:
            return list(self._surveys.values())

    def get_survey(self, survey_id: str) -> SurveyHandle:
        with self._lock:
            if survey_id not in self._surveys:
                raise KeyError(f"Unknown survey '{survey_id}'")
            return self._surveys[survey_id]

    def read_section(
        self, survey_id: str, axis: str, index: int, offset_index: int = 0
    ) -> tuple[np.ndarray, float | int, float]:
        survey = self.get_survey(survey_id)
        n_il, n_xl, n_t = survey.shape[0], survey.shape[1], survey.shape[2]

        if axis == "inline":
            if not 0 <= index < n_il:
                raise IndexError(f"inline index {index} out of range [0, {n_il - 1}]")
            header_val: float | int = survey.ilines[index]
        elif axis == "xline":
            if not 0 <= index < n_xl:
                raise IndexError(f"xline index {index} out of range [0, {n_xl - 1}]")
            header_val = survey.xlines[index]
        elif axis == "sample":
            if not 0 <= index < n_t:
                raise IndexError(f"sample index {index} out of range [0, {n_t - 1}]")
            header_val = survey.samples[index]
        else:
            raise ValueError(f"Invalid axis '{axis}'; expected 'inline', 'xline', or 'sample'")

        if survey.is_4d:
            n_off = survey.shape[3]
            if not 0 <= offset_index < n_off:
                raise IndexError(f"offset_index {offset_index} out of range [0, {n_off - 1}]")
            offset_val = float(survey.offsets[offset_index])
            with survey.acquire() as reader:
                if axis == "inline":
                    arr = reader.read_subvolume_4d(
                        index, index + 1, 0, n_xl, offset_index, offset_index + 1, 0, n_t
                    )[0, :, 0, :]
                elif axis == "xline":
                    arr = reader.read_subvolume_4d(
                        0, n_il, index, index + 1, offset_index, offset_index + 1, 0, n_t
                    )[:, 0, 0, :]
                else:
                    arr = reader.read_subvolume_4d(
                        0, n_il, 0, n_xl, offset_index, offset_index + 1, index, index + 1
                    )[:, :, 0, 0]
        else:
            offset_val = 0.0
            with survey.acquire() as reader:
                if axis == "inline":
                    arr = reader.read_inline(index)
                elif axis == "xline":
                    arr = reader.read_crossline(index)
                else:
                    arr = reader.read_zslice(index)

        return np.ascontiguousarray(arr, dtype="<f4"), header_val, offset_val

    def read_gather_panel(
        self,
        survey_id: str,
        line_axis: str,
        il_start_idx: int,
        xl_start_idx: int,
        increment: int,
        count: int,
    ) -> tuple[np.ndarray, list[dict[str, int]]]:
        survey = self.get_survey(survey_id)
        if not survey.is_4d:
            raise ValueError(f"Survey '{survey_id}' is 3D; gather-panel requires a 4D pre-stack survey")
        if line_axis not in ("xline", "inline"):
            raise ValueError(f"Invalid line_axis '{line_axis}'; expected 'xline' or 'inline'")
        if not 1 <= count <= 64:
            raise ValueError(f"count must be in [1, 64], got {count}")
        if increment == 0:
            raise ValueError("increment must be non-zero")

        n_il, n_xl, n_t, n_off = survey.shape
        coords: list[tuple[int, int]] = []
        for k in range(count):
            il_idx = il_start_idx + (k * increment if line_axis == "inline" else 0)
            xl_idx = xl_start_idx + (k * increment if line_axis == "xline" else 0)
            if 0 <= il_idx < n_il and 0 <= xl_idx < n_xl:
                coords.append((il_idx, xl_idx))
            else:
                break

        if not coords:
            raise IndexError(
                f"No valid gather coordinates starting at ({il_start_idx}, {xl_start_idx}) "
                f"within grid ({n_il}, {n_xl})"
            )

        gathers = np.empty((len(coords), n_off, n_t), dtype="<f4")
        locations: list[dict[str, int]] = []
        with survey.acquire() as reader:
            for idx, (il_idx, xl_idx) in enumerate(coords):
                gathers[idx] = reader.read_gather(il_idx, xl_idx)
                locations.append(
                    {
                        "il_idx": il_idx,
                        "xl_idx": xl_idx,
                        "il": survey.ilines[il_idx],
                        "xl": survey.xlines[xl_idx],
                    }
                )
        return np.ascontiguousarray(gathers, dtype="<f4"), locations

    def close(self) -> None:
        with self._lock:
            surveys = list(self._surveys.values())
            self._surveys.clear()
        for survey in surveys:
            survey.close()
