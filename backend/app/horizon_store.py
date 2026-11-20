"""Ground-truth horizon surfaces (TWT grids on the survey IL/XL lattice) built by scripts/build_reservoir_horizons.py."""

from __future__ import annotations

import base64
import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

HORIZONS_SUFFIX = "_horizons.npz"


@dataclass
class HorizonSurface:
    id: str
    label: str
    kind: str
    color: str
    twt: np.ndarray  # (n_il, n_xl) float32 ms, NaN where undefined


@dataclass
class HorizonSetHandle:
    survey_id: str
    path: Path
    ilines: np.ndarray
    xlines: np.ndarray
    surfaces: list[HorizonSurface]
    well_picks: list[dict]
    meta: dict
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)


def _load_handle(path: Path, survey_id: str) -> HorizonSetHandle:
    with np.load(path, allow_pickle=False) as z:
        ids = [str(x) for x in z["horizon_ids"]]
        labels = [str(x) for x in z["horizon_labels"]]
        kinds = [str(x) for x in z["horizon_kinds"]]
        colors = [str(x) for x in z["horizon_colors"]]
        twt = np.asarray(z["twt"], dtype=np.float32)
        if twt.ndim != 3 or twt.shape[0] != len(ids):
            raise ValueError(f"{path.name}: twt must be (n_horizons, n_il, n_xl), got {twt.shape}")
        well_picks = json.loads(str(z["well_picks_json"])) if "well_picks_json" in z else []
        meta = json.loads(str(z["meta_json"])) if "meta_json" in z else {}
        surfaces = [
            HorizonSurface(id=i, label=lb, kind=k, color=c, twt=np.ascontiguousarray(twt[n]))
            for n, (i, lb, k, c) in enumerate(zip(ids, labels, kinds, colors))
        ]
        return HorizonSetHandle(
            survey_id=survey_id,
            path=path,
            ilines=np.asarray(z["ilines"]),
            xlines=np.asarray(z["xlines"]),
            surfaces=surfaces,
            well_picks=well_picks,
            meta=meta,
        )


class HorizonStore:
    """Catalog of ``<survey_id>_horizons.npz`` files in DATA_DIR."""

    def __init__(self, data_dir: Path) -> None:
        self.data_dir = Path(data_dir)
        self._lock = threading.RLock()
        self._sets: dict[str, HorizonSetHandle] = {}

    def scan(self, survey_ids: list[str]) -> None:
        found: dict[str, HorizonSetHandle] = {}
        for survey_id in survey_ids:
            path = self.data_dir / f"{survey_id}{HORIZONS_SUFFIX}"
            if not path.is_file():
                continue
            try:
                found[survey_id] = _load_handle(path, survey_id)
            except Exception as exc:
                log.exception("Failed to load horizon set %s: %s", path, exc)
        with self._lock:
            self._sets = found

    def has(self, survey_id: str) -> bool:
        with self._lock:
            return survey_id in self._sets

    def horizon_ids(self, survey_id: str) -> list[str]:
        with self._lock:
            handle = self._sets.get(survey_id)
            return [s.id for s in handle.surfaces] if handle else []

    def get(self, survey_id: str) -> HorizonSetHandle:
        with self._lock:
            if survey_id not in self._sets:
                raise KeyError(f"No horizons available for survey '{survey_id}'")
            return self._sets[survey_id]

    def get_payload(self, survey_id: str, survey_shape: list[int]) -> dict:
        handle = self.get(survey_id)
        n_il, n_xl = int(survey_shape[0]), int(survey_shape[1])
        if len(handle.ilines) != n_il or len(handle.xlines) != n_xl:
            raise ValueError(
                f"Horizon grid {len(handle.ilines)}x{len(handle.xlines)} does not match survey {n_il}x{n_xl}"
            )
        horizons = []
        for s in handle.surfaces:
            finite = np.isfinite(s.twt)
            horizons.append(
                {
                    "id": s.id,
                    "label": s.label,
                    "kind": s.kind,
                    "color": s.color,
                    "grid_shape": [n_il, n_xl],
                    "twt_min": float(s.twt[finite].min()) if finite.any() else 0.0,
                    "twt_max": float(s.twt[finite].max()) if finite.any() else 0.0,
                    "coverage": float(finite.mean()),
                    "twt_b64": base64.b64encode(s.twt.astype("<f4", copy=False).tobytes()).decode("ascii"),
                }
            )
        return {
            "survey_id": survey_id,
            "horizons": horizons,
            "well_picks": handle.well_picks,
            "meta": handle.meta,
        }
