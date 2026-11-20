"""Zero-copy latent embedding & offset-trajectory store with GPU/MPS similarity search."""

from __future__ import annotations

import base64
import logging
import struct
import threading
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat
import torch
import torch.nn.functional as F

from .device import get_device
from .models import PickCoord, PickedFeaturePayload
from .seismic_store import SurveyHandle

log = logging.getLogger(__name__)

ALL_MODES = (
    "active_offset",
    "zero_offset",
    "offset_direction",
    "offset_curvature",
    "combined_trajectory",
)
ALL_ATTRIBUTES = (
    "offset_slope",
    "offset_fit_r2",
    "offset_axis_var_1",
    "offset_axis_var_2",
    "offset_axis_var_plane",
)


def _zero_offset_indices(offsets: np.ndarray) -> np.ndarray:
    exact = np.flatnonzero(offsets == 0)
    if exact.size:
        return exact
    neg = np.flatnonzero(offsets < 0)
    pos = np.flatnonzero(offsets > 0)
    if neg.size == 0 or pos.size == 0:
        return np.array([int(np.argmin(np.abs(offsets)))])
    return np.array([int(neg[np.argmax(offsets[neg])]), int(pos[np.argmin(offsets[pos])])])


def _read_npy_header_from_zip(
    path: Path, member: str
) -> tuple[tuple[int, ...], bool, np.dtype, int] | None:
    """Inspect an uncompressed ZIP member's .npy header and return (shape, fortran, dtype, data_offset)."""
    with zipfile.ZipFile(path, "r") as zf:
        if member not in zf.namelist():
            return None
        info = zf.getinfo(member)
        if info.compress_type != zipfile.ZIP_STORED:
            return None
        header_offset = info.header_offset

    with open(path, "rb") as fh:
        fh.seek(header_offset)
        local_hdr = fh.read(30)
        if len(local_hdr) < 30 or local_hdr[:4] != b"PK\x03\x04":
            return None
        _, _, _, _, _, _, _, _, _, fn_len, ex_len = struct.unpack("<4sHHHHHIIIHH", local_hdr)
        fh.seek(header_offset + 30 + fn_len + ex_len)
        version = npformat.read_magic(fh)
        if version == (1, 0):
            shape, fortran_order, dtype = npformat.read_array_header_1_0(fh)
        elif version in ((2, 0), (3, 0)):
            shape, fortran_order, dtype = npformat.read_array_header_2_0(fh)
        else:
            return None
        data_offset = fh.tell()
    return tuple(int(d) for d in shape), bool(fortran_order), np.dtype(dtype), int(data_offset)


def _find_patch_idx(starts: np.ndarray, patch_size: int, idx: int) -> int:
    if len(starts) == 0:
        return 0
    hits = np.flatnonzero((starts <= idx) & (idx < starts + patch_size))
    if hits.size:
        return int(hits[0])
    centers = starts.astype(np.float64) + patch_size / 2.0
    nearest = int(np.argmin(np.abs(centers - idx)))
    return max(0, min(nearest, len(starts) - 1))


def _build_patch_lut(starts: np.ndarray, patch_size: int, length: int) -> np.ndarray:
    if length <= 0:
        return np.empty(0, dtype=np.int64)
    if len(starts) == 0:
        return np.zeros(length, dtype=np.int64)
    indices = np.arange(length, dtype=np.float64)[:, None]
    starts_f = starts.astype(np.float64)[None, :]
    in_patch = (starts_f <= indices) & (indices < starts_f + patch_size)
    has_hit = in_patch.any(axis=1)
    first_hit = np.argmax(in_patch, axis=1)
    centers = starts_f + patch_size / 2.0
    nearest = np.argmin(np.abs(centers - indices), axis=1)
    lut = np.where(has_hit, first_hit, nearest).astype(np.int64)
    return np.clip(lut, 0, len(starts) - 1)


def _rasterize_segment_2d(
    h0: int,
    v0: int,
    h1: int,
    v1: int,
    h_lut: np.ndarray,
    t_lut: np.ndarray,
) -> set[tuple[int, int]]:
    steps = max(abs(h1 - h0), abs(v1 - v0), 1)
    hs = np.rint(np.linspace(h0, h1, steps + 1)).astype(np.int64)
    vs = np.rint(np.linspace(v0, v1, steps + 1)).astype(np.int64)
    return set(zip(h_lut[hs].tolist(), t_lut[vs].tolist()))


def _rasterize_feature_patches(
    feature: PickedFeaturePayload,
    n_il: int,
    n_xl: int,
    n_t: int,
    il_lut: np.ndarray,
    xl_lut: np.ndarray,
    t_lut: np.ndarray,
) -> set[tuple[int, int, int]]:
    if feature.plane_axis == "inline":
        if not 0 <= feature.plane_index < n_il:
            raise IndexError(
                f"plane_index {feature.plane_index} out of range [0, {n_il - 1}] for inline"
            )
    elif feature.plane_axis == "xline":
        if not 0 <= feature.plane_index < n_xl:
            raise IndexError(
                f"plane_index {feature.plane_index} out of range [0, {n_xl - 1}] for xline"
            )
    else:
        raise ValueError(f"Unsupported plane_axis '{feature.plane_axis}'")

    for pick in feature.picks:
        if not 0 <= pick.il_idx < n_il:
            raise IndexError(f"pick.il_idx {pick.il_idx} out of range [0, {n_il - 1}]")
        if not 0 <= pick.xl_idx < n_xl:
            raise IndexError(f"pick.xl_idx {pick.xl_idx} out of range [0, {n_xl - 1}]")
        if not 0 <= pick.sample_idx < n_t:
            raise IndexError(f"pick.sample_idx {pick.sample_idx} out of range [0, {n_t - 1}]")
        if feature.plane_axis == "inline" and pick.il_idx != feature.plane_index:
            raise ValueError(
                f"Pick il_idx={pick.il_idx} is not coplanar with inline plane_index={feature.plane_index}"
            )
        if feature.plane_axis == "xline" and pick.xl_idx != feature.plane_index:
            raise ValueError(
                f"Pick xl_idx={pick.xl_idx} is not coplanar with xline plane_index={feature.plane_index}"
            )

    if not feature.picks:
        return set()

    if feature.geometry_mode == "single":
        p = feature.picks[0]
        return {(int(il_lut[p.il_idx]), int(xl_lut[p.xl_idx]), int(t_lut[p.sample_idx]))}

    if feature.geometry_mode == "points" or len(feature.picks) == 1:
        return {
            (int(il_lut[p.il_idx]), int(xl_lut[p.xl_idx]), int(t_lut[p.sample_idx]))
            for p in feature.picks
        }

    if feature.plane_axis == "inline":
        fixed_p_il = int(il_lut[feature.plane_index])
        h_lut = xl_lut
        verts = [(p.xl_idx, p.sample_idx) for p in feature.picks]
    else:
        fixed_p_xl = int(xl_lut[feature.plane_index])
        h_lut = il_lut
        verts = [(p.il_idx, p.sample_idx) for p in feature.picks]

    patches_2d: set[tuple[int, int]] = set()
    if len(verts) == 2 or feature.geometry_mode == "polyline":
        for i in range(len(verts) - 1):
            h0, v0 = verts[i]
            h1, v1 = verts[i + 1]
            patches_2d.update(_rasterize_segment_2d(h0, v0, h1, v1, h_lut, t_lut))
    elif feature.geometry_mode == "polygon":
        k = len(verts)
        for i in range(k):
            h0, v0 = verts[i]
            h1, v1 = verts[(i + 1) % k]
            patches_2d.update(_rasterize_segment_2d(h0, v0, h1, v1, h_lut, t_lut))

        min_v = min(v for _, v in verts)
        max_v = max(v for _, v in verts)
        max_h_idx = len(h_lut) - 1
        for v in range(min_v, max_v):
            h_ints: list[float] = []
            for i in range(k):
                h0, v0 = verts[i]
                h1, v1 = verts[(i + 1) % k]
                if v0 == v1:
                    continue
                vmin, vmax = (v0, v1) if v0 < v1 else (v1, v0)
                if vmin <= v < vmax:
                    h_int = h0 + (v + 0.0 - v0) * (h1 - h0) / (v1 - v0)
                    h_ints.append(h_int)
            h_ints.sort()
            p_t = int(t_lut[v])
            for j in range(0, len(h_ints) - 1, 2):
                h_left = max(0, int(np.ceil(h_ints[j])))
                h_right = min(max_h_idx, int(np.floor(h_ints[j + 1])))
                if h_left <= h_right:
                    for p_h in np.unique(h_lut[h_left : h_right + 1]).tolist():
                        patches_2d.add((p_h, p_t))
    else:
        raise ValueError(f"Unsupported geometry_mode '{feature.geometry_mode}'")

    if feature.plane_axis == "inline":
        return {(fixed_p_il, p_h, p_t) for p_h, p_t in patches_2d}
    return {(p_h, fixed_p_xl, p_t) for p_h, p_t in patches_2d}


@dataclass
class EmbeddingSetHandle:
    id: str
    label: str
    survey_id: str
    patch_path: Path | None
    summary_path: Path | None
    patch_grid_shape: list[int]
    patch_size: int
    il_patch_start: np.ndarray
    xl_patch_start: np.ndarray
    t_patch_start: np.ndarray
    inline_center: np.ndarray
    xline_center: np.ndarray
    time_center: np.ndarray
    embedded_offsets: np.ndarray
    available_modes: list[str]
    available_attributes: list[str]
    _memmap_meta: tuple[tuple[int, ...], bool, np.dtype, int] | None = None
    _embeddings_cache: np.ndarray | None = field(default=None, init=False, repr=False)
    _summary_cache: dict[str, np.ndarray] | None = field(default=None, init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _closed: bool = field(default=False, init=False, repr=False)

    def _get_embeddings_unlocked(self) -> np.ndarray:
        if self.patch_path is None:
            raise ValueError(f"Embedding set '{self.id}' has no patch_embeddings file")
        if self._closed:
            raise RuntimeError("Embedding set is closed")
        if self._embeddings_cache is not None:
            return self._embeddings_cache
        if self._memmap_meta is not None:
            shape, fortran_order, dtype, data_offset = self._memmap_meta
            arr = np.memmap(
                self.patch_path,
                dtype=dtype,
                mode="r",
                offset=data_offset,
                shape=shape,
                order="F" if fortran_order else "C",
            )
            self._embeddings_cache = arr
            return arr
        # Compressed .npz fallback
        with np.load(self.patch_path, allow_pickle=False) as npz:
            arr = np.array(npz["embeddings"])
        self._embeddings_cache = arr
        return arr

    def read_offset_volume(self, off_idx: int) -> np.ndarray:
        with self._lock:
            if self._closed:
                raise RuntimeError("Embedding set is closed")
            embeddings = self._get_embeddings_unlocked()
            return np.array(embeddings[off_idx], dtype=np.float32, copy=True)

    def read_zero_offset_volume(self, z_idx: np.ndarray) -> np.ndarray:
        with self._lock:
            if self._closed:
                raise RuntimeError("Embedding set is closed")
            embeddings = self._get_embeddings_unlocked()
            return np.array(embeddings[z_idx], dtype=np.float32, copy=True).mean(axis=0)

    def get_summary(self) -> dict[str, np.ndarray]:
        if self.summary_path is None:
            raise ValueError(f"Embedding set '{self.id}' has no offset_summary file")
        with self._lock:
            if self._closed:
                raise RuntimeError("Embedding set is closed")
            if self._summary_cache is not None:
                return self._summary_cache
            summary: dict[str, np.ndarray] = {}
            with np.load(self.summary_path, allow_pickle=False) as npz:
                for key in (
                    "zero_offset",
                    "offset_direction",
                    "offset_curvature",
                    "offset_coords",
                    "offset_slope",
                    "offset_fit_r2",
                    "offset_axis_var",
                    "zero_offset_source",
                    "regression_offsets",
                ):
                    if key in npz.files:
                        summary[key] = np.array(npz[key])
            self._summary_cache = summary
            return summary

    def close(self) -> None:
        with self._lock:
            self._closed = True
            if isinstance(self._embeddings_cache, np.memmap):
                mmap_obj = getattr(self._embeddings_cache, "_mmap", None)
                if mmap_obj is not None:
                    try:
                        mmap_obj.close()
                    except Exception:
                        pass
            self._embeddings_cache = None
            self._summary_cache = None


def _discover_npz_pairs(
    survey_ids: list[str], directories: list[Path]
) -> dict[str, dict[str, dict[str, Path]]]:
    """Pair <survey_id>_<tag>_patch_embeddings.npz and _offset_summary.npz across directories.

    Each .npz binds only to the longest matching ``<survey_id>_`` prefix.
    """
    pairs: dict[str, dict[str, dict[str, Path]]] = {sid: {} for sid in survey_ids}
    sorted_ids = sorted(set(survey_ids), key=len, reverse=True)
    if not sorted_ids:
        return pairs

    for directory in directories:
        if not directory.is_dir():
            continue
        for npz_path in sorted(directory.glob("*.npz"), key=lambda p: p.name.lower()):
            name = npz_path.name
            matched_sid: str | None = None
            for sid in sorted_ids:
                if name.startswith(f"{sid}_"):
                    matched_sid = sid
                    break
            if matched_sid is None:
                continue
            rem = name[len(matched_sid) + 1 :]
            if rem == "patch_embeddings.npz":
                tag, kind = "default", "patch"
            elif rem.endswith("_patch_embeddings.npz"):
                tag = rem[: -len("_patch_embeddings.npz")]
                kind = "patch"
            elif rem == "offset_summary.npz":
                tag, kind = "default", "summary"
            elif rem.endswith("_offset_summary.npz"):
                tag = rem[: -len("_offset_summary.npz")]
                kind = "summary"
            else:
                continue
            if not tag:
                tag = "default"
            slot = pairs[matched_sid].setdefault(tag, {})
            slot.setdefault(kind, npz_path.resolve())
    return pairs


def _build_embedding_handle(
    survey_id: str, tag: str, patch_path: Path | None, summary_path: Path | None
) -> EmbeddingSetHandle | None:
    meta_path = patch_path or summary_path
    if meta_path is None:
        return None

    memmap_meta: tuple[tuple[int, ...], bool, np.dtype, int] | None = None
    grid_shape: list[int] | None = None
    if patch_path is not None:
        memmap_meta = _read_npy_header_from_zip(patch_path, "embeddings.npy")
        if memmap_meta is not None:
            shape = memmap_meta[0]
            if len(shape) == 5:
                grid_shape = [int(shape[1]), int(shape[2]), int(shape[3])]

    with zipfile.ZipFile(meta_path, "r") as zf:
        def _load_member(member: str, default=None):
            fname = f"{member}.npy"
            if fname not in zf.namelist():
                return default
            with zf.open(fname, "r") as fh:
                return np.load(fh, allow_pickle=False)

        il_patch_start = _load_member("il_patch_start")
        xl_patch_start = _load_member("xl_patch_start")
        t_patch_start = _load_member("t_patch_start")
        inline_center = _load_member("inline_center")
        xline_center = _load_member("xline_center")
        time_center = _load_member("time_center")
        patch_size_arr = _load_member("patch_size")
        offsets_arr = _load_member("offsets")
        if offsets_arr is None:
            offsets_arr = _load_member("regression_offsets", np.empty(0, dtype=np.float32))

    # If patch_path was compressed or missing, determine offsets/grid from patch_path or summary_path
    if offsets_arr is None or offsets_arr.size == 0:
        if patch_path is not None and patch_path != meta_path:
            with zipfile.ZipFile(patch_path, "r") as zf:
                if "offsets.npy" in zf.namelist():
                    with zf.open("offsets.npy", "r") as fh:
                        offsets_arr = np.load(fh, allow_pickle=False)
        elif summary_path is not None:
            with zipfile.ZipFile(summary_path, "r") as zf:
                if "regression_offsets.npy" in zf.namelist():
                    with zf.open("regression_offsets.npy", "r") as fh:
                        offsets_arr = np.load(fh, allow_pickle=False)

    if il_patch_start is None or xl_patch_start is None or t_patch_start is None:
        log.warning("Skipping %s (%s): missing patch start metadata", meta_path, tag)
        return None

    if grid_shape is None:
        if patch_path is not None:
            with np.load(patch_path, allow_pickle=False) as npz:
                s = npz["embeddings"].shape
                grid_shape = [int(s[1]), int(s[2]), int(s[3])]
        else:
            grid_shape = [len(il_patch_start), len(xl_patch_start), len(t_patch_start)]

    patch_size = int(np.asarray(patch_size_arr).item()) if patch_size_arr is not None else 16
    if inline_center is None:
        inline_center = il_patch_start.astype(np.float64) + patch_size // 2
    if xline_center is None:
        xline_center = xl_patch_start.astype(np.float64) + patch_size // 2
    if time_center is None:
        time_center = t_patch_start.astype(np.float64) + patch_size // 2

    has_patch = patch_path is not None
    has_summary = summary_path is not None

    modes: list[str] = []
    if has_patch:
        modes.append("active_offset")
    if has_patch or has_summary:
        modes.append("zero_offset")
    if has_summary:
        modes.extend(["offset_direction", "offset_curvature", "combined_trajectory"])

    attributes = list(ALL_ATTRIBUTES) if has_summary else []

    return EmbeddingSetHandle(
        id=tag,
        label=tag,
        survey_id=survey_id,
        patch_path=patch_path,
        summary_path=summary_path,
        patch_grid_shape=grid_shape,
        patch_size=patch_size,
        il_patch_start=np.asarray(il_patch_start, dtype=np.int64),
        xl_patch_start=np.asarray(xl_patch_start, dtype=np.int64),
        t_patch_start=np.asarray(t_patch_start, dtype=np.int64),
        inline_center=np.asarray(inline_center, dtype=np.float64),
        xline_center=np.asarray(xline_center, dtype=np.float64),
        time_center=np.asarray(time_center, dtype=np.float64),
        embedded_offsets=np.asarray(offsets_arr, dtype=np.float64).reshape(-1),
        available_modes=modes,
        available_attributes=attributes,
        _memmap_meta=memmap_meta,
    )


def _prepare_normalized_volume(
    volume_np: np.ndarray,
    whiten: bool,
    device: torch.device,
) -> torch.Tensor:
    feats = torch.as_tensor(np.array(volume_np, dtype=np.float32, copy=True), device=device)
    if whiten:
        mean = feats.mean(dim=(0, 1, 2), keepdim=True)
        std = feats.std(dim=(0, 1, 2), unbiased=False, keepdim=True)
        feats = (feats - mean) / (std + 1e-6)
    feats_norm = F.normalize(feats, p=2, dim=-1, eps=1e-8)
    return feats_norm.reshape(-1, feats_norm.shape[-1])


def _max_similarity_over_queries(
    feats_flat: torch.Tensor,
    query_flat_indices: list[int],
    chunk_size: int = 256,
) -> torch.Tensor:
    q_idx_t = torch.as_tensor(query_flat_indices, dtype=torch.long, device=feats_flat.device)
    max_sim: torch.Tensor | None = None
    for start in range(0, q_idx_t.numel(), chunk_size):
        chunk_idx = q_idx_t[start : start + chunk_size]
        sims = feats_flat @ feats_flat[chunk_idx].T
        chunk_max = sims.amax(dim=1)
        max_sim = chunk_max if max_sim is None else torch.maximum(max_sim, chunk_max)
    assert max_sim is not None
    return max_sim


def _max_combined_similarity_over_queries(
    zero_flat: torch.Tensor,
    dir_flat: torch.Tensor,
    query_flat_indices: list[int],
    chunk_size: int = 256,
) -> torch.Tensor:
    q_idx_t = torch.as_tensor(query_flat_indices, dtype=torch.long, device=zero_flat.device)
    max_sim: torch.Tensor | None = None
    for start in range(0, q_idx_t.numel(), chunk_size):
        chunk_idx = q_idx_t[start : start + chunk_size]
        sims = 0.5 * (zero_flat @ zero_flat[chunk_idx].T) + 0.5 * (
            dir_flat @ dir_flat[chunk_idx].T
        )
        chunk_max = sims.amax(dim=1)
        max_sim = chunk_max if max_sim is None else torch.maximum(max_sim, chunk_max)
    assert max_sim is not None
    return max_sim


class LatentStore:
    """Catalog and GPU/MPS similarity engine for pre-stack patch embeddings & summaries."""

    def __init__(self, data_dir: Path, embeddings_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir)
        self.embeddings_dir = Path(embeddings_dir) if embeddings_dir is not None else None
        self._lock = threading.RLock()
        self._sets: dict[str, dict[str, EmbeddingSetHandle]] = {}

    def _search_dirs(self) -> list[Path]:
        dirs = [self.data_dir]
        if self.embeddings_dir is not None and self.embeddings_dir != self.data_dir:
            dirs.append(self.embeddings_dir)
        return dirs

    def scan(self, survey_ids: list[str]) -> None:
        search_dirs = self._search_dirs()
        all_pairs = _discover_npz_pairs(survey_ids, search_dirs)
        new_sets: dict[str, dict[str, EmbeddingSetHandle]] = {}
        for survey_id in survey_ids:
            pairs = all_pairs.get(survey_id, {})
            survey_sets: dict[str, EmbeddingSetHandle] = {}
            for tag, paths in pairs.items():
                try:
                    handle = _build_embedding_handle(
                        survey_id=survey_id,
                        tag=tag,
                        patch_path=paths.get("patch"),
                        summary_path=paths.get("summary"),
                    )
                    if handle is not None:
                        survey_sets[tag] = handle
                except Exception as exc:
                    log.exception("Failed to load embedding set '%s' for %s: %s", tag, survey_id, exc)
            new_sets[survey_id] = survey_sets

        with self._lock:
            old_sets = self._sets
            self._sets = new_sets
        for survey_map in old_sets.values():
            for handle in survey_map.values():
                handle.close()

    def list_for_survey(self, survey_id: str) -> list[EmbeddingSetHandle]:
        with self._lock:
            return list(self._sets.get(survey_id, {}).values())

    def get_set(self, survey_id: str, embedding_set_id: str) -> EmbeddingSetHandle:
        with self._lock:
            survey_map = self._sets.get(survey_id, {})
            if embedding_set_id not in survey_map:
                raise KeyError(
                    f"Unknown embedding_set_id '{embedding_set_id}' for survey '{survey_id}'"
                )
            return survey_map[embedding_set_id]

    def _build_pick_and_trajectory(
        self,
        handle: EmbeddingSetHandle,
        survey: SurveyHandle,
        il_idx: int,
        xl_idx: int,
        sample_idx: int,
        offset_idx: int,
        p_il: int,
        p_xl: int,
        p_t: int,
    ) -> tuple[dict[str, object], dict[str, object]]:
        embedded_offset_used: float | None = None
        if handle.embedded_offsets.size > 0:
            nearest_emb_off_idx = 0
            if survey.is_4d:
                target_off = survey.offsets[offset_idx]
                nearest_emb_off_idx = int(np.argmin(np.abs(handle.embedded_offsets - target_off)))
            embedded_offset_used = float(handle.embedded_offsets[nearest_emb_off_idx])

        pick_info: dict[str, object] = {
            "il_idx": il_idx,
            "xl_idx": xl_idx,
            "sample_idx": sample_idx,
            "offset_idx": offset_idx,
            "patch_idx": [p_il, p_xl, p_t],
            "patch_center": {
                "il": float(handle.inline_center[p_il]),
                "xl": float(handle.xline_center[p_xl]),
                "sample": float(handle.time_center[p_t]),
            },
            "embedded_offset_used": embedded_offset_used,
        }

        if handle.summary_path is not None:
            summary = handle.get_summary()
            reg_offsets = summary.get("regression_offsets", handle.embedded_offsets)
            coords = summary["offset_coords"][:, p_il, p_xl, p_t, :]
            axis_var = summary["offset_axis_var"][p_il, p_xl, p_t, :]
            trajectory: dict[str, object] = {
                "available": True,
                "offsets": [float(x) for x in reg_offsets],
                "c1": [float(x) for x in coords[:, 0]],
                "c2": [float(x) for x in coords[:, 1]],
                "slope": float(summary["offset_slope"][p_il, p_xl, p_t]),
                "r2": float(summary["offset_fit_r2"][p_il, p_xl, p_t]),
                "axis_var": [float(axis_var[0]), float(axis_var[1])],
            }
        else:
            trajectory = {
                "available": False,
                "offsets": [],
                "c1": [],
                "c2": [],
                "slope": None,
                "r2": None,
                "axis_var": None,
            }
        return pick_info, trajectory

    def compute_trajectory(
        self,
        survey: SurveyHandle,
        embedding_set_id: str,
        il_idx: int,
        xl_idx: int,
        sample_idx: int,
        offset_idx: int = 0,
    ) -> dict[str, object]:
        handle = self.get_set(survey.survey_id, embedding_set_id)
        n_il, n_xl, n_t = survey.shape[0], survey.shape[1], survey.shape[2]
        if not 0 <= il_idx < n_il:
            raise IndexError(f"il_idx {il_idx} out of range [0, {n_il - 1}]")
        if not 0 <= xl_idx < n_xl:
            raise IndexError(f"xl_idx {xl_idx} out of range [0, {n_xl - 1}]")
        if not 0 <= sample_idx < n_t:
            raise IndexError(f"sample_idx {sample_idx} out of range [0, {n_t - 1}]")
        if survey.is_4d:
            n_off = survey.shape[3]
            if not 0 <= offset_idx < n_off:
                raise IndexError(f"offset_idx {offset_idx} out of range [0, {n_off - 1}]")

        p_il = _find_patch_idx(handle.il_patch_start, handle.patch_size, il_idx)
        p_xl = _find_patch_idx(handle.xl_patch_start, handle.patch_size, xl_idx)
        p_t = _find_patch_idx(handle.t_patch_start, handle.patch_size, sample_idx)
        pick_info, trajectory = self._build_pick_and_trajectory(
            handle=handle,
            survey=survey,
            il_idx=il_idx,
            xl_idx=xl_idx,
            sample_idx=sample_idx,
            offset_idx=offset_idx,
            p_il=p_il,
            p_xl=p_xl,
            p_t=p_t,
        )
        return {"pick": pick_info, "trajectory": trajectory}

    def compute_similarity(
        self,
        survey: SurveyHandle,
        embedding_set_id: str,
        il_idx: int | None = None,
        xl_idx: int | None = None,
        sample_idx: int | None = None,
        offset_idx: int = 0,
        mode: str = "active_offset",
        whiten: bool = False,
        features: Sequence[PickedFeaturePayload] | None = None,
        exclusion_weight: float = 0.75,
        focus_pick: PickCoord | None = None,
    ) -> dict[str, object]:
        handle = self.get_set(survey.survey_id, embedding_set_id)
        if mode not in handle.available_modes:
            raise ValueError(
                f"Mode '{mode}' is not available for embedding set '{embedding_set_id}'. "
                f"Available: {handle.available_modes}"
            )

        n_il, n_xl, n_t = survey.shape[0], survey.shape[1], survey.shape[2]
        if survey.is_4d:
            n_off = survey.shape[3]
            if not 0 <= offset_idx < n_off:
                raise IndexError(f"offset_idx {offset_idx} out of range [0, {n_off - 1}]")

        il_lut = _build_patch_lut(handle.il_patch_start, handle.patch_size, n_il)
        xl_lut = _build_patch_lut(handle.xl_patch_start, handle.patch_size, n_xl)
        t_lut = _build_patch_lut(handle.t_patch_start, handle.patch_size, n_t)

        if features is None:
            if il_idx is None or xl_idx is None or sample_idx is None:
                raise ValueError("il_idx, xl_idx, and sample_idx are required when features is None")
            if not 0 <= il_idx < n_il:
                raise IndexError(f"il_idx {il_idx} out of range [0, {n_il - 1}]")
            if not 0 <= xl_idx < n_xl:
                raise IndexError(f"xl_idx {xl_idx} out of range [0, {n_xl - 1}]")
            if not 0 <= sample_idx < n_t:
                raise IndexError(f"sample_idx {sample_idx} out of range [0, {n_t - 1}]")
            resolved_features: Sequence[PickedFeaturePayload] = [
                PickedFeaturePayload(
                    id="legacy-single",
                    role="include",
                    plane_axis="inline",
                    plane_index=il_idx,
                    geometry_mode="single",
                    picks=[PickCoord(il_idx=il_idx, xl_idx=xl_idx, sample_idx=sample_idx)],
                )
            ]
            effective_exclusion_weight = 0.0
            resolved_focus = (il_idx, xl_idx, sample_idx)
        else:
            resolved_features = features
            effective_exclusion_weight = float(exclusion_weight)
            resolved_focus = None

        include_patches: set[tuple[int, int, int]] = set()
        exclude_patches: set[tuple[int, int, int]] = set()
        first_include_last_pick: tuple[int, int, int] | None = None

        for feat in resolved_features:
            patches = _rasterize_feature_patches(
                feat,
                n_il=n_il,
                n_xl=n_xl,
                n_t=n_t,
                il_lut=il_lut,
                xl_lut=xl_lut,
                t_lut=t_lut,
            )
            if feat.role == "include":
                include_patches.update(patches)
                if first_include_last_pick is None and feat.picks:
                    last = feat.picks[-1]
                    first_include_last_pick = (last.il_idx, last.xl_idx, last.sample_idx)
            elif feat.role == "exclude":
                exclude_patches.update(patches)
            else:
                raise ValueError(f"Unsupported feature role '{feat.role}'")

        if not include_patches:
            raise ValueError("At least one active Inclusion feature with >= 1 pick is required")

        overlap = include_patches & exclude_patches
        overlap_resolved_count = len(overlap)
        remaining_include = include_patches - exclude_patches
        if remaining_include:
            effective_include = remaining_include
            effective_exclude = exclude_patches
        else:
            effective_include = include_patches
            effective_exclude = exclude_patches - effective_include

        if focus_pick is not None:
            f_il, f_xl, f_sample = focus_pick.il_idx, focus_pick.xl_idx, focus_pick.sample_idx
        elif resolved_focus is not None:
            f_il, f_xl, f_sample = resolved_focus
        elif il_idx is not None and xl_idx is not None and sample_idx is not None:
            f_il, f_xl, f_sample = il_idx, xl_idx, sample_idx
        else:
            assert first_include_last_pick is not None
            f_il, f_xl, f_sample = first_include_last_pick

        if not 0 <= f_il < n_il:
            raise IndexError(f"focus_pick.il_idx {f_il} out of range [0, {n_il - 1}]")
        if not 0 <= f_xl < n_xl:
            raise IndexError(f"focus_pick.xl_idx {f_xl} out of range [0, {n_xl - 1}]")
        if not 0 <= f_sample < n_t:
            raise IndexError(f"focus_pick.sample_idx {f_sample} out of range [0, {n_t - 1}]")

        f_p_il = int(il_lut[f_il])
        f_p_xl = int(xl_lut[f_xl])
        f_p_t = int(t_lut[f_sample])

        nearest_emb_off_idx = 0
        if handle.embedded_offsets.size > 0 and survey.is_4d:
            target_off = survey.offsets[offset_idx]
            nearest_emb_off_idx = int(np.argmin(np.abs(handle.embedded_offsets - target_off)))

        grid_shape = (
            len(handle.il_patch_start),
            len(handle.xl_patch_start),
            len(handle.t_patch_start),
        )
        inc_flat = [int(np.ravel_multi_index(p, grid_shape)) for p in sorted(effective_include)]
        exc_flat = [int(np.ravel_multi_index(p, grid_shape)) for p in sorted(effective_exclude)]

        device = get_device()

        if mode == "combined_trajectory":
            summary = handle.get_summary()
            zero_flat = _prepare_normalized_volume(summary["zero_offset"], whiten, device)
            dir_flat = _prepare_normalized_volume(summary["offset_direction"], whiten, device)
            sim_pos = _max_combined_similarity_over_queries(zero_flat, dir_flat, inc_flat)
            if exc_flat and effective_exclusion_weight > 0.0:
                sim_neg = torch.clamp(
                    _max_combined_similarity_over_queries(zero_flat, dir_flat, exc_flat),
                    min=0.0,
                )
                sim_flat = sim_pos - effective_exclusion_weight * sim_neg
            else:
                sim_flat = sim_pos
        else:
            if mode == "active_offset":
                vol = handle.read_offset_volume(nearest_emb_off_idx)
            elif mode == "zero_offset":
                if handle.summary_path is not None:
                    vol = handle.get_summary()["zero_offset"]
                else:
                    z_idx = _zero_offset_indices(handle.embedded_offsets)
                    vol = handle.read_zero_offset_volume(z_idx)
            elif mode == "offset_direction":
                vol = handle.get_summary()["offset_direction"]
            elif mode == "offset_curvature":
                vol = handle.get_summary()["offset_curvature"]
            else:
                raise ValueError(f"Unsupported mode '{mode}'")

            feats_flat = _prepare_normalized_volume(vol, whiten, device)
            sim_pos = _max_similarity_over_queries(feats_flat, inc_flat)
            if exc_flat and effective_exclusion_weight > 0.0:
                sim_neg = torch.clamp(_max_similarity_over_queries(feats_flat, exc_flat), min=0.0)
                sim_flat = sim_pos - effective_exclusion_weight * sim_neg
            else:
                sim_flat = sim_pos

        sim_t = torch.nan_to_num(sim_flat.reshape(grid_shape), nan=0.0, posinf=1.0, neginf=-1.0)
        sim_np = np.ascontiguousarray(sim_t.detach().cpu().numpy(), dtype="<f4")
        sim_min = float(sim_np.min())
        sim_max = float(sim_np.max())
        sim_b64 = base64.b64encode(sim_np.tobytes()).decode("ascii")

        all_picked = include_patches | exclude_patches
        masked_flat_indices = [int(np.ravel_multi_index(p, sim_np.shape)) for p in all_picked]
        flat = sim_np.ravel().copy()
        flat[masked_flat_indices] = -np.inf
        k = min(12, max(0, flat.size - len(all_picked)))
        top_matches: list[dict[str, object]] = []
        if k > 0:
            cand_idx = np.argpartition(flat, -k)[-k:]
            cand_idx = cand_idx[np.argsort(flat[cand_idx])[::-1]]
            half_p = handle.patch_size // 2
            for idx_flat in cand_idx:
                score = float(flat[idx_flat])
                if not np.isfinite(score):
                    continue
                m_il, m_xl, m_t = (int(v) for v in np.unravel_index(int(idx_flat), sim_np.shape))
                m_il_idx = min(int(handle.il_patch_start[m_il]) + half_p, n_il - 1)
                m_xl_idx = min(int(handle.xl_patch_start[m_xl]) + half_p, n_xl - 1)
                m_t_idx = min(int(handle.t_patch_start[m_t]) + half_p, n_t - 1)
                top_matches.append(
                    {
                        "patch_idx": [m_il, m_xl, m_t],
                        "il_idx": m_il_idx,
                        "xl_idx": m_xl_idx,
                        "sample_idx": m_t_idx,
                        "il": int(survey.ilines[m_il_idx]),
                        "xl": int(survey.xlines[m_xl_idx]),
                        "sample": float(survey.samples[m_t_idx]),
                        "score": score,
                    }
                )

        pick_info, trajectory = self._build_pick_and_trajectory(
            handle=handle,
            survey=survey,
            il_idx=f_il,
            xl_idx=f_xl,
            sample_idx=f_sample,
            offset_idx=offset_idx,
            p_il=f_p_il,
            p_xl=f_p_xl,
            p_t=f_p_t,
        )

        return {
            "pick": pick_info,
            "mode": mode,
            "whiten": whiten,
            "grid_shape": list(sim_np.shape),
            "patch_size": handle.patch_size,
            "il_patch_start": [int(x) for x in handle.il_patch_start],
            "xl_patch_start": [int(x) for x in handle.xl_patch_start],
            "t_patch_start": [int(x) for x in handle.t_patch_start],
            "similarity_min": sim_min,
            "similarity_max": sim_max,
            "similarity_b64": sim_b64,
            "top_matches": top_matches,
            "trajectory": trajectory,
            "exclusion_weight": effective_exclusion_weight,
            "included_patch_count": len(effective_include),
            "excluded_patch_count": len(effective_exclude),
            "overlap_resolved_count": overlap_resolved_count,
        }

    def get_attribute(
        self, survey_id: str, embedding_set_id: str, attribute: str
    ) -> dict[str, object]:
        handle = self.get_set(survey_id, embedding_set_id)
        if attribute not in handle.available_attributes:
            raise ValueError(
                f"Attribute '{attribute}' not available for embedding set '{embedding_set_id}'. "
                f"Available: {handle.available_attributes}"
            )
        summary = handle.get_summary()
        if attribute == "offset_slope":
            arr = summary["offset_slope"]
        elif attribute == "offset_fit_r2":
            arr = summary["offset_fit_r2"]
        elif attribute == "offset_axis_var_1":
            arr = summary["offset_axis_var"][..., 0]
        elif attribute == "offset_axis_var_2":
            arr = summary["offset_axis_var"][..., 1]
        elif attribute == "offset_axis_var_plane":
            arr = summary["offset_axis_var"].sum(axis=-1)
        else:
            raise ValueError(f"Unsupported attribute '{attribute}'")

        arr_f32 = np.ascontiguousarray(arr, dtype="<f4")
        return {
            "attribute": attribute,
            "grid_shape": list(arr_f32.shape),
            "patch_size": handle.patch_size,
            "il_patch_start": [int(x) for x in handle.il_patch_start],
            "xl_patch_start": [int(x) for x in handle.xl_patch_start],
            "t_patch_start": [int(x) for x in handle.t_patch_start],
            "value_min": float(arr_f32.min()),
            "value_max": float(arr_f32.max()),
            "values_b64": base64.b64encode(arr_f32.tobytes()).decode("ascii"),
        }

    def release_set(self, survey_id: str, embedding_set_id: str) -> None:
        with self._lock:
            survey_map = self._sets.get(survey_id)
            handle = survey_map.pop(embedding_set_id, None) if survey_map is not None else None
        if handle is not None:
            handle.close()

    def close(self) -> None:
        with self._lock:
            sets = self._sets
            self._sets = {}
        for survey_map in sets.values():
            for handle in survey_map.values():
                handle.close()
