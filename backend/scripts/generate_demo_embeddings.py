"""Generate signal-derived 3D patch embeddings and offset trajectory summaries for fast demos."""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Callable

import numpy as np
from numpy.lib import format as npformat
from seismic_zfp.read import SgzReader


PATCH_SIZE = 16
HIDDEN_SIZE = 768


def _patch_centers(first: float, step: float, starts: np.ndarray, patch_size: int) -> np.ndarray:
    return first + (starts + patch_size // 2) * step


def _axis_step(values: np.ndarray) -> float:
    return float(values[1] - values[0]) if len(values) > 1 else 1.0


def _extract_patch_features(volume: np.ndarray, patch_size: int = PATCH_SIZE) -> np.ndarray:
    """Compute fast vectorized 3D geophysical features per 16x16x16 patch.

    Returns array of shape ``(n_il_p, n_xl_p, n_t_p, n_features)`` in float32.
    """
    n_il, n_xl, n_t = volume.shape
    n_il_p = math.ceil(n_il / patch_size)
    n_xl_p = math.ceil(n_xl / patch_size)
    n_t_p = math.ceil(n_t / patch_size)

    pad_il = n_il_p * patch_size - n_il
    pad_xl = n_xl_p * patch_size - n_xl
    pad_t = n_t_p * patch_size - n_t
    if pad_il or pad_xl or pad_t:
        padded = np.pad(volume.astype(np.float32, copy=False), ((0, pad_il), (0, pad_xl), (0, pad_t)))
    else:
        padded = volume.astype(np.float32, copy=False)

    # Make contiguous (n_il_p, n_xl_p, n_t_p, P, P, P) once for fast SIMD reductions
    blocks = np.ascontiguousarray(
        padded.reshape(n_il_p, patch_size, n_xl_p, patch_size, n_t_p, patch_size).transpose(0, 2, 4, 1, 3, 5)
    )
    axes = (3, 4, 5)

    mean_amp = blocks.mean(axis=axes)
    abs_blocks = np.abs(blocks)
    max_abs = abs_blocks.max(axis=axes)
    pos_mean = 0.5 * (mean_amp + abs_blocks.mean(axis=axes))
    neg_mean = pos_mean - mean_amp

    sq = np.square(blocks)
    mean_sq = sq.mean(axis=axes)
    rms_amp = np.sqrt(mean_sq)
    std_amp = np.sqrt(np.maximum(mean_sq - np.square(mean_amp), 0.0))

    # Vertical (time/sample) reflection dynamics & dominant frequency proxies
    dt = blocks[..., 1:] - blocks[..., :-1]
    dt_rms = np.sqrt(np.square(dt).mean(axis=axes))
    d2t_rms = np.sqrt(np.square(dt[..., 1:] - dt[..., :-1]).mean(axis=axes))
    zero_cross = ((blocks[..., 1:] * blocks[..., :-1]) < 0).mean(axis=axes, dtype=np.float32)

    # Spatial dip proxies (inline and crossline differences & coherency with vertical gradient)
    dil = blocks[:, :, :, 1:, :, :] - blocks[:, :, :, :-1, :, :]
    dxl = blocks[:, :, :, :, 1:, :] - blocks[:, :, :, :, :-1, :]
    dil_rms = np.sqrt(np.square(dil).mean(axis=axes))
    dxl_rms = np.sqrt(np.square(dxl).mean(axis=axes))
    il_dip = (dil[..., :-1] * dt[:, :, :, :-1, :, :]).mean(axis=axes)
    xl_dip = (dxl[..., :-1] * dt[:, :, :, :, :-1, :]).mean(axis=axes)

    # 2x2x2 sub-block (octant) RMS energies and signed means (8 + 8 = 16 features)
    half = patch_size // 2
    oct_rms = np.sqrt(
        sq.reshape(n_il_p, n_xl_p, n_t_p, 2, half, 2, half, 2, half).mean(axis=(4, 6, 8))
    ).reshape(n_il_p, n_xl_p, n_t_p, 8)
    oct_mean = (
        blocks.reshape(n_il_p, n_xl_p, n_t_p, 2, half, 2, half, 2, half).mean(axis=(4, 6, 8))
    ).reshape(n_il_p, n_xl_p, n_t_p, 8)

    # 4 vertical quarter-window RMS energies (within-patch amplitude envelope trend)
    q = patch_size // 4
    vert_quarters = np.sqrt(
        sq.reshape(n_il_p, n_xl_p, n_t_p, patch_size, patch_size, 4, q).mean(axis=(3, 4, 6))
    )  # (..., 4)

    # 4 spectral band energies on the patch-averaged vertical waveform (fast 1D FFT)
    stacked_trace = blocks.mean(axis=(3, 4))  # (n_il_p, n_xl_p, n_t_p, 16)
    spec = np.abs(np.fft.rfft(stacked_trace, axis=-1))  # (..., 9)
    spec_bands = np.stack(
        [
            spec[..., 0:2].mean(axis=-1),
            spec[..., 2:4].mean(axis=-1),
            spec[..., 4:6].mean(axis=-1),
            spec[..., 6:].mean(axis=-1),
        ],
        axis=-1,
    )

    scalars = np.stack(
        [
            mean_amp,
            rms_amp,
            std_amp,
            max_abs,
            pos_mean,
            neg_mean,
            dt_rms,
            d2t_rms,
            zero_cross,
            dil_rms,
            dxl_rms,
            il_dip,
            xl_dip,
        ],
        axis=-1,
    )
    return np.concatenate([scalars, oct_rms, oct_mean, vert_quarters, spec_bands], axis=-1).astype(np.float32)


def generate_demo_embeddings(
    sgz_path: Path | str,
    output_dir: Path | str,
    tag: str = "demo",
    offset_stride: int = 5,
    force: bool = True,
    summarizer_script: Path | None = None,
    progress_callback: Callable[[float, str], None] | None = None,
) -> tuple[Path, Path]:
    """Generate signal-derived demo patch embeddings + offset trajectory summary for an SGZ survey."""
    sgz_path = Path(sgz_path).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = sgz_path.stem
    suffix = f"_{tag}" if tag and tag != "default" else ""
    patch_npz = output_dir / f"{stem}{suffix}_patch_embeddings.npz"
    summary_npz = output_dir / f"{stem}{suffix}_offset_summary.npz"

    if not force and patch_npz.exists() and summary_npz.exists():
        return patch_npz, summary_npz

    if summarizer_script is None:
        summarizer_script = Path(__file__).resolve().parents[2] / "embedders" / "offset_direction_summarizer.py"

    reader = SgzReader(str(sgz_path))
    try:
        if reader.is_4d:
            offset_indices = list(range(0, reader.n_offsets, max(1, offset_stride)))
            offsets_arr = np.asarray(reader.offsets, dtype=np.float64)[offset_indices]
        elif reader.is_3d:
            offset_indices = [0]
            offsets_arr = np.array([0.0], dtype=np.float64)
        else:
            raise ValueError(f"{sgz_path} is a 2D SGZ file; expected 4D or 3D")

        n_il, n_xl, n_t = reader.n_ilines, reader.n_xlines, reader.n_samples
        n_il_p = math.ceil(n_il / PATCH_SIZE)
        n_xl_p = math.ceil(n_xl / PATCH_SIZE)
        n_t_p = math.ceil(n_t / PATCH_SIZE)

        feat_list: list[np.ndarray] = []
        live_sum = 0.0
        live_sq = 0.0
        live_cnt = 0

        for i, off_idx in enumerate(offset_indices):
            if reader.is_4d:
                vol = reader.read_offset(off_idx)
            else:
                vol = reader.read_volume()

            if i % max(1, len(offset_indices) // 4) == 0:
                live = vol[np.any(vol != 0, axis=-1)].astype(np.float64)
                live_sum += float(live.sum())
                live_sq += float(np.square(live).sum())
                live_cnt += int(live.size)

            feat_list.append(_extract_patch_features(vol, PATCH_SIZE))
            if progress_callback is not None:
                progress_callback(10.0 + 60.0 * (i + 1) / len(offset_indices), f"Extracted offset {i + 1}/{len(offset_indices)}")

        raw_feats = np.stack(feat_list, axis=0)  # (n_off, n_il_p, n_xl_p, n_t_p, F)
        ilines = np.asarray(reader.ilines)
        xlines = np.asarray(reader.xlines)
        zslices = np.asarray(reader.zslices)
    finally:
        reader.close()

    norm_mean = live_sum / live_cnt if live_cnt > 0 else 0.0
    norm_std = math.sqrt(max(live_sq / live_cnt - norm_mean**2, 0.0)) if live_cnt > 0 else 1.0

    # Standardize features across the full pre-stack patch grid so offset trends remain intact
    f_mean = raw_feats.mean(axis=(0, 1, 2, 3), keepdims=True)
    f_std = raw_feats.std(axis=(0, 1, 2, 3), keepdims=True)
    norm_feats = np.clip((raw_feats - f_mean) / (f_std + 1e-6), -6.0, 6.0)

    n_features = norm_feats.shape[-1]
    rng = np.random.default_rng(42)
    q_mat, _ = np.linalg.qr(rng.standard_normal((HIDDEN_SIZE, n_features)))
    proj = q_mat.T.astype(np.float32)  # (F, 768) orthonormal rows

    if progress_callback is not None:
        progress_callback(75.0, "Projecting & writing ZIP_STORED patch embeddings")

    emb_shape = (len(offset_indices), n_il_p, n_xl_p, n_t_p, HIDDEN_SIZE)
    store_dtype = np.dtype("float16")

    with zipfile.ZipFile(patch_npz, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        with zf.open("embeddings.npy", "w", force_zip64=True) as fh:
            npformat.write_array_header_1_0(
                fh, {"descr": npformat.dtype_to_descr(store_dtype), "fortran_order": False, "shape": emb_shape}
            )
            for off_i in range(len(offset_indices)):
                projected = (norm_feats[off_i] @ proj).astype(store_dtype)
                fh.write(np.ascontiguousarray(projected).tobytes())

        il_starts = np.arange(n_il_p, dtype=np.int64) * PATCH_SIZE
        xl_starts = np.arange(n_xl_p, dtype=np.int64) * PATCH_SIZE
        t_starts = np.arange(n_t_p, dtype=np.int64) * PATCH_SIZE
        metadata = {
            "offsets": offsets_arr,
            "offset_index": np.asarray(offset_indices, dtype=np.int64),
            "il_patch_start": il_starts,
            "xl_patch_start": xl_starts,
            "t_patch_start": t_starts,
            "inline_center": _patch_centers(float(ilines[0]), _axis_step(ilines), il_starts, PATCH_SIZE),
            "xline_center": _patch_centers(float(xlines[0]), _axis_step(xlines), xl_starts, PATCH_SIZE),
            "time_center": _patch_centers(float(zslices[0]), _axis_step(zslices), t_starts, PATCH_SIZE),
            "normalize_mean": float(norm_mean),
            "normalize_std": float(norm_std),
            "clip_sigma": 3.0,
            "normalization_mode": "std",
            "patch_size": PATCH_SIZE,
            "crop_size": 224,
            "num_overlap_patches": 0,
            "overlap_filter": "ramp",
            "model_path": "signal-derived-demo",
            "model_dtype": "float16",
            "input_path": str(sgz_path),
        }
        for name, value in metadata.items():
            with zf.open(f"{name}.npy", "w") as fh:
                npformat.write_array(fh, np.asarray(value), allow_pickle=False)

    if len(offset_indices) >= 2 and summarizer_script.is_file():
        if progress_callback is not None:
            progress_callback(85.0, "Running offset_direction_summarizer.py")
        cmd = [
            sys.executable,
            str(summarizer_script),
            "--input",
            str(patch_npz),
            "--output",
            str(summary_npz),
            "--force",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            raise RuntimeError(f"offset_direction_summarizer failed ({res.returncode}): {res.stderr or res.stdout}")

    if progress_callback is not None:
        progress_callback(100.0, "Demo embeddings ready")
    return patch_npz, summary_npz


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Input .sgz survey path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory (defaults to input parent).")
    parser.add_argument("--tag", default="demo", help="Tag for output .npz files (default: demo).")
    parser.add_argument("--offset-stride", type=int, default=5, help="Stride across offsets (default: 5).")
    parser.add_argument("--no-force", action="store_true", help="Skip if outputs already exist.")
    args = parser.parse_args(argv)

    out_dir = args.output_dir or args.input.parent
    patch_npz, summary_npz = generate_demo_embeddings(
        sgz_path=args.input,
        output_dir=out_dir,
        tag=args.tag,
        offset_stride=args.offset_stride,
        force=not args.no_force,
    )
    print(f"Generated {patch_npz} and {summary_npz}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
