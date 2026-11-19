"""Summarise per-offset patch embeddings as a zero-offset vector plus a 2-D offset basis.

Reads the ``embeddings`` array ``(n_offsets, n_il, n_xl, n_t, hidden)`` written by
``prestack_patch_embedder.py`` and, for every patch, describes the trajectory the embedding
traces through latent space as the offset changes with an orthonormal 2-D basis:

zero_offset       (n_il, n_xl, n_t, hidden)  embedding at zero offset. If the file has no exact
                  zero offset, the mean of the nearest negative and positive offsets is used.
offset_direction  (n_il, n_xl, n_t, hidden)  axis 1, unit vector: direction along which the
                  embedding moves as |offset| increases, i.e. the normalised slope ``b`` of the
                  least-squares fit ``e(|h|) = a + b * |h|`` over all offsets ``h``.
offset_curvature  (n_il, n_xl, n_t, hidden)  axis 2, unit vector orthogonal to axis 1: the top
                  principal component of the across-offset variation after projecting axis 1
                  out. Its sign is chosen so the trajectory bends towards it at far offsets
                  (projection positively correlated with the centred ``|h|^2``).
offset_coords     (n_offsets, n_il, n_xl, n_t, 2)  coordinates of ``e_h - zero_offset`` in the
                  basis, so ``e_h ~= zero_offset + c1 * offset_direction + c2 * offset_curvature``.
offset_slope      (n_il, n_xl, n_t)  ||b||, latent-space displacement per unit |offset|.
offset_fit_r2     (n_il, n_xl, n_t)  fraction of the embedding variance across offsets explained
                  by the linear fit in |offset| (1 = embeddings lie exactly on a line in |h|).
offset_axis_var   (n_il, n_xl, n_t, 2)  fraction of the across-offset variance lying along each
                  axis; their sum is the fraction captured by the 2-D plane.

Patch coordinates and the run metadata of the input file are copied through, together with
``zero_offset_source`` (offsets averaged for ``zero_offset``) and ``regression_offsets``.

Example
-------
python embedders\\offset_direction_summarizer.py ^
    --input data\\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time_ncs3d_patch_embeddings.npz
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np


log = logging.getLogger("offset_direction_summarizer")

PASSTHROUGH_KEYS = (
    "il_patch_start",
    "xl_patch_start",
    "t_patch_start",
    "inline_center",
    "xline_center",
    "time_center",
    "normalize_mean",
    "normalize_std",
    "clip_sigma",
    "normalization_mode",
    "patch_size",
    "crop_size",
    "num_overlap_patches",
    "overlap_filter",
    "model_path",
    "model_dtype",
    "input_path",
)


def zero_offset_indices(offsets: np.ndarray) -> np.ndarray:
    """Indices of the offset(s) representing zero offset: an exact zero, else nearest ± pair."""
    exact = np.flatnonzero(offsets == 0)
    if exact.size:
        return exact
    neg = np.flatnonzero(offsets < 0)
    pos = np.flatnonzero(offsets > 0)
    if neg.size == 0 or pos.size == 0:
        return np.array([np.argmin(np.abs(offsets))])
    return np.array([neg[np.argmax(offsets[neg])], pos[np.argmin(offsets[pos])]])


def _unit(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return np.divide(v, norm, out=np.zeros_like(v), where=norm > 0)


def fit_offset_basis(
    embeddings: np.ndarray, abs_offsets: np.ndarray, reference: np.ndarray, chunk: int = 1
) -> dict[str, np.ndarray]:
    """Fit a 2-D orthonormal basis to the offset trajectory of every patch.

    Axis 1 is the least-squares slope of embedding vs |offset|; axis 2 is the leading principal
    component of the centred embeddings after axis 1 has been projected out.

    Args:
        embeddings: ``(n_offsets, ..., hidden)``.
        abs_offsets: ``(n_offsets,)`` regressor values.
        reference: ``(..., hidden)`` origin used for ``coords`` (the zero-offset embedding).
        chunk: number of leading spatial indices processed at a time (controls peak memory).

    Returns:
        dict with ``direction``, ``curvature`` ``(..., hidden)``; ``coords`` ``(n_offsets, ..., 2)``;
        ``slope``, ``r2`` ``(...)``; ``axis_var`` ``(..., 2)``.
    """
    x = abs_offsets.astype(np.float64)
    xc = (x - x.mean()).astype(np.float32)
    sxx = float(np.dot(xc, xc))
    if sxx == 0.0:
        raise ValueError("All offsets have the same magnitude; cannot fit a direction")
    quad = (xc**2 - np.mean(xc**2)).astype(np.float32)  # centred |h|^2, used to orient axis 2

    n_off = embeddings.shape[0]
    spatial = embeddings.shape[1:-1]
    hidden = embeddings.shape[-1]
    out = {
        "direction": np.zeros((*spatial, hidden), dtype=np.float32),
        "curvature": np.zeros((*spatial, hidden), dtype=np.float32),
        "coords": np.zeros((n_off, *spatial, 2), dtype=np.float32),
        "slope": np.zeros(spatial, dtype=np.float32),
        "r2": np.zeros(spatial, dtype=np.float32),
        "axis_var": np.zeros((*spatial, 2), dtype=np.float32),
    }

    for start in range(0, spatial[0], chunk):
        block = embeddings[:, start : start + chunk].astype(np.float32)  # (n_off, c, ..., hidden)
        sl = slice(start, start + block.shape[1])
        centred = block - block.mean(axis=0, keepdims=True)
        ss_tot = np.square(centred).sum(axis=(0, -1))

        # Axis 1: sum_o xc_o * centred_o == sum_o xc_o * e_o because xc sums to zero.
        slope = np.tensordot(xc, block, axes=(0, 0)) / sxx
        slope_norm = np.linalg.norm(slope, axis=-1)
        dir1 = _unit(slope)

        # Axis 2: top PC of the residual after removing dir1, via the (n_off x n_off) Gram matrix.
        p1 = np.einsum("o...h,...h->o...", centred, dir1)
        resid = centred - p1[..., None] * dir1[None]
        gram = np.einsum("o...h,p...h->...op", resid, resid)
        _, vecs = np.linalg.eigh(gram)
        u = np.moveaxis(vecs[..., -1], -1, 0)  # (n_off, c, ...) leading eigenvector
        dir2 = _unit(np.einsum("o...,o...h->...h", u, resid))
        p2 = np.einsum("o...h,...h->o...", centred, dir2)
        flip = np.tensordot(quad, p2, axes=(0, 0)) < 0
        dir2[flip] *= -1
        p2[:, flip] *= -1

        out["direction"][sl] = dir1
        out["curvature"][sl] = dir2
        out["slope"][sl] = slope_norm
        np.divide(np.square(slope_norm) * sxx, ss_tot, out=out["r2"][sl], where=ss_tot > 0)
        for k, p in enumerate((p1, p2)):
            np.divide(np.square(p).sum(axis=0), ss_tot, out=out["axis_var"][sl][..., k], where=ss_tot > 0)

        shift = block - reference[sl][None]
        out["coords"][:, sl, ..., 0] = np.einsum("o...h,...h->o...", shift, dir1)
        out["coords"][:, sl, ..., 1] = np.einsum("o...h,...h->o...", shift, dir2)

    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Patch-embedding .npz from prestack_patch_embedder.py.")
    p.add_argument("--output", type=Path, default=None, help="Output .npz (default: <input stem>_offset_summary.npz).")
    p.add_argument("--exclude-zero-offset", action="store_true", help="Leave the zero-offset source offsets out of the regression.")
    p.add_argument("--store-dtype", choices=["float16", "float32"], default="float32", help="dtype of the stored vectors.")
    p.add_argument("--chunk", type=int, default=1, help="Inline-patch rows processed per step (memory vs speed).")
    p.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    output = args.output or args.input.with_name(args.input.stem.replace("_patch_embeddings", "") + "_offset_summary.npz")
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists; pass --force to overwrite")

    t0 = time.time()
    data = np.load(args.input)
    embeddings = data["embeddings"]
    offsets = np.asarray(data["offsets"])
    log.info("Loaded %s: embeddings %s %s, %d offsets (%.1fs)", args.input, embeddings.shape, embeddings.dtype, len(offsets), time.time() - t0)
    if embeddings.ndim != 5 or embeddings.shape[0] != len(offsets):
        raise SystemExit("embeddings must have shape (n_offsets, n_il, n_xl, n_t, hidden)")

    zero_idx = zero_offset_indices(offsets)
    zero_offset = embeddings[zero_idx].astype(np.float32).mean(axis=0)
    log.info("Zero-offset embedding from offsets %s", offsets[zero_idx].tolist())

    reg_mask = np.ones(len(offsets), dtype=bool)
    if args.exclude_zero_offset:
        reg_mask[zero_idx] = False
    log.info("Fitting 2-D offset basis using %d offsets", int(reg_mask.sum()))
    fit = fit_offset_basis(embeddings[reg_mask], np.abs(offsets[reg_mask]), zero_offset, args.chunk)
    log.info(
        "Fit done (%.1fs): median linear R^2 %.3f, median variance along axes %.3f + %.3f = %.3f in plane",
        time.time() - t0,
        np.median(fit["r2"]),
        np.median(fit["axis_var"][..., 0]),
        np.median(fit["axis_var"][..., 1]),
        np.median(fit["axis_var"].sum(-1)),
    )

    store = np.dtype(args.store_dtype)
    out = {
        "zero_offset": zero_offset.astype(store),
        "offset_direction": fit["direction"].astype(store),
        "offset_curvature": fit["curvature"].astype(store),
        "offset_coords": fit["coords"],
        "offset_slope": fit["slope"],
        "offset_fit_r2": fit["r2"],
        "offset_axis_var": fit["axis_var"],
        "zero_offset_source": offsets[zero_idx],
        "regression_offsets": offsets[reg_mask],
        "embeddings_path": str(args.input),
    }
    out.update({k: data[k] for k in PASSTHROUGH_KEYS if k in data.files})

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **out)
    log.info("Wrote %s (%.1fs)", output, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
