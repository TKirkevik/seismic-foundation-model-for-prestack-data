"""Summarise per-offset patch embeddings as a zero-offset vector plus a 2-D offset basis.

Reads the ``embeddings`` array ``(n_offsets, n_il, n_xl, n_t, hidden)`` written by
``prestack_patch_embedder.py`` and, for every patch, describes the trajectory the embedding
traces through latent space as the offset changes with an orthonormal 2-D basis:

zero_offset       (n_il, n_xl, n_t, hidden)  embedding at zero offset. By default the intercept of
                  the per-patch fit below, i.e. the trajectory extrapolated to |offset| = 0 from
                  the live, un-tapered offsets (``--zero-offset nearest`` instead averages the
                  nearest negative and positive offsets, e.g. -50 m and +50 m).
offset_direction  (n_il, n_xl, n_t, hidden)  axis 1, unit vector: direction along which the
                  embedding moves as |offset| increases, i.e. the normalised slope ``b`` of the
                  least-squares fit ``e(|h|) = a + b * |h|`` over the live offsets ``h``.
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
live_offsets      (n_offsets, n_il, n_xl, n_t) bool  offsets used in the fit for each patch. Offsets
                  with |offset| < ``--min-offset`` (300 m, near-offset taper) are never used. With
                  ``--rms`` (per-patch RMS from ``patch_offset_rms.py``) an offset is additionally
                  required to have RMS >= ``--live-threshold`` x the patch's maximum RMS, which
                  removes stretch-muted far offsets and dead traces from the regression / PCA.
n_live_offsets    (n_il, n_xl, n_t)  number of live offsets; patches below ``--min-live`` are zero.

Patch coordinates and the run metadata of the input file are copied through, together with
``zero_offset_mode``, ``zero_offset_source`` (offsets averaged in ``nearest`` mode), ``min_offset``
and ``regression_offsets`` (all offsets, in the order of the ``live_offsets`` axis).

Example
-------
python embedders\\offset_direction_summarizer.py ^
    --input data\\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time_ncs3d_patch_embeddings.npz ^
    --rms data\\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time_patch_rms.npz
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
    embeddings: np.ndarray,
    abs_offsets: np.ndarray,
    reference: np.ndarray | None = None,
    live: np.ndarray | None = None,
    min_live: int = 4,
    chunk: int = 1,
) -> dict[str, np.ndarray]:
    """Fit a 2-D orthonormal basis to the offset trajectory of every patch.

    Axis 1 is the least-squares slope of embedding vs |offset|; axis 2 is the leading principal
    component of the centred embeddings after axis 1 has been projected out. When ``live`` is
    given, every statistic is computed per patch over its live offsets only (masked least squares).

    Args:
        embeddings: ``(n_offsets, ..., hidden)``.
        abs_offsets: ``(n_offsets,)`` regressor values.
        reference: ``(..., hidden)`` origin used for ``coords``. ``None`` uses the fitted
            intercept (the extrapolated embedding at |offset| = 0).
        live: optional ``(n_offsets, ...)`` boolean mask of offsets carrying live data per patch.
        min_live: patches with fewer live offsets are left as zeros (``n_live`` tells which).
        chunk: number of leading spatial indices processed at a time (controls peak memory).

    Returns:
        dict with ``direction``, ``curvature``, ``intercept`` ``(..., hidden)``;
        ``coords`` ``(n_offsets, ..., 2)``; ``slope``, ``r2`` ``(...)``; ``axis_var`` ``(..., 2)``;
        ``n_live`` ``(...)``.
    """
    n_off = embeddings.shape[0]
    spatial = embeddings.shape[1:-1]
    hidden = embeddings.shape[-1]
    x = abs_offsets.astype(np.float32)
    if live is None:
        live = np.ones((n_off, *spatial), dtype=bool)
    out = {
        "direction": np.zeros((*spatial, hidden), dtype=np.float32),
        "curvature": np.zeros((*spatial, hidden), dtype=np.float32),
        "intercept": np.zeros((*spatial, hidden), dtype=np.float32),
        "coords": np.zeros((n_off, *spatial, 2), dtype=np.float32),
        "slope": np.zeros(spatial, dtype=np.float32),
        "r2": np.zeros(spatial, dtype=np.float32),
        "axis_var": np.zeros((*spatial, 2), dtype=np.float32),
        "n_live": live.sum(axis=0).astype(np.int32),
    }

    for start in range(0, spatial[0], chunk):
        block = embeddings[:, start : start + chunk].astype(np.float32)  # (n_off, c, ..., hidden)
        sl = slice(start, start + block.shape[1])
        w = live[:, sl].astype(np.float32)  # (n_off, c, ...)
        n = w.sum(axis=0)
        ok = n >= max(min_live, 2)
        n_safe = np.maximum(n, 1)

        # Per-patch weighted centring of the regressor and the embeddings.
        xw = x.reshape(-1, *([1] * (w.ndim - 1)))
        xbar = (w * xw).sum(axis=0) / n_safe
        xc = (xw - xbar) * w  # zero at masked offsets
        sxx = np.square(xc).sum(axis=0)
        mean = (w[..., None] * block).sum(axis=0) / n_safe[..., None]
        centred = (block - mean) * w[..., None]
        ss_tot = np.square(centred).sum(axis=(0, -1))
        quad = (xc**2 - (xc**2).sum(axis=0) / n_safe) * w  # centred |h|^2, used to orient axis 2

        # Axis 1: weighted least-squares slope; intercept is the fit evaluated at |h| = 0.
        slope = np.einsum("o...,o...h->...h", xc, centred) / np.maximum(sxx, 1e-12)[..., None]
        slope[~ok] = 0
        intercept = mean - slope * xbar[..., None]
        slope_norm = np.linalg.norm(slope, axis=-1)
        dir1 = _unit(slope)

        # Axis 2: top PC of the residual after removing dir1, via the (n_off x n_off) Gram matrix.
        p1 = np.einsum("o...h,...h->o...", centred, dir1)
        resid = centred - p1[..., None] * dir1[None]
        gram = np.einsum("o...h,p...h->...op", resid, resid)
        _, vecs = np.linalg.eigh(gram)
        u = np.moveaxis(vecs[..., -1], -1, 0)  # (n_off, c, ...) leading eigenvector
        dir2 = _unit(np.einsum("o...,o...h->...h", u, resid))
        dir2[~ok] = 0
        p2 = np.einsum("o...h,...h->o...", centred, dir2)
        flip = np.einsum("o...,o...->...", quad, p2) < 0
        dir2[flip] *= -1
        p2[:, flip] *= -1

        out["direction"][sl] = dir1
        out["curvature"][sl] = dir2
        out["intercept"][sl] = intercept
        out["slope"][sl] = slope_norm
        np.divide(np.square(slope_norm) * sxx, ss_tot, out=out["r2"][sl], where=(ss_tot > 0) & ok)
        for k, p in enumerate((p1, p2)):
            np.divide(np.square(p).sum(axis=0), ss_tot, out=out["axis_var"][sl][..., k], where=(ss_tot > 0) & ok)

        origin = intercept if reference is None else reference[sl]
        shift = block - origin[None]
        out["coords"][:, sl, ..., 0] = np.einsum("o...h,...h->o...", shift, dir1)
        out["coords"][:, sl, ..., 1] = np.einsum("o...h,...h->o...", shift, dir2)

    return out


def live_offset_mask(rms: np.ndarray, threshold: float) -> np.ndarray:
    """Offsets whose patch RMS is at least ``threshold`` times that patch's maximum RMS over offsets."""
    peak = rms.max(axis=0, keepdims=True)
    return (rms >= threshold * peak) & (peak > 0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Patch-embedding .npz from prestack_patch_embedder.py.")
    p.add_argument("--output", type=Path, default=None, help="Output .npz (default: <input stem>_offset_summary.npz).")
    p.add_argument("--min-offset", type=float, default=300.0, help="Exclude offsets with |offset| below this from the fit (near-offset taper).")
    p.add_argument("--zero-offset", choices=["intercept", "nearest"], default="intercept",
                   help="zero_offset = fitted intercept at |offset|=0 (default) or mean of the nearest +/- offsets.")
    p.add_argument("--rms", type=Path, default=None, help="Per-patch RMS .npz from patch_offset_rms.py; enables mute-aware fitting over live offsets.")
    p.add_argument("--live-threshold", type=float, default=0.2, help="An offset is live for a patch if its RMS >= this fraction of the patch's maximum RMS.")
    p.add_argument("--min-live", type=int, default=4, help="Minimum live offsets needed to fit a patch.")
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
    nearest_zero = embeddings[zero_idx].astype(np.float32).mean(axis=0)

    # Live mask: offsets below --min-offset are never used; --rms additionally drops muted ones.
    live = np.abs(offsets) >= args.min_offset
    live = np.broadcast_to(live[:, None, None, None], embeddings.shape[:-1]).copy()
    log.info("Excluding %d offsets with |offset| < %g m from the fit", int((np.abs(offsets) < args.min_offset).sum()), args.min_offset)
    if args.rms is not None:
        rms_data = np.load(args.rms)
        rms = rms_data["rms"]
        if rms.shape != embeddings.shape[:-1] or not np.array_equal(rms_data["offsets"], offsets):
            raise SystemExit(f"{args.rms} does not match the embedding grid/offsets: rms {rms.shape} vs embeddings {embeddings.shape[:-1]}")
        live &= live_offset_mask(rms, args.live_threshold)
    n_live = live.sum(axis=0)
    log.info("Live offsets per patch: median %d, min %d, %d patches below --min-live %d",
             int(np.median(n_live)), int(n_live.min()), int((n_live < args.min_live).sum()), args.min_live)

    log.info("Fitting 2-D offset basis (zero_offset = %s)", args.zero_offset)
    reference = nearest_zero if args.zero_offset == "nearest" else None
    fit = fit_offset_basis(embeddings, np.abs(offsets), reference, live, args.min_live, args.chunk)
    zero_offset = fit["intercept"] if args.zero_offset == "intercept" else nearest_zero
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
        "n_live_offsets": fit["n_live"],
        "live_offsets": live,
        "live_threshold": args.live_threshold if args.rms is not None else 0.0,
        "min_offset": args.min_offset,
        "zero_offset_mode": args.zero_offset,
        "zero_offset_source": offsets[zero_idx] if args.zero_offset == "nearest" else np.zeros(0, dtype=offsets.dtype),
        "regression_offsets": offsets,
        "embeddings_path": str(args.input),
    }
    out.update({k: data[k] for k in PASSTHROUGH_KEYS if k in data.files})

    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, **out)
    log.info("Wrote %s (%.1fs)", output, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
