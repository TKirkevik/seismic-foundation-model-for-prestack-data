"""Plot the per-patch offset summary (zero-offset vector + 2-D offset basis) as PNG figures.

Input is the ``*_offset_summary.npz`` written by ``offset_direction_summarizer.py``. Every
figure is written to ``--figures`` (default ``figures/``). Slices are always produced in all
three orientations (inline: xl x t, crossline: il x t, time: il x xl), passing through the
reference patch where a single position is needed.

Figures
-------
scalars_{inline,xline,time}.png   rows = slope, linear R2, variance along axis 1 / axis 2 / plane;
                                  columns = slices at 25 / 50 / 75 % of the axis.
cosine_<field>_ref<il_xl_t>.png   cosine similarity of every patch's vector to that of the
                                  reference patch, three orthogonal slices through the reference.
pca_rgb_<field>.png               top-3 principal components of the field over all patches
                                  mapped to RGB, three orthogonal slices.
clusters_map.png                  spherical k-means classes of offset_direction.
clusters_trajectories.png         mean 2-D offset trajectory (offset_coords) per class, and the
                                  axis-1 / axis-2 coordinate versus signed offset.
coherence_{inline,xline,time}.png mean cosine between each patch's vector and its face
                                  neighbours, rows = zero_offset, offset_direction, offset_curvature.
trajectories_column.png           trajectories of every patch in the reference (il, xl) column,
                                  coloured by time.
trajectories_grid.png             small multiples of the trajectories around the reference patch
                                  in its time slice.

Patches whose embedding does not change with offset (``offset_slope == 0``: fully padded /
dead areas) are masked out of every statistic and shown blank.

Example
-------
python visualization\\plot_offset_summary.py ^
    --input data\\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time_ncs3d_offset_summary.npz ^
    --ref 6,9,35 --clusters 6
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402


log = logging.getLogger("plot_offset_summary")

VECTOR_FIELDS = ("zero_offset", "offset_direction", "offset_curvature")
SCALAR_FIELDS = (
    ("n_live", "live offsets per patch", "cividis", None),
    ("offset_slope", "slope ||b|| per m offset", "viridis", None),
    ("offset_fit_r2", "linear R$^2$ in |offset|", "magma", (0, 1)),
    ("axis_var_1", "variance along axis 1", "magma", (0, 1)),
    ("axis_var_2", "variance along axis 2", "magma", (0, 1)),
    ("plane_var", "variance in 2-D plane", "magma", (0, 1)),
)
ORIENTATIONS = ("inline", "xline", "time")
GROUPS = ("scalars", "cosine", "pca", "clusters", "coherence", "trajectories")


# ---------------------------------------------------------------------------
# Data access
# ---------------------------------------------------------------------------


class Summary:
    def __init__(self, path: Path):
        d = np.load(path)
        self.zero_offset = d["zero_offset"].astype(np.float32)
        self.offset_direction = d["offset_direction"].astype(np.float32)
        self.offset_curvature = d["offset_curvature"].astype(np.float32)
        self.coords = d["offset_coords"].astype(np.float32)  # (n_off, il, xl, t, 2)
        self.offsets = d["regression_offsets"].astype(float)
        self.scalars = {
            "offset_slope": d["offset_slope"],
            "offset_fit_r2": d["offset_fit_r2"],
            "axis_var_1": d["offset_axis_var"][..., 0],
            "axis_var_2": d["offset_axis_var"][..., 1],
            "plane_var": d["offset_axis_var"].sum(-1),
        }
        n_off = len(self.offsets)
        self.live = d["live_offsets"] if "live_offsets" in d.files else np.ones((n_off, *d["offset_slope"].shape), bool)
        self.scalars["n_live"] = self.live.sum(0)
        self.valid = d["offset_slope"] > 0
        self.il = d["inline_center"]
        self.xl = d["xline_center"]
        self.t = d["time_center"]
        self.shape = self.valid.shape

    def field(self, name: str) -> np.ndarray:
        return getattr(self, name)


# ---------------------------------------------------------------------------
# Slicing helpers
# ---------------------------------------------------------------------------


def _extent(x: np.ndarray, y: np.ndarray) -> list[float]:
    """imshow extent placing pixel centres on the coordinate values, y increasing downward."""
    dx = (x[1] - x[0]) / 2 if len(x) > 1 else 0.5
    dy = (y[1] - y[0]) / 2 if len(y) > 1 else 0.5
    return [x[0] - dx, x[-1] + dx, y[-1] + dy, y[0] - dy]


def take_slice(s: Summary, vol: np.ndarray, orientation: str, idx: int):
    """Return ``(image, extent, xlabel, ylabel, title)`` for a 2-D slice of an (il, xl, t[, c]) volume."""
    if orientation == "inline":
        return np.swapaxes(vol[idx], 0, 1), _extent(s.xl, s.t), "crossline", "time (ms)", f"inline {s.il[idx]:.0f}"
    if orientation == "xline":
        return np.swapaxes(vol[:, idx], 0, 1), _extent(s.il, s.t), "inline", "time (ms)", f"crossline {s.xl[idx]:.0f}"
    return np.swapaxes(vol[:, :, idx], 0, 1), _extent(s.il, s.xl), "inline", "crossline", f"time {s.t[idx]:.0f} ms"


def axis_len(s: Summary, orientation: str) -> int:
    return s.shape[ORIENTATIONS.index(orientation)]


def quartile_positions(n: int) -> list[int]:
    return sorted({min(int(round(n * f)), n - 1) for f in (0.25, 0.5, 0.75)})


def draw_slice(ax, s, vol, orientation, idx, cmap=None, vmin=None, vmax=None, mark=None):
    img, extent, xlabel, ylabel, title = take_slice(s, vol, orientation, idx)
    im = ax.imshow(img, extent=extent, aspect="auto", interpolation="nearest", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel(ylabel, fontsize=8)
    ax.tick_params(labelsize=7)
    if mark is not None:
        mi, mj, mk = mark
        xy = {"inline": (s.xl[mj], s.t[mk]), "xline": (s.il[mi], s.t[mk]), "time": (s.il[mi], s.xl[mj])}[orientation]
        ax.plot(*xy, marker="s", mfc="none", mec="white", mew=1.5, ms=9)
        ax.plot(*xy, marker="s", mfc="none", mec="black", mew=0.5, ms=9)
    return im


def masked(vol: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.array(vol, dtype=np.float32)
    out[~valid] = np.nan
    return out


# ---------------------------------------------------------------------------
# Vector-field reductions
# ---------------------------------------------------------------------------


def cosine_to_reference(field: np.ndarray, ref: tuple[int, int, int]) -> np.ndarray:
    v = field[ref]
    v = v / max(np.linalg.norm(v), 1e-12)
    norms = np.linalg.norm(field, axis=-1)
    return np.divide(field @ v, norms, out=np.zeros(norms.shape, np.float32), where=norms > 0)


def pca_rgb(field: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Top-3 PCs of the field over valid patches, robustly scaled to [0, 1] per channel.

    Returns ``(rgb volume (..., 3), explained variance ratio of the 3 PCs)``.
    """
    x = field[valid]
    mean = x.mean(0)
    _, sing, vt = np.linalg.svd(x - mean, full_matrices=False)
    explained = sing[:3] ** 2 / np.square(sing).sum()
    proj = (field - mean) @ vt[:3].T
    lo, hi = np.percentile(proj[valid], [2, 98], axis=0)
    rgb = np.clip((proj - lo) / np.maximum(hi - lo, 1e-12), 0, 1).astype(np.float32)
    rgb[~valid] = 1.0  # white where masked
    return rgb, explained


def spherical_kmeans(x: np.ndarray, k: int, seed: int, n_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """k-means on the unit sphere (cosine similarity). Returns ``(labels, unit centroids)``, largest class first."""
    rng = np.random.default_rng(seed)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    centroids = [x[rng.integers(len(x))]]
    for _ in range(1, k):  # k-means++ seeding with cosine distance
        d = np.clip(1 - np.max(x @ np.stack(centroids).T, axis=1), 0, None)
        centroids.append(x[rng.choice(len(x), p=d / d.sum())])
    c = np.stack(centroids)
    labels = np.full(len(x), -1)
    for _ in range(n_iter):
        new_labels = np.argmax(x @ c.T, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = x[labels == j]
            c[j] = members.mean(0) if len(members) else x[rng.integers(len(x))]
        c /= np.maximum(np.linalg.norm(c, axis=1, keepdims=True), 1e-12)
    order = np.argsort(-np.bincount(labels, minlength=k))
    remap = np.empty(k, dtype=int)
    remap[order] = np.arange(k)
    return remap[labels], c[order]


def neighbour_coherence(field: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Mean cosine between each patch's vector and its valid face neighbours (NaN where masked)."""
    unit = field / np.maximum(np.linalg.norm(field, axis=-1, keepdims=True), 1e-12)
    unit[~valid] = 0
    total = np.zeros(valid.shape, np.float32)
    count = np.zeros(valid.shape, np.float32)
    for axis in range(3):
        for shift in (-1, 1):
            rolled = np.roll(unit, shift, axis=axis)
            rolled_valid = np.roll(valid, shift, axis=axis)
            edge = [slice(None)] * 3
            edge[axis] = 0 if shift == 1 else -1  # wrapped-around positions are not neighbours
            rolled_valid[tuple(edge)] = False
            total += (unit * rolled).sum(-1) * rolled_valid
            count += rolled_valid
    coh = np.divide(total, count, out=np.full(valid.shape, np.nan, np.float32), where=count > 0)
    coh[~valid] = np.nan
    return coh


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def plot_scalar_slices(s: Summary, out: Path) -> None:
    for orientation in ORIENTATIONS:
        positions = quartile_positions(axis_len(s, orientation))
        fig, axes = plt.subplots(
            len(SCALAR_FIELDS), len(positions), figsize=(4.2 * len(positions), 3.3 * len(SCALAR_FIELDS)), squeeze=False
        )
        for r, (key, label, cmap, lim) in enumerate(SCALAR_FIELDS):
            vol = masked(s.scalars[key], s.valid) if key != "n_live" else s.scalars[key].astype(np.float32)
            vmin, vmax = lim if lim else np.nanpercentile(vol, [1, 99])
            for c, idx in enumerate(positions):
                im = draw_slice(axes[r, c], s, vol, orientation, idx, cmap=cmap, vmin=vmin, vmax=vmax)
                if r < len(SCALAR_FIELDS) - 1:
                    axes[r, c].set_xlabel("")
            fig.colorbar(im, ax=axes[r, :].tolist(), label=label, fraction=0.02, pad=0.02)
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(f"Scalar offset diagnostics – {orientation} slices", fontsize=12)
        fig.savefig(out / f"scalars_{orientation}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def plot_three_views(s, vol, ref, title, path, cmap=None, vmin=None, vmax=None, cbar_label=None, cbar_ticks=None, cbar_ticklabels=None):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, orientation, idx in zip(axes, ORIENTATIONS, ref):
        im = draw_slice(ax, s, vol, orientation, idx, cmap=cmap, vmin=vmin, vmax=vmax, mark=ref)
    if cbar_label is not None:
        cb = fig.colorbar(im, ax=axes.tolist(), label=cbar_label, fraction=0.02, pad=0.02, ticks=cbar_ticks)
        if cbar_ticklabels is not None:
            cb.ax.set_yticklabels(cbar_ticklabels)
    fig.suptitle(title, fontsize=12)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_reference_cosine(s: Summary, ref, out: Path) -> None:
    tag = "_".join(str(r) for r in ref)
    for name in VECTOR_FIELDS:
        cos = masked(cosine_to_reference(s.field(name), ref), s.valid)
        plot_three_views(
            s, cos, ref,
            f"cosine({name}, {name} at reference patch il={ref[0]}, xl={ref[1]}, t={ref[2]})",
            out / f"cosine_{name}_ref{tag}.png", cmap="RdBu_r", vmin=-1, vmax=1, cbar_label="cosine similarity",
        )


def plot_pca_rgb(s: Summary, ref, out: Path) -> None:
    for name in VECTOR_FIELDS:
        rgb, explained = pca_rgb(s.field(name), s.valid)
        plot_three_views(
            s, rgb, ref,
            f"{name}: PCs 1/2/3 as R/G/B (explained {explained[0]:.2f} / {explained[1]:.2f} / {explained[2]:.2f})",
            out / f"pca_rgb_{name}.png",
        )


def plot_clusters(s: Summary, ref, out: Path, k: int, seed: int) -> None:
    labels_flat, centroids = spherical_kmeans(s.offset_direction[s.valid], k, seed)
    labels = np.full(s.shape, np.nan, np.float32)
    labels[s.valid] = labels_flat
    cmap = ListedColormap(plt.get_cmap("tab10").colors[:k])
    sizes = np.bincount(labels_flat, minlength=k)
    plot_three_views(
        s, labels, ref, f"offset_direction classes – spherical k-means, k={k}", out / "clusters_map.png",
        cmap=cmap, vmin=-0.5, vmax=k - 0.5, cbar_label="class", cbar_ticks=np.arange(k),
        cbar_ticklabels=[f"{j} (n={sizes[j]})" for j in range(k)],
    )

    coords = s.coords.reshape(len(s.offsets), -1, 2)[:, s.valid.ravel()]  # (n_off, n_valid, 2)
    live = s.live.reshape(len(s.offsets), -1)[:, s.valid.ravel()]
    order = np.argsort(s.offsets)
    signed = s.offsets[order]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    for j in range(k):
        members = labels_flat == j
        w = live[:, members].astype(np.float32)
        # mean over live members only; offsets live for < 10 % of the class are not drawn
        mean = (coords[:, members] * w[..., None]).sum(1) / np.maximum(w.sum(1), 1)[:, None]
        mean[w.sum(1) < 0.1 * members.sum()] = np.nan
        mean = mean[order]
        colour = cmap(j)
        axes[0].plot(mean[:, 0], mean[:, 1], "-", color=colour, lw=1.5, label=f"class {j} (n={sizes[j]})")
        axes[0].scatter(mean[:, 0], mean[:, 1], c=np.abs(signed), cmap="viridis", s=10, zorder=3)
        axes[1].plot(signed, mean[:, 0], color=colour, lw=1.5)
        axes[2].plot(signed, mean[:, 1], color=colour, lw=1.5)
    axes[0].set(xlabel="axis 1 (offset_direction) coordinate", ylabel="axis 2 (offset_curvature) coordinate",
                title="mean trajectory per class over live offsets (dots coloured by |offset|)")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="offset (m)", ylabel="axis 1 coordinate", title="axis 1 vs signed offset")
    axes[2].set(xlabel="offset (m)", ylabel="axis 2 coordinate", title="axis 2 vs signed offset")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
    off_diag = ~np.eye(k, dtype=bool)
    sim = centroids @ centroids.T
    fig.suptitle(
        f"Offset trajectories per class (centroid cosines: min {sim[off_diag].min():.2f}, max {sim[off_diag].max():.2f})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out / "clusters_trajectories.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_coherence(s: Summary, out: Path) -> None:
    coh = {name: neighbour_coherence(s.field(name), s.valid) for name in VECTOR_FIELDS}
    for orientation in ORIENTATIONS:
        positions = quartile_positions(axis_len(s, orientation))
        fig, axes = plt.subplots(len(VECTOR_FIELDS), len(positions), figsize=(4.2 * len(positions), 3.3 * len(VECTOR_FIELDS)), squeeze=False)
        for r, name in enumerate(VECTOR_FIELDS):
            vmin = np.nanpercentile(coh[name], 1)
            for c, idx in enumerate(positions):
                im = draw_slice(axes[r, c], s, coh[name], orientation, idx, cmap="cividis", vmin=vmin, vmax=1)
                if r < len(VECTOR_FIELDS) - 1:
                    axes[r, c].set_xlabel("")
            fig.colorbar(im, ax=axes[r, :].tolist(), label=f"{name}\nneighbour cosine", fraction=0.02, pad=0.02)
        fig.subplots_adjust(hspace=0.3)
        fig.suptitle(f"Spatial coherence of latent vectors – {orientation} slices", fontsize=12)
        fig.savefig(out / f"coherence_{orientation}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)


def plot_trajectories(s: Summary, ref, out: Path) -> None:
    i, j, k = ref
    order = np.argsort(s.offsets)
    signed = s.offsets[order]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    cmap = plt.get_cmap("plasma")
    for kk in np.flatnonzero(s.valid[i, j]):
        c = s.coords[order, i, j, kk]
        m = s.live[order, i, j, kk]
        colour = cmap(kk / max(len(s.t) - 1, 1))
        style = {"color": colour, "lw": 2.2 if kk == k else 0.8, "alpha": 0.9 if kk == k else 0.6}
        faint = {"color": colour, "lw": 0.6, "alpha": 0.12, "ls": ":"}
        live_c = np.where(m[:, None], c, np.nan)  # muted offsets drawn dotted and faint
        axes[0].plot(c[:, 0], c[:, 1], **faint)
        axes[0].plot(live_c[:, 0], live_c[:, 1], **style)
        axes[1].plot(signed, c[:, 0], **faint)
        axes[1].plot(signed, live_c[:, 0], **style)
        axes[2].plot(signed, c[:, 1], **faint)
        axes[2].plot(signed, live_c[:, 1], **style)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(s.t[0], s.t[-1]))
    fig.colorbar(sm, ax=axes.tolist(), label="time (ms)", fraction=0.02, pad=0.02)
    axes[0].set(xlabel="axis 1 coordinate", ylabel="axis 2 coordinate", title="trajectory in own 2-D basis")
    axes[1].set(xlabel="offset (m)", ylabel="axis 1 coordinate", title="axis 1 vs signed offset")
    axes[2].set(xlabel="offset (m)", ylabel="axis 2 coordinate", title="axis 2 vs signed offset")
    for ax in axes:
        ax.axhline(0, color="k", lw=0.5)
        ax.axvline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
    fig.suptitle(
        f"Offset trajectories of all patches in column inline {s.il[i]:.0f}, crossline {s.xl[j]:.0f} "
        f"(bold: reference t={s.t[k]:.0f} ms; dotted: muted offsets)",
        fontsize=12,
    )
    fig.savefig(out / "trajectories_column.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    half = 2
    ii = list(range(max(i - half, 0), min(i + half + 1, s.shape[0])))
    jj = list(range(max(j - half, 0), min(j + half + 1, s.shape[1])))
    fig, axes = plt.subplots(len(jj), len(ii), figsize=(2.6 * len(ii), 2.4 * len(jj)), squeeze=False, sharex=True, sharey=True)
    abs_off = np.abs(signed)
    for r, jx in enumerate(reversed(jj)):  # crossline increasing upwards, like the time-slice view
        for c, ix in enumerate(ii):
            ax = axes[r, c]
            if s.valid[ix, jx, k]:
                xy = s.coords[order, ix, jx, k]
                m = s.live[order, ix, jx, k]
                ax.plot(xy[:, 0], xy[:, 1], color="0.6", lw=0.8)
                ax.scatter(xy[~m, 0], xy[~m, 1], c="0.8", s=4)
                ax.scatter(xy[m, 0], xy[m, 1], c=abs_off[m], cmap="viridis", s=6, vmin=0, vmax=abs_off.max())
            ax.set_title(f"il {s.il[ix]:.0f}  xl {s.xl[jx]:.0f}", fontsize=7, color="red" if (ix, jx) == (i, j) else "black")
            ax.axhline(0, color="k", lw=0.4)
            ax.axvline(0, color="k", lw=0.4)
            ax.tick_params(labelsize=6)
    fig.suptitle(f"Trajectories (axis 1 vs axis 2, live offsets coloured by |offset|, muted grey) around the reference at t={s.t[k]:.0f} ms", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "trajectories_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="*_offset_summary.npz from offset_direction_summarizer.py")
    p.add_argument("--figures", type=Path, default=Path("figures"), help="Output directory for PNGs.")
    p.add_argument("--ref", default=None, help="Reference patch indices 'il,xl,t' (default: centre of the valid region).")
    p.add_argument("--clusters", type=int, default=6, help="Number of spherical k-means classes.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", nargs="*", choices=GROUPS, default=None, help="Subset of figure groups to produce.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t0 = time.time()
    s = Summary(args.input)
    log.info("Loaded %s: patch grid %s, %d offsets, %d/%d valid patches",
             args.input, s.shape, len(s.offsets), s.valid.sum(), s.valid.size)

    if args.ref:
        ref = tuple(int(v) for v in args.ref.split(","))
        if len(ref) != 3 or not all(0 <= r < n for r, n in zip(ref, s.shape)):
            raise SystemExit(f"--ref must be 'il,xl,t' within {s.shape}")
        if not s.valid[ref]:
            log.warning("Reference patch %s is masked (no offset variation)", ref)
    else:
        ref = tuple(int(round(np.median(ax))) for ax in np.nonzero(s.valid))
    log.info("Reference patch il=%d xl=%d t=%d (inline %.0f, crossline %.0f, %.0f ms)",
             *ref, s.il[ref[0]], s.xl[ref[1]], s.t[ref[2]])

    groups = set(args.only) if args.only else set(GROUPS)
    args.figures.mkdir(parents=True, exist_ok=True)
    steps = {
        "scalars": lambda: plot_scalar_slices(s, args.figures),
        "cosine": lambda: plot_reference_cosine(s, ref, args.figures),
        "pca": lambda: plot_pca_rgb(s, ref, args.figures),
        "clusters": lambda: plot_clusters(s, ref, args.figures, args.clusters, args.seed),
        "coherence": lambda: plot_coherence(s, args.figures),
        "trajectories": lambda: plot_trajectories(s, ref, args.figures),
    }
    for name in GROUPS:
        if name in groups:
            steps[name]()
            log.info("%s done (%.0fs)", name, time.time() - t0)

    log.info("Figures written to %s", args.figures.resolve())
    return 0


if __name__ == "__main__":
    sys.exit(main())
