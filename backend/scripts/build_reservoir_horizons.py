"""Build the Volve "ground truth" reservoir horizons (Hugin Fm.) in PP time on a survey's IL/XL grid.

Sources (all open Volve data, Equinor Volve licence; downloaded on first run into ``--work-dir``):

1. Top Hugin TWT horizon, ``x y twt`` points interpreted on the Volve seismic
   (MadsLorentzen/seis_viz mirror of the Volve data village).
2. Volve well picks (Open Well Data standardised copy of
   ``Geophysical_Interpretations/Wells/Well_picks_Volve_v1.dat``, Zenodo 10.5281/zenodo.24271246):
   every pick carries TVDSS, TWT and easting/northing, giving a time-depth tie and the
   Hugin interval velocity.
3. Volve Eclipse corner-point reservoir grid + ACTNUM (dabiged/Volve2OPM mirror of the
   ``VOLVE_2016`` model), i.e. *where the reservoir model actually is*, in depth.

Depth -> time: the per-column top of the active reservoir cells coincides with the interpreted
Top Hugin horizon (checked against the well picks, |dz| < 10 m at 13/16 wells), so the reservoir
top is placed on the interpreted TWT horizon where it exists and on ``2 z / V_avg`` elsewhere,
with ``V_avg = 2 z_top / twt_top`` gap-filled by diffusion.  The base is ``twt_top + 2 dz / V_int``
with ``V_int`` the median Hugin interval velocity from the Top/Base Hugin well-pick pairs.

Run from ``backend/`` (needs pyarrow for the Parquet well picks, not part of the backend deps):

    uv run --with pyarrow python scripts/build_reservoir_horizons.py \
        --survey ../data/ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time.sgz

Writes ``<survey>_horizons.npz`` next to the survey; the backend serves it at
``/api/surveys/{id}/horizons``.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import segyio
from seismic_zfp.read import SgzReader

SOURCES = {
    "horizon": (
        "hor_twt_hugin_fm_top.dat",
        "https://raw.githubusercontent.com/MadsLorentzen/seis_viz/"
        "0896899dbbbca2d358c54f984a3c319ff6372928/data/hor_twt_hugin_fm_top.dat",
    ),
    "tops": (
        "volve-tops-v16.zip",
        "https://zenodo.org/api/records/24271246/files/volve-tops-v16.zip/content",
    ),
    "grid": (
        "GRID_postF1B_Nov2013_locupd_12112013.grdecl",
        "https://raw.githubusercontent.com/dabiged/Volve2OPM/master/Includes/"
        "GRID_postF1B_Nov2013_locupd_12112013.grdecl",
    ),
    "actnum": (
        "ACTNUM_2013.grdecl",
        "https://raw.githubusercontent.com/dabiged/Volve2OPM/master/Includes/ACTNUM_2013",
    ),
}

TOP_FM = "Hugin Fm. VOLVE Top"
BASE_FM = "Hugin Fm. VOLVE Base"
PICK_FORMATIONS = (TOP_FM, BASE_FM, "Sleipner Fm. Top", "Heather Fm. Top", "Draupne Fm. Top")


# ----------------------------------------------------------------------------- downloads


def fetch(work_dir: Path, key: str) -> Path:
    name, url = SOURCES[key]
    path = work_dir / name
    if not path.is_file():
        print(f"Downloading {url}")
        work_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(url) as resp, open(path, "wb") as fh:
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
    return path


# ----------------------------------------------------------------------------- survey geometry


class SurveyGeometry:
    """IL/XL <-> UTM affine map fitted to the CDP_X/CDP_Y trace headers."""

    def __init__(self, sgz_path: Path, header_stride: int = 997) -> None:
        reader = SgzReader(str(sgz_path))
        try:
            self.ilines = np.asarray(reader.ilines, dtype=np.int64)
            self.xlines = np.asarray(reader.xlines, dtype=np.int64)
            self.samples = np.asarray(reader.zslices, dtype=np.float64)
            tf = segyio.TraceField
            rows = []
            for i in range(0, reader.tracecount, header_stride):
                h = reader.gen_trace_header(i)
                scalar = int(h[tf.SourceGroupScalar]) or 1
                mult = scalar if scalar > 0 else 1.0 / -scalar
                rows.append(
                    (h[tf.INLINE_3D], h[tf.CROSSLINE_3D], h[tf.CDP_X] * mult, h[tf.CDP_Y] * mult)
                )
        finally:
            reader.close()
        a = np.asarray(rows, dtype=np.float64)
        design = np.c_[a[:, 0], a[:, 1], np.ones(len(a))]
        cx, *_ = np.linalg.lstsq(design, a[:, 2], rcond=None)
        cy, *_ = np.linalg.lstsq(design, a[:, 3], rcond=None)
        self.A = np.array([[cx[0], cx[1]], [cy[0], cy[1]]])
        self.b = np.array([cx[2], cy[2]])
        self.A_inv = np.linalg.inv(self.A)
        resid = np.hypot(design @ cx - a[:, 2], design @ cy - a[:, 3])
        self.fit_residual_m = float(resid.max())
        self.bin_il_m = float(np.hypot(cx[0], cy[0]) * self.il_step)
        self.bin_xl_m = float(np.hypot(cx[1], cy[1]) * self.xl_step)

    @property
    def il_step(self) -> int:
        return int(self.ilines[1] - self.ilines[0]) if len(self.ilines) > 1 else 1

    @property
    def xl_step(self) -> int:
        return int(self.xlines[1] - self.xlines[0]) if len(self.xlines) > 1 else 1

    @property
    def shape(self) -> tuple[int, int]:
        return len(self.ilines), len(self.xlines)

    def utm_to_ilxl(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v = self.A_inv @ (np.vstack([x, y]) - self.b[:, None])
        return v[0], v[1]

    def utm_to_index(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Nearest grid indices and an in-bounds mask."""
        il, xl = self.utm_to_ilxl(np.asarray(x, float), np.asarray(y, float))
        i = np.rint((il - self.ilines[0]) / self.il_step).astype(np.int64)
        j = np.rint((xl - self.xlines[0]) / self.xl_step).astype(np.int64)
        ok = (i >= 0) & (i < self.shape[0]) & (j >= 0) & (j < self.shape[1])
        return i, j, ok


# ----------------------------------------------------------------------------- gridding helpers


def bin_average(geom: SurveyGeometry, x: np.ndarray, y: np.ndarray, v: np.ndarray) -> np.ndarray:
    i, j, ok = geom.utm_to_index(x, y)
    acc = np.zeros(geom.shape)
    cnt = np.zeros(geom.shape)
    np.add.at(acc, (i[ok], j[ok]), v[ok])
    np.add.at(cnt, (i[ok], j[ok]), 1.0)
    with np.errstate(invalid="ignore"):
        return np.where(cnt > 0, acc / cnt, np.nan)


def dilate(mask: np.ndarray, n: int) -> np.ndarray:
    """Grow ``mask`` by ``n`` bins in all 8 directions."""
    out = mask.copy()
    for _ in range(n):
        padded = np.pad(out, 1)
        grown = out.copy()
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                grown |= padded[1 + di : padded.shape[0] - 1 + di, 1 + dj : padded.shape[1] - 1 + dj]
        out = grown
    return out


def diffusion_fill(arr: np.ndarray, fill_mask: np.ndarray, iters: int = 2000, tol: float = 1e-4) -> np.ndarray:
    """Fill NaNs inside ``fill_mask`` by iterating the discrete Laplace equation (known cells fixed)."""
    known = np.isfinite(arr)
    target = fill_mask & ~known
    if not target.any():
        return arr.copy()
    out = arr.copy()
    out[~known] = np.nanmean(arr)
    for _ in range(iters):
        padded = np.pad(out, 1, mode="edge")
        avg = 0.25 * (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:])
        delta = np.abs(avg[target] - out[target]).max()
        out[target] = avg[target]
        if delta < tol:
            break
    out[~(known | target)] = np.nan
    return out


# ----------------------------------------------------------------------------- source parsers


def read_grdecl_keywords(path: Path, wanted: set[str]) -> dict[str, np.ndarray]:
    txt = re.sub(r"--[^\n]*", "", path.read_text())
    toks = txt.split()
    out: dict[str, np.ndarray] = {}
    i = 0
    while i < len(toks):
        key = toks[i]
        if key in wanted:
            j = toks.index("/", i + 1)
            vals: list[float] = []
            for tok in toks[i + 1 : j]:
                if "*" in tok:
                    n, v = tok.split("*")
                    vals.extend([float(v)] * int(n))
                elif tok[0].isdigit() or tok[0] in "-.":
                    vals.append(float(tok))
            out[key] = np.asarray(vals)
            i = j + 1
        else:
            i += 1
    missing = wanted - out.keys()
    if missing:
        raise ValueError(f"{path.name}: missing keywords {sorted(missing)}")
    return out


def reservoir_columns(grid_path: Path, actnum_path: Path, min_thickness_m: float) -> dict[str, np.ndarray]:
    """Per-column (j, i) UTM centre and depth of the top/base of the active cells."""
    kw = read_grdecl_keywords(grid_path, {"MAPAXES", "SPECGRID", "COORD", "ZCORN"})
    nx, ny, nz = (int(v) for v in kw["SPECGRID"][:3])
    act = read_grdecl_keywords(actnum_path, {"ACTNUM"})["ACTNUM"].reshape(nz, ny, nx) > 0
    coord = kw["COORD"].reshape(ny + 1, nx + 1, 6)
    zc = kw["ZCORN"].reshape(nz, 2, ny, 2, nx, 2)
    z_top = zc[:, 0].mean(axis=(2, 4))
    z_bot = zc[:, 1].mean(axis=(2, 4))
    top = np.where(act, z_top, np.inf).min(axis=0)
    base = np.where(act, z_bot, -np.inf).max(axis=0)
    # Collapsed (zero-thickness) active columns along the grid edges are not reservoir.
    has = np.isfinite(top) & ((base - top) >= min_thickness_m)
    n_active_layers = act.sum(axis=0)

    px, py = coord[..., 0], coord[..., 1]
    cx = 0.25 * (px[:-1, :-1] + px[:-1, 1:] + px[1:, :-1] + px[1:, 1:])
    cy = 0.25 * (py[:-1, :-1] + py[:-1, 1:] + py[1:, :-1] + py[1:, 1:])
    # MAPAXES: point on local Y axis, origin, point on local X axis
    m = kw["MAPAXES"]
    origin = m[2:4]
    ux = (m[4:6] - origin) / np.linalg.norm(m[4:6] - origin)
    uy = (m[0:2] - origin) / np.linalg.norm(m[0:2] - origin)
    utm_x = origin[0] + cx * ux[0] + cy * uy[0]
    utm_y = origin[1] + cx * ux[1] + cy * uy[1]
    cell_size = float(max(np.abs(np.diff(cx, axis=1)).mean(), np.abs(np.diff(cy, axis=0)).mean()))
    return {
        "x": utm_x[has],
        "y": utm_y[has],
        "top": top[has],
        "base": base[has],
        "n_layers": n_active_layers[has],
        "dims": np.array([nx, ny, nz]),
        "cell_size_m": cell_size,
    }


def read_well_picks(zip_path: Path) -> list[dict]:
    import pyarrow.parquet as pq

    with zipfile.ZipFile(zip_path) as zf:
        member = next(n for n in zf.namelist() if n.endswith("data/tops.parquet"))
        with zf.open(member) as fh:
            table = pq.read_table(fh).to_pydict()

    pats = {
        "tvdss": r"TVDSS (-?[\d.]+) m",
        "twt": r"TWT (-?[\d.]+) ms",
        "x": r"Easting ([\d.]+) m",
        "y": r"Northing ([\d.]+) m",
    }
    picks = []
    for well, fm, comment in zip(table["source_well_name"], table["formation"], table["comment"]):
        comment = comment or ""
        vals = {k: re.search(p, comment) for k, p in pats.items()}
        if any(v is None for v in vals.values()):
            continue
        qlf = re.search(r"Qlf ([A-Z]+)", comment)
        picks.append(
            {
                "well": well.removeprefix("NO ").strip(),
                "formation": fm,
                "tvdss": float(vals["tvdss"].group(1)),
                "twt": float(vals["twt"].group(1)),
                "x": float(vals["x"].group(1)),
                "y": float(vals["y"].group(1)),
                "qualifier": qlf.group(1) if qlf else "",
            }
        )
    return picks


def hugin_interval_velocity(picks: list[dict]) -> tuple[float, int]:
    by_well: dict[str, dict[str, dict]] = {}
    for p in picks:
        if p["qualifier"] or p["formation"] not in (TOP_FM, BASE_FM):
            continue
        by_well.setdefault(p["well"], {}).setdefault(p["formation"], p)
    vels = []
    for fms in by_well.values():
        if TOP_FM in fms and BASE_FM in fms:
            dz = fms[TOP_FM]["tvdss"] - fms[BASE_FM]["tvdss"]
            dt = fms[BASE_FM]["twt"] - fms[TOP_FM]["twt"]
            if dz > 5.0 and dt > 2.0:
                vels.append(2000.0 * dz / dt)
    if not vels:
        raise ValueError("No usable Top/Base Hugin pick pairs for the interval velocity")
    return float(np.median(vels)), len(vels)


# ----------------------------------------------------------------------------- main build


def build(args: argparse.Namespace) -> Path:
    work_dir = Path(args.work_dir)
    sgz_path = Path(args.survey)
    geom = SurveyGeometry(sgz_path)
    n_il, n_xl = geom.shape
    print(
        f"Survey {sgz_path.stem}: {n_il} x {n_xl} bins "
        f"({geom.bin_il_m:.2f} m x {geom.bin_xl_m:.2f} m), header fit residual {geom.fit_residual_m:.2f} m"
    )

    # 1. interpreted Top Hugin TWT horizon
    pts = np.loadtxt(fetch(work_dir, "horizon"))
    hz_raw = bin_average(geom, pts[:, 0], pts[:, 1], pts[:, 2])
    hz_mask = np.isfinite(hz_raw)
    hz = diffusion_fill(hz_raw, dilate(hz_mask, args.horizon_fill_cells))
    print(f"Top Hugin horizon: {hz_mask.sum()} bins with picks, {np.isfinite(hz).sum()} after gap fill")

    # 2. well picks -> interval velocity + marker list
    picks = read_well_picks(fetch(work_dir, "tops"))
    v_int, n_pairs = hugin_interval_velocity(picks)
    print(f"Hugin interval velocity: {v_int:.0f} m/s (median of {n_pairs} well Top/Base pairs)")

    # 3. reservoir grid columns -> depth top/base on the IL/XL grid
    cols = reservoir_columns(fetch(work_dir, "grid"), fetch(work_dir, "actnum"), args.min_thickness)
    z_top = bin_average(geom, cols["x"], cols["y"], cols["top"])
    z_base = bin_average(geom, cols["x"], cols["y"], cols["base"])
    # Eclipse cells are coarser than seismic bins: fill the bins out to half a cell from each centre.
    cell_bins = max(1, int(np.ceil(0.5 * cols["cell_size_m"] / min(geom.bin_il_m, geom.bin_xl_m))))
    res_mask = dilate(np.isfinite(z_top), cell_bins)
    z_top = diffusion_fill(z_top, res_mask)
    z_base = diffusion_fill(z_base, res_mask)
    res_mask = np.isfinite(z_top) & np.isfinite(z_base)
    print(
        f"Reservoir model: {res_mask.sum()} bins inside the survey "
        f"(grid {cols['dims'].tolist()}, {cols['cell_size_m']:.1f} m cells)"
    )

    # 4. depth -> time
    both = res_mask & np.isfinite(hz)
    v_avg = np.full(geom.shape, np.nan)
    v_avg[both] = 2.0 * z_top[both] / hz[both]
    v_avg = diffusion_fill(v_avg, res_mask)
    with np.errstate(invalid="ignore", divide="ignore"):
        twt_top = np.where(np.isfinite(hz) & res_mask, hz, 2.0 * z_top / v_avg)
    twt_top[~res_mask] = np.nan
    twt_base = twt_top + 2000.0 * (z_base - z_top) / v_int
    print(
        f"V_avg to Top Hugin: median {np.nanmedian(v_avg[both]) * 1000:.0f} m/s "
        f"(p5-p95 {np.nanpercentile(v_avg[both], 5) * 1000:.0f}-{np.nanpercentile(v_avg[both], 95) * 1000:.0f}); "
        f"{both.sum()} bins tied to the horizon, {(res_mask & ~both).sum()} extrapolated"
    )

    # 5. well markers on the grid + QC
    markers = []
    qc = []
    for p in picks:
        if p["formation"] not in PICK_FORMATIONS:
            continue
        i, j, ok = geom.utm_to_index(np.array([p["x"]]), np.array([p["y"]]))
        if not ok[0]:
            continue
        markers.append(
            {
                "well": p["well"],
                "formation": p["formation"],
                "il_idx": int(i[0]),
                "xl_idx": int(j[0]),
                "twt": p["twt"],
                "tvdss": p["tvdss"],
                "qualifier": p["qualifier"],
            }
        )
        if not p["qualifier"]:
            ref = twt_top if p["formation"] == TOP_FM else twt_base if p["formation"] == BASE_FM else None
            if ref is not None and np.isfinite(ref[i[0], j[0]]):
                qc.append(ref[i[0], j[0]] - p["twt"])
    qc_arr = np.asarray(qc)
    if qc_arr.size:
        print(
            f"QC model vs {qc_arr.size} Top/Base Hugin well picks: median {np.median(qc_arr):+.1f} ms, "
            f"RMS {np.sqrt(np.mean(qc_arr**2)):.1f} ms, max |dt| {np.abs(qc_arr).max():.1f} ms"
        )

    sample_rate = float(geom.samples[1] - geom.samples[0]) if len(geom.samples) > 1 else 1.0
    meta = {
        "survey_id": sgz_path.stem,
        "sample_rate_ms": sample_rate,
        "hugin_interval_velocity_mps": v_int,
        "hugin_interval_velocity_pairs": n_pairs,
        "v_avg_median_mps": float(np.nanmedian(v_avg[both]) * 1000.0),
        "qc_rms_ms": float(np.sqrt(np.mean(qc_arr**2))) if qc_arr.size else None,
        "sources": {k: v[1] for k, v in SOURCES.items()},
        "licence": "Equinor Volve data licence (Equinor ASA and the former Volve licence partners)",
    }
    horizons = [
        ("top_hugin_twt", "Top Hugin (interpreted TWT)", "horizon", "#f472b6", hz),
        ("reservoir_top", "Reservoir model top (Eclipse grid)", "top", "#4ade80", twt_top),
        ("reservoir_base", "Reservoir model base (Eclipse grid)", "base", "#4ade80", twt_base),
    ]
    out_path = Path(args.output) if args.output else sgz_path.with_name(f"{sgz_path.stem}_horizons.npz")
    np.savez_compressed(
        out_path,
        ilines=geom.ilines,
        xlines=geom.xlines,
        horizon_ids=np.array([h[0] for h in horizons]),
        horizon_labels=np.array([h[1] for h in horizons]),
        horizon_kinds=np.array([h[2] for h in horizons]),
        horizon_colors=np.array([h[3] for h in horizons]),
        twt=np.stack([h[4] for h in horizons]).astype(np.float32),
        depth_top=z_top.astype(np.float32),
        depth_base=z_base.astype(np.float32),
        v_avg=(v_avg * 1000.0).astype(np.float32),
        well_picks_json=json.dumps(markers),
        meta_json=json.dumps(meta),
    )
    print(f"Wrote {out_path} ({len(markers)} well markers)")
    return out_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--survey", required=True, help="Path to the ST0202 .sgz survey (IL/XL + CDP headers)")
    parser.add_argument("--work-dir", default=None, help="Download cache (default: <data dir>/volve_reservoir)")
    parser.add_argument("--output", default=None, help="Output .npz (default: <survey stem>_horizons.npz)")
    parser.add_argument(
        "--horizon-fill-cells",
        type=int,
        default=3,
        help="Fill horizon gaps up to this many bins from the nearest pick (default 3)",
    )
    parser.add_argument(
        "--min-thickness",
        type=float,
        default=0.5,
        help="Drop reservoir-grid columns thinner than this (m); removes collapsed edge cells (default 0.5)",
    )
    args = parser.parse_args()
    if args.work_dir is None:
        args.work_dir = str(Path(args.survey).resolve().parent / "volve_reservoir")
    build(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
