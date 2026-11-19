"""Compute NCS-v1-3d-base patch embeddings for every offset of a pre-stack SGZ volume.

Each common-offset volume ``(inline, crossline, time)`` is zero-padded to a whole number
of 16x16x16 patches, tiled into 224x224x224 crops (optionally overlapping), and run
through the NCS 3D ViT. The patch tokens of overlapping crops are blended back into a
single patch-space feature cube per offset, following the reference NCS inference
pipeline. Results for all offsets are streamed into one ``.npz`` file.

Output arrays
-------------
embeddings       (n_offsets, n_il_patches, n_xl_patches, n_t_patches, hidden_size)
offsets          (n_offsets,)  offset header value of each common-offset volume
offset_index     (n_offsets,)  index of each offset in the SGZ file
il_patch_start   (n_il_patches,) 0-based inline index of the first sample in each patch
xl_patch_start   (n_xl_patches,) 0-based crossline index of the first sample in each patch
t_patch_start    (n_t_patches,)  0-based time-sample index of the first sample in each patch
inline_center    (n_il_patches,) inline number at each patch centre
xline_center     (n_xl_patches,) crossline number at each patch centre
time_center      (n_t_patches,)  time/depth at each patch centre (header units)
normalize_mean, normalize_std, clip_sigma, normalization_mode, patch_size, crop_size,
num_overlap_patches, overlap_filter, model_path, input_path, model_dtype: run metadata

Example
-------
python embedders\\prestack_patch_embedder.py ^
    --input data\\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time.sgz ^
    --weights weights ^
    --output embeddings\\ST0202R08_patch_embeddings.npz

Requires ``torch``, ``seismic-zfp`` and the NCS model package
(``pip install git+https://github.com/NorskRegnesentral/NCS_models``). Tiling and blending
follow ``NCS.inference`` (Apache-2.0, Norsk Regnesentral) and are reimplemented here to avoid
its zarr/xarray/dask dependencies.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.lib import format as npformat
from seismic_zfp.read import SgzReader
from tqdm import tqdm

from NCS.models.vit3d import ViT3DModel


log = logging.getLogger("prestack_patch_embedder")

TORCH_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
OVERLAP_FILTER_CHOICES = ("exponential", "gaussian", "ramp", "uniform", "center")


# ---------------------------------------------------------------------------
# Tiling (equivalent to NCS.inference.tiling, inlined to avoid its zarr/xarray/dask imports)
# ---------------------------------------------------------------------------


@dataclass
class TilePositions:
    starts: list[int]  # tile start positions in the padded axis (samples)
    n_patches_total: int  # patches covered by all tiles
    pad_before: int  # samples of zero padding before the data
    pad_after: int


def compute_tile_positions(
    axis_length: int, crop_size: int, patch_size: int, num_overlap_patches: int
) -> TilePositions:
    """Place overlapping tiles so they cover the axis, splitting extra padding before/after."""
    patches_per_crop = crop_size // patch_size
    if num_overlap_patches >= patches_per_crop:
        raise ValueError(f"num_overlap_patches ({num_overlap_patches}) must be < {patches_per_crop}")
    stride_patches = patches_per_crop - num_overlap_patches
    actual_patches = math.ceil(axis_length / patch_size)
    extra = max(actual_patches - patches_per_crop, 0)
    num_tiles = max(1, math.ceil(extra / stride_patches) + 1)
    n_patches_total = patches_per_crop + stride_patches * (num_tiles - 1)
    total_pad = n_patches_total - actual_patches
    return TilePositions(
        starts=[t * stride_patches * patch_size for t in range(num_tiles)],
        n_patches_total=n_patches_total,
        pad_before=(total_pad // 2) * patch_size,
        pad_after=(total_pad - total_pad // 2) * patch_size,
    )


def _raw_patch_profile(patches_per_crop: int, overlap_filter: str) -> np.ndarray:
    if overlap_filter == "uniform":
        return np.ones(patches_per_crop, dtype=np.float32)
    if overlap_filter == "gaussian":
        d = np.abs(np.arange(patches_per_crop, dtype=np.float32) - (patches_per_crop - 1) / 2)
        d = d / d.max() if d.max() > 0 else d
        return np.exp(-3.0 * d**2).astype(np.float32)
    idx = np.arange(1, patches_per_crop + 1, dtype=np.float32)
    edge = np.minimum(idx, idx[::-1]) / max(math.ceil(patches_per_crop / 2), 1)
    if overlap_filter == "ramp":
        return edge
    if overlap_filter == "exponential":
        return ((np.exp(4.0 * edge) - 1.0) / (np.exp(4.0) - 1.0)).astype(np.float32)
    raise ValueError(f"Unsupported overlap_filter {overlap_filter!r}")


def compute_tile_patch_overlap_weights(
    num_tiles: int, patches_per_crop: int, num_overlap_patches: int, overlap_filter: str
) -> np.ndarray:
    """Per-tile, per-patch blending weights (num_tiles, patches_per_crop) that sum to 1 per patch."""
    if num_overlap_patches == 0:
        return np.ones((num_tiles, patches_per_crop), dtype=np.float32)
    stride = patches_per_crop - num_overlap_patches
    starts = np.arange(num_tiles) * stride
    centers = starts + (patches_per_crop - 1) / 2
    profile = None if overlap_filter == "center" else _raw_patch_profile(patches_per_crop, overlap_filter)
    weights = np.zeros((num_tiles, patches_per_crop), dtype=np.float32)
    for patch in range(patches_per_crop + stride * (num_tiles - 1)):
        covering = np.flatnonzero((starts <= patch) & (patch < starts + patches_per_crop))
        if profile is None:
            scores = np.zeros(len(covering), dtype=np.float32)
            scores[np.argmin(np.abs(centers[covering] - patch))] = 1.0
        else:
            scores = profile[patch - starts[covering]]
        scores = scores / scores.sum()
        weights[covering, patch - starts[covering]] = scores
    return weights


@dataclass
class AxisTiling:
    """Tiling of one volume axis into overlapping crops."""

    length: int
    tiles: TilePositions
    weights: np.ndarray  # (num_tiles, patches_per_crop) blending weights
    n_patches: int  # number of (possibly partial) patches covering the real data
    patch_offset: int  # index of the first real patch within the padded patch grid
    padded_length: int


def build_axis_tiling(
    length: int, crop_size: int, patch_size: int, num_overlap_patches: int, overlap_filter: str
) -> AxisTiling:
    tiles = compute_tile_positions(length, crop_size, patch_size, num_overlap_patches)
    patches_per_crop = crop_size // patch_size
    weights = compute_tile_patch_overlap_weights(
        len(tiles.starts), patches_per_crop, num_overlap_patches, overlap_filter
    )
    return AxisTiling(
        length=length,
        tiles=tiles,
        weights=weights,
        n_patches=math.ceil(length / patch_size),
        patch_offset=tiles.pad_before // patch_size,
        padded_length=tiles.n_patches_total * patch_size,
    )


def pad_volume(volume: np.ndarray, axes: tuple[AxisTiling, AxisTiling, AxisTiling]) -> np.ndarray:
    padded = np.zeros(tuple(a.padded_length for a in axes), dtype=np.float32)
    padded[
        axes[0].tiles.pad_before : axes[0].tiles.pad_before + volume.shape[0],
        axes[1].tiles.pad_before : axes[1].tiles.pad_before + volume.shape[1],
        axes[2].tiles.pad_before : axes[2].tiles.pad_before + volume.shape[2],
    ] = volume
    return padded


def iter_crops(padded: np.ndarray, axes: tuple[AxisTiling, AxisTiling, AxisTiling], crop_size: int):
    """Yield ``(crop, (il_tile, xl_tile, t_tile))`` with crop in seismic (il, xl, t) order."""
    for i, il0 in enumerate(axes[0].tiles.starts):
        for j, xl0 in enumerate(axes[1].tiles.starts):
            for k, t0 in enumerate(axes[2].tiles.starts):
                yield padded[il0 : il0 + crop_size, xl0 : xl0 + crop_size, t0 : t0 + crop_size], (i, j, k)


def batched(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------


def compute_cube_stats(reader: SgzReader, n_offsets: int, seed: int) -> tuple[float, float]:
    """Approximate mean/std of the pre-stack cube from a random sample of common-offset volumes.

    Dead (all-zero) traces are excluded so the padding of missing traces does not bias the estimate.
    """
    rng = np.random.default_rng(seed)
    n_offsets = min(n_offsets, reader.n_offsets)
    indices = np.sort(rng.choice(reader.n_offsets, size=n_offsets, replace=False))

    running_sum = 0.0
    running_sq = 0.0
    count = 0
    for idx in tqdm(indices, desc="Cube statistics", unit="offset", leave=False):
        volume = reader.read_offset(int(idx))
        live = volume[np.any(volume != 0, axis=-1)].astype(np.float64)
        running_sum += live.sum()
        running_sq += np.square(live).sum()
        count += live.size
    if count == 0:
        raise ValueError("No live traces found while computing cube statistics")
    mean = running_sum / count
    std = math.sqrt(max(running_sq / count - mean**2, 0.0))
    return float(mean), float(std)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


def load_model(weights: Path, device: torch.device, dtype: torch.dtype) -> ViT3DModel:
    model, info = ViT3DModel.from_pretrained(str(weights), torch_dtype=dtype, output_loading_info=True)
    for key in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        if info.get(key):
            log.warning("%s when loading weights: %s", key, info[key])
    return model.to(device).eval()


def volume_index_tensors(device: torch.device, crop_size: int, patch_size: int, batch_size: int):
    """Patch-centre coordinates passed to the model for its LieRE positional encoding."""
    patches_per_crop = crop_size // patch_size
    centers = torch.arange(0, crop_size, patch_size, device=device) + patch_size // 2
    volume_indices = (
        torch.cartesian_prod(centers, centers, centers)
        .reshape(patches_per_crop, patches_per_crop, patches_per_crop, 3)
        .expand(batch_size, 1, -1, -1, -1, -1)
    )
    volume_size = torch.full((3,), crop_size, device=device, dtype=torch.long)
    return volume_indices, volume_size


@dataclass
class Normalization:
    enabled: bool
    mean: float
    std: float
    mode: str
    clip_sigma: float

    def apply_(self, x: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return x
        if self.mode == "zscore":
            x.sub_(self.mean)
        x.div_(self.std + 1e-10)
        return x.clamp_(-self.clip_sigma, self.clip_sigma)


@torch.inference_mode()
def embed_volume(
    model: ViT3DModel,
    volume: np.ndarray,
    axes: tuple[AxisTiling, AxisTiling, AxisTiling],
    crop_size: int,
    patch_size: int,
    norm: Normalization,
    device: torch.device,
    dtype: torch.dtype,
    batch_size: int,
    progress: tqdm | None = None,
) -> np.ndarray:
    """Return blended patch embeddings of shape (n_il_patches, n_xl_patches, n_t_patches, hidden)."""
    patches_per_crop = crop_size // patch_size
    hidden = model.config.hidden_size
    padded = pad_volume(volume, axes)

    feature_sum = torch.zeros(
        (axes[0].tiles.n_patches_total, axes[1].tiles.n_patches_total, axes[2].tiles.n_patches_total, hidden),
        dtype=torch.float32,
        device=device,
    )
    weight_sum = torch.zeros(feature_sum.shape[:3], dtype=torch.float32, device=device)
    weights = [torch.as_tensor(a.weights, device=device) for a in axes]
    all_vi, volume_size = volume_index_tensors(device, crop_size, patch_size, batch_size)

    for batch in batched(iter_crops(padded, axes, crop_size), batch_size):
        crops = np.stack([c for c, _ in batch])  # (B, il, xl, t)
        # Model expects (B, 1, t, il, xl); normalisation is applied after zero padding as in the reference.
        pixel_values = torch.from_numpy(crops).to(device, non_blocking=True).movedim(-1, 1).unsqueeze(1)
        pixel_values = norm.apply_(pixel_values).to(dtype)

        outputs = model(pixel_values=pixel_values, volume_indices=all_vi[: len(batch)], volume_size=volume_size)
        feats = outputs.last_hidden_state[:, 1:, :].float()  # drop CLS -> (B, P^3, H)
        feats = feats.reshape(len(batch), patches_per_crop, patches_per_crop, patches_per_crop, hidden)
        feats = feats.permute(0, 2, 3, 1, 4)  # (B, t, il, xl, H) -> (B, il, xl, t, H)

        for b, (_, (i, j, k)) in enumerate(batch):
            w3d = torch.einsum("i,j,k->ijk", weights[0][i], weights[1][j], weights[2][k])
            il_sl = slice(axes[0].tiles.starts[i] // patch_size, axes[0].tiles.starts[i] // patch_size + patches_per_crop)
            xl_sl = slice(axes[1].tiles.starts[j] // patch_size, axes[1].tiles.starts[j] // patch_size + patches_per_crop)
            t_sl = slice(axes[2].tiles.starts[k] // patch_size, axes[2].tiles.starts[k] // patch_size + patches_per_crop)
            feature_sum[il_sl, xl_sl, t_sl] += feats[b] * w3d.unsqueeze(-1)
            weight_sum[il_sl, xl_sl, t_sl] += w3d

        if progress is not None:
            progress.update(len(batch))

    real = tuple(slice(a.patch_offset, a.patch_offset + a.n_patches) for a in axes)
    features = feature_sum[real] / weight_sum[real].clamp_min(1e-12).unsqueeze(-1)
    return features.cpu().numpy()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_offsets(spec: str | None, n_offsets: int) -> list[int]:
    """Parse ``start:stop[:step]`` or a comma-separated list of offset indices."""
    if spec is None:
        return list(range(n_offsets))
    if ":" in spec:
        parts = [int(p) if p else None for p in spec.split(":")]
        indices = list(range(n_offsets))[slice(*parts)]
    else:
        indices = [int(p) for p in spec.split(",") if p.strip()]
    for idx in indices:
        if not 0 <= idx < n_offsets:
            raise ValueError(f"offset index {idx} out of range [0, {n_offsets})")
    return indices


def patch_centers(first: float, step: float, starts: np.ndarray, patch_size: int) -> np.ndarray:
    return first + (starts + patch_size // 2) * step


def axis_step(values: np.ndarray) -> float:
    return float(values[1] - values[0]) if len(values) > 1 else 1.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Pre-stack (4D) .sgz file.")
    p.add_argument("--output", required=True, type=Path, help="Output .npz file.")
    p.add_argument("--weights", type=Path, default=Path("weights"), help="Directory with config.json + model.safetensors.")
    p.add_argument("--offsets", default=None, help="Offset indices to embed: 'start:stop[:step]' or 'i,j,k'. Default: all.")
    p.add_argument("--crop-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--num-overlap-patches", type=int, default=7, help="Patch overlap between adjacent crops.")
    p.add_argument("--overlap-filter", choices=OVERLAP_FILTER_CHOICES, default="ramp")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", choices=["auto", *TORCH_DTYPES], default="auto", help="Model compute dtype (auto: float16 on CUDA, else float32).")
    p.add_argument("--store-dtype", choices=["float16", "float32"], default="float16", help="dtype of the stored embeddings.")
    p.add_argument("--compress", action="store_true", help="Deflate-compress the npz (slower, smaller).")
    p.add_argument("--no-normalize", action="store_true")
    p.add_argument("--normalization-mode", choices=["std", "zscore"], default="std")
    p.add_argument("--clip-sigma", type=float, default=3.0)
    p.add_argument("--normalize-std", type=float, default=None, help="Use this std instead of estimating it from the data.")
    p.add_argument("--normalize-mean", type=float, default=0.0, help="Mean used with --normalize-std in zscore mode.")
    p.add_argument("--n-stat-offsets", type=int, default=10, help="Common-offset volumes sampled to estimate mean/std.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Overwrite an existing output file.")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.crop_size % args.patch_size:
        raise SystemExit("--crop-size must be divisible by --patch-size")
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} exists; pass --force to overwrite")

    device = torch.device(args.device)
    dtype = TORCH_DTYPES[args.dtype] if args.dtype != "auto" else (torch.float16 if device.type == "cuda" else torch.float32)

    reader = SgzReader(str(args.input))
    if not reader.is_4d:
        raise SystemExit(f"{args.input} is not a pre-stack (4D) SGZ file")
    offset_indices = parse_offsets(args.offsets, reader.n_offsets)
    n_il, n_xl, n_t = reader.n_ilines, reader.n_xlines, reader.n_samples
    log.info("Volume: %d inlines x %d crosslines x %d samples x %d offsets (embedding %d offsets)",
             n_il, n_xl, n_t, reader.n_offsets, len(offset_indices))

    if args.no_normalize:
        norm = Normalization(False, 0.0, 1.0, args.normalization_mode, args.clip_sigma)
    elif args.normalize_std is not None:
        norm = Normalization(True, args.normalize_mean, args.normalize_std, args.normalization_mode, args.clip_sigma)
    else:
        t0 = time.time()
        mean, std = compute_cube_stats(reader, args.n_stat_offsets, args.seed)
        log.info("Cube statistics from %d offset volumes: mean=%.6g std=%.6g (%.1fs)",
                 min(args.n_stat_offsets, reader.n_offsets), mean, std, time.time() - t0)
        norm = Normalization(True, mean, std, args.normalization_mode, args.clip_sigma)

    axes = tuple(
        build_axis_tiling(n, args.crop_size, args.patch_size, args.num_overlap_patches, args.overlap_filter)
        for n in (n_il, n_xl, n_t)
    )
    crops_per_offset = math.prod(len(a.tiles.starts) for a in axes)
    log.info("Patch grid %s, %d crops per offset", tuple(a.n_patches for a in axes), crops_per_offset)

    log.info("Loading model from %s on %s (%s)", args.weights, device, dtype)
    model = load_model(args.weights, device, dtype)
    hidden = model.config.hidden_size
    store_dtype = np.dtype(args.store_dtype)

    emb_shape = (len(offset_indices), axes[0].n_patches, axes[1].n_patches, axes[2].n_patches, hidden)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    compression = zipfile.ZIP_DEFLATED if args.compress else zipfile.ZIP_STORED
    progress = tqdm(total=len(offset_indices) * crops_per_offset, desc="Embedding", unit="crop")

    t_start = time.time()
    with (
        zipfile.ZipFile(args.output, "w", compression=compression, allowZip64=True) as zf,
        ThreadPoolExecutor(max_workers=1) as pool,
    ):
        # Stream the big array one offset at a time into a single .npy member.
        with zf.open("embeddings.npy", "w", force_zip64=True) as fh:
            npformat.write_array_header_1_0(
                fh, {"descr": npformat.dtype_to_descr(store_dtype), "fortran_order": False, "shape": emb_shape}
            )
            future = pool.submit(reader.read_offset, offset_indices[0])
            for n, offset_idx in enumerate(offset_indices):
                volume = future.result()
                if n + 1 < len(offset_indices):
                    future = pool.submit(reader.read_offset, offset_indices[n + 1])
                progress.set_postfix(offset=int(reader.offsets[offset_idx]))
                features = embed_volume(
                    model, volume, axes, args.crop_size, args.patch_size, norm, device, dtype, args.batch_size, progress
                )
                fh.write(np.ascontiguousarray(features.astype(store_dtype, copy=False)).tobytes())

        il_starts = np.arange(axes[0].n_patches) * args.patch_size
        xl_starts = np.arange(axes[1].n_patches) * args.patch_size
        t_starts = np.arange(axes[2].n_patches) * args.patch_size
        metadata = {
            "offsets": np.asarray(reader.offsets)[offset_indices],
            "offset_index": np.asarray(offset_indices),
            "il_patch_start": il_starts,
            "xl_patch_start": xl_starts,
            "t_patch_start": t_starts,
            "inline_center": patch_centers(reader.ilines[0], axis_step(reader.ilines), il_starts, args.patch_size),
            "xline_center": patch_centers(reader.xlines[0], axis_step(reader.xlines), xl_starts, args.patch_size),
            "time_center": patch_centers(reader.zslices[0], axis_step(reader.zslices), t_starts, args.patch_size),
            "normalize_mean": norm.mean,
            "normalize_std": norm.std,
            "clip_sigma": norm.clip_sigma,
            "normalization_mode": norm.mode if norm.enabled else "none",
            "patch_size": args.patch_size,
            "crop_size": args.crop_size,
            "num_overlap_patches": args.num_overlap_patches,
            "overlap_filter": args.overlap_filter,
            "model_path": str(args.weights),
            "model_dtype": str(dtype).removeprefix("torch."),
            "input_path": str(args.input),
        }
        for name, value in metadata.items():
            with zf.open(f"{name}.npy", "w") as fh:
                npformat.write_array(fh, np.asarray(value), allow_pickle=False)

    progress.close()
    reader.close()
    log.info("Wrote %s with embeddings %s (%s) in %.1fs", args.output, emb_shape, store_dtype, time.time() - t_start)
    return 0


if __name__ == "__main__":
    sys.exit(main())
