"""Compute per-patch, per-offset RMS amplitude of a pre-stack SGZ volume on the embedding patch grid.

Used to detect which offsets carry live data for each patch (near-offset taper, stretch mute,
dead traces), so that ``offset_direction_summarizer.py`` can fit its offset basis over live
offsets only. The grid is the same zero-padded 16x16x16 patch grid as ``prestack_patch_embedder.py``.

Output arrays
-------------
rms            (n_offsets, n_il_patches, n_xl_patches, n_t_patches) float32
offsets        (n_offsets,) offset header value of each common-offset volume
offset_index   (n_offsets,)
il_patch_start, xl_patch_start, t_patch_start, patch_size, input_path

Example
-------
python embedders\\patch_offset_rms.py --input data\\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time.sgz
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

import numpy as np
from seismic_zfp.read import SgzReader
from tqdm import tqdm


log = logging.getLogger("patch_offset_rms")


def patch_rms(volume: np.ndarray, patch_size: int) -> np.ndarray:
    """RMS over each patch of a zero-padded (il, xl, t) volume -> (n_il, n_xl, n_t) patches."""
    n = [math.ceil(d / patch_size) for d in volume.shape]
    padded = np.zeros([k * patch_size for k in n], dtype=np.float32)
    padded[: volume.shape[0], : volume.shape[1], : volume.shape[2]] = volume
    blocks = padded.reshape(n[0], patch_size, n[1], patch_size, n[2], patch_size)
    return np.sqrt(np.square(blocks, dtype=np.float64).mean(axis=(1, 3, 5))).astype(np.float32)


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True, type=Path, help="Pre-stack (4D) .sgz file.")
    p.add_argument("--output", type=Path, default=None, help="Output .npz (default: <input stem>_patch_rms.npz next to the input).")
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--force", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    output = args.output or args.input.with_name(args.input.stem + "_patch_rms.npz")
    if output.exists() and not args.force:
        raise SystemExit(f"{output} exists; pass --force to overwrite")

    t0 = time.time()
    reader = SgzReader(str(args.input))
    if not reader.is_4d:
        raise SystemExit(f"{args.input} is not a pre-stack (4D) SGZ file")
    rms = np.stack([patch_rms(reader.read_offset(i), args.patch_size) for i in tqdm(range(reader.n_offsets), desc="Patch RMS", unit="offset")])
    reader.close()

    np.savez(
        output,
        rms=rms,
        offsets=np.asarray(reader.offsets),
        offset_index=np.arange(reader.n_offsets),
        il_patch_start=np.arange(rms.shape[1]) * args.patch_size,
        xl_patch_start=np.arange(rms.shape[2]) * args.patch_size,
        t_patch_start=np.arange(rms.shape[3]) * args.patch_size,
        patch_size=args.patch_size,
        input_path=str(args.input),
    )
    log.info("Wrote %s with rms %s in %.0fs", output, rms.shape, time.time() - t0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
