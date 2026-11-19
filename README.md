# Seismic foundation model for pre-stack data

Experiments in applying the [NCS-v1-3d-base](https://huggingface.co/NorskRegnesentralSTI/NCS-v1-3d-base)
seismic foundation model (Norsk Regnesentral, 3D ViT, 768-d latent space, 16×16×16 patches,
224³ crops) to **pre-stack** seismic data, and summarising how a patch's embedding moves through
latent space as a function of offset.

## Layout

| Path | Contents |
|---|---|
| `weights/` | `config.json` + `model.safetensors` from the NCS-v1-3d-base Hugging Face repo (not tracked) |
| `data/` | Input pre-stack volume (`.sgz`, [seismic-zfp](https://github.com/equinor/seismic-zfp) compressed) and derived `.npz` products (not tracked) |
| `embedders/prestack_patch_embedder.py` | Step 1 – per-offset patch embeddings |
| `embedders/patch_offset_rms.py` | Step 1b – per-patch, per-offset RMS of the input data (live-offset / mute detection) |
| `embedders/offset_direction_summarizer.py` | Step 2 – zero-offset vector + 2-D offset basis per patch |
| `visualization/plot_offset_summary.py` | Step 3 – PNG figures of the summary (`figures/`, not tracked) |

Current dataset: `ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time.sgz` — 193 inlines × 303 crosslines ×
1126 samples (4 ms) × 100 offsets (−4950 … +4950 m, 100 m steps; no exact zero offset).
Note the acquisition/processing footprint in offset: the ±50 m bins are **near-offset tapered**
(RMS ≈ 40 % of the ±350–550 m plateau) and far offsets are **stretch-muted** — live offsets reach
only ≈ 950 m at 350 ms, ≈ 3500 m at 2300 ms, ≈ 4600 m below 3500 ms.

## Setup

```powershell
python -m venv .venv --system-site-packages   # inherits an existing seismic-zfp / numpy install
.\.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.\.venv\Scripts\python.exe -m pip install "git+https://github.com/NorskRegnesentral/NCS_models"
```

The NCS package provides `ViT3DModel`; its heavier inference extras (zarr/xarray/dask) are not
needed — the tiling/blending logic is re-implemented in the embedder.

## Step 1 – patch embeddings per offset

```powershell
.\.venv\Scripts\python.exe embedders\prestack_patch_embedder.py `
    --input data\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time.sgz `
    --weights weights `
    --output data\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time_ncs3d_patch_embeddings.npz
```

For each common-offset volume `(il, xl, t)`:

1. Zero-pad to a whole 16-sample patch grid, split into 224³ crops overlapping by 7 patches
   (ramp blending), identical to the reference `NCS.inference` pipeline.
2. Normalise by the cube standard deviation and clip at ±3σ. **The std is global across all
   offsets** (estimated from 10 random offset volumes, dead traces excluded) so relative
   amplitude / AVO behaviour between offsets is preserved.
3. Run `ViT3DModel` with patch-centre `volume_indices` (needed for its LieRE positional
   encoding), drop the CLS token, reorder tokens `(t, il, xl) → (il, xl, t)` and blend
   overlapping crops.
4. Stream each offset's block straight into the `.npz`, so memory is flat in the number of offsets.

Output arrays (float16 by default):

| Key | Shape | Meaning |
|---|---|---|
| `embeddings` | `(n_off, n_il, n_xl, n_t, 768)` | patch embeddings, here `(100, 13, 19, 71, 768)` |
| `offsets`, `offset_index` | `(n_off,)` | offset header value / index in the SGZ |
| `il_patch_start`, `xl_patch_start`, `t_patch_start` | per axis | 0-based first sample of each patch |
| `inline_center`, `xline_center`, `time_center` | per axis | header coordinates of patch centres |
| `normalize_std`, `clip_sigma`, `num_overlap_patches`, … | scalars | run metadata |

Runtime on an RTX 5000 Ada (fp16): ~9 min for 100 offsets (2000 crops); file ≈ 2.5 GB.
Correctness was checked against direct model forward passes (fp32, no overlap: max diff ~1e-5).

Useful flags: `--offsets 0:100:5`, `--num-overlap-patches 0` (≈2× faster), `--dtype float32`,
`--store-dtype float32`, `--normalize-std <value>` to reuse a previous run's statistics.

## Step 1b – live-offset detection

```powershell
.\.venv\Scripts\python.exe embedders\patch_offset_rms.py --input data\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time.sgz
# -> data\..._patch_rms.npz : rms (n_off, n_il, n_xl, n_t) on the same patch grid  (~90 s)
```

## Step 2 – zero-offset vector and 2-D offset basis

```powershell
.\.venv\Scripts\python.exe embedders\offset_direction_summarizer.py `
    --input data\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time_ncs3d_patch_embeddings.npz `
    --rms   data\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time_patch_rms.npz
# -> data\..._ncs3d_offset_summary.npz
```

For every patch, the 100 embeddings `e_h` (one per offset `h`) trace a trajectory in the 768-d
latent space. The script describes it with a point and an orthonormal 2-D basis. With `--rms`,
an offset is **live** for a patch when its RMS is ≥ `--live-threshold` (0.2) × the patch's
maximum RMS over offsets; the fit, PCA and variance statistics use live offsets only, so muted
far offsets / dead traces do not define the directions. Without `--rms` all offsets are used —
on this dataset that makes axis 1 largely "the direction from data to no-data".

| Key | Shape | Meaning |
|---|---|---|
| `zero_offset` | `(…, 768)` | mean of the −50 m and +50 m embeddings (no exact zero offset exists; note these bins are tapered) |
| `offset_direction` | `(…, 768)` | **axis 1**: unit slope `b/‖b‖` of the least-squares fit `e ≈ a + b·|h|` over the live offsets. Regressing on `|h|` makes the sign unambiguous: it points the way the embedding moves as offset grows. |
| `offset_curvature` | `(…, 768)` | **axis 2**: top principal component of the across-offset variation after projecting axis 1 out; orthogonal to axis 1, oriented so its coordinate correlates positively with centred `|h|²` (points the way the trajectory bows at far offsets). |
| `offset_coords` | `(n_off, …, 2)` | coordinates of `e_h − zero_offset` in the basis (all offsets): `e_h ≈ zero_offset + c₁·direction + c₂·curvature` |
| `offset_slope` | `(…)` | `‖b‖`, latent displacement per metre of offset |
| `offset_fit_r2` | `(…)` | fraction of across-offset variance explained by the *linear* fit in `|h|` |
| `offset_axis_var` | `(…, 2)` | fraction of across-offset variance along each axis; the sum is the fraction captured by the plane |
| `live_offsets`, `n_live_offsets` | `(n_off, …)`, `(…)` | live-offset mask and count; patches with < `--min-live` (4) live offsets are left as zeros |

Flags: `--exclude-zero-offset` (leave ±50 m out of the fit), `--live-threshold`, `--min-live`,
`--store-dtype float16`, `--chunk`.

### Findings on the current dataset

Mute-aware fit (`--rms`, threshold 0.2; median 41 live offsets per patch, 3 % of patches unfittable):

| | q10 | median | q90 |
|---|---|---|---|
| linear-in-`|h|` fit R² | 0.18 | 0.28 | 0.45 |
| variance along axis 1 | 0.21 | 0.33 | 0.52 |
| variance along axis 2 | 0.14 | 0.21 | 0.32 |
| variance in the 2-D plane | 0.43 | **0.57** | 0.73 |

For comparison, fitting over *all* offsets gives median plane variance 0.72 (axis 1 0.54) — the
difference is the trivial variance between live data and muted zeros, which the mute-aware fit
removes.

- The mean trajectory (see `figures/clusters_trajectories.png`) is the same for almost all
  classes: a sharp excursion over the tapered ±50 → ±350 m bins, then a long, nearly straight
  run along axis 1 with a bowl on axis 2 (≈ −6 at mid offsets), turning back near the mute.
  Spherical k-means on `offset_direction` therefore separates *directions* in latent space
  (centroid cosines ≤ 0.56) that share a common *shape*; the classes form layer-conformable
  bands in section and coherent patches in map view.
- Fitting positive and negative offsets separately (all-offset fit) gives directions with median
  cosine 0.94 (q10 0.60), so regressing on `|h|` is a reasonable default.

## Step 3 – figures

```powershell
.\.venv\Scripts\python.exe visualization\plot_offset_summary.py `
    --input data\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time_ncs3d_offset_summary.npz [--ref il,xl,t] [--clusters 6] [--only ...]
```

Writes to `figures/`: scalar diagnostic slices in all three orientations (`scalars_*.png`),
cosine similarity to a reference patch for each 768-d field (`cosine_*.png`), PCA→RGB overviews
(`pca_rgb_*.png`), spherical k-means classes of `offset_direction` with mean trajectory per class
(`clusters_*.png`), neighbour-cosine coherence (`coherence_*.png`) and per-patch trajectories
for the reference column and its neighbourhood (`trajectories_*.png`). Muted offsets are drawn
faint/grey in the trajectory plots.

## Possible next steps

- Replace the tapered ±50 m `zero_offset` by the intercept of the fit over un-tapered live
  offsets, and/or start the regression at ≈ 300 m.
- Separate positive/negative offset bases for the asymmetric patches.
- Per-offset normalisation (instead of a global std) to isolate waveform/shape changes from
  amplitude-vs-offset.
- Cluster on trajectory *shape* (`offset_coords` over live offsets) rather than on direction.
