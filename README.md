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
| `embedders/offset_direction_summarizer.py` | Step 2 – zero-offset vector + 2-D offset basis per patch |

Current dataset: `ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time.sgz` — 193 inlines × 303 crosslines ×
1126 samples (4 ms) × 100 offsets (−4950 … +4950 m, 100 m steps; no exact zero offset).

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

## Step 2 – zero-offset vector and 2-D offset basis

```powershell
.\.venv\Scripts\python.exe embedders\offset_direction_summarizer.py `
    --input data\ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time_ncs3d_patch_embeddings.npz
# -> data\..._ncs3d_offset_summary.npz
```

For every patch, the 100 embeddings `e_h` (one per offset `h`) trace a trajectory in the 768-d
latent space. The script describes it with a point and an orthonormal 2-D basis:

| Key | Shape | Meaning |
|---|---|---|
| `zero_offset` | `(…, 768)` | mean of the −50 m and +50 m embeddings (no exact zero offset exists) |
| `offset_direction` | `(…, 768)` | **axis 1**: unit slope `b/‖b‖` of the least-squares fit `e ≈ a + b·|h|` over all offsets. Regressing on `|h|` makes the sign unambiguous: it points the way the embedding moves as offset grows. |
| `offset_curvature` | `(…, 768)` | **axis 2**: top principal component of the across-offset variation after projecting axis 1 out; orthogonal to axis 1, oriented so its coordinate correlates positively with centred `|h|²` (points the way the trajectory bows at far offsets). |
| `offset_coords` | `(n_off, …, 2)` | coordinates of `e_h − zero_offset` in the basis: `e_h ≈ zero_offset + c₁·direction + c₂·curvature` |
| `offset_slope` | `(…)` | `‖b‖`, latent displacement per metre of offset |
| `offset_fit_r2` | `(…)` | fraction of across-offset variance explained by the *linear* fit in `|h|` |
| `offset_axis_var` | `(…, 2)` | fraction of across-offset variance along each axis; the sum is the fraction captured by the plane |

Flags: `--exclude-zero-offset` (leave ±50 m out of the fit), `--store-dtype float16`, `--chunk`.

### Findings on the current dataset

| | q10 | median | q90 |
|---|---|---|---|
| linear-in-`|h|` fit R² | 0.20 | 0.39 | 0.57 |
| variance along axis 1 | 0.26 | 0.54 | 0.72 |
| variance along axis 2 | 0.08 | 0.16 | 0.32 |
| variance in the 2-D plane | 0.53 | **0.72** | 0.82 |

- A single direction captures roughly half the offset-dependent variation; the straight-line
  fit in `|h|` is weaker (0.39) because movement along axis 1 is not linear in offset.
- Axis 2 adds a consistent **bowl-shaped** excursion: mean axis-2 coordinate ≈ 0 near zero
  offset, ≈ −3 at mid offsets, back to ≈ 0 at the far offsets.
- Fitting positive and negative offsets separately gives directions with median cosine 0.94
  (q10 0.60), so regressing on `|h|` is a reasonable default; sign-asymmetric patches are a
  minority.

## Possible next steps

- Separate positive/negative offset bases for the asymmetric patches.
- Per-offset normalisation (instead of a global std) to isolate waveform/shape changes from
  amplitude-vs-offset.
- Visualise `offset_coords` trajectories, `offset_slope` and `offset_axis_var` as maps to find
  where offset behaviour is strong, linear or curved.
