# Seismic Pre-Stack Backend

FastAPI backend, zero-copy `.npz` latent similarity engine, and ML job orchestrator for 4D pre-stack `.sgz` surveys.

## Setup

```bash
cd backend
uv sync
```

## Generate Signal-Derived Demo Embeddings

```bash
cd backend
uv run python scripts/generate_demo_embeddings.py --input ../data/ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time.sgz
```

## Build the Volve Ground-Truth Reservoir Horizons

Downloads open Volve data (Top Hugin TWT horizon, well picks with TVDSS+TWT, the Eclipse
corner-point reservoir grid + ACTNUM) into `data/volve_reservoir/`, maps everything onto the
survey IL/XL grid via the CDP_X/CDP_Y trace headers, converts the reservoir model from depth to
PP time (top tied to the interpreted horizon, base from the well-derived Hugin interval
velocity) and writes `data/<survey>_horizons.npz`. `pyarrow` is only needed for this step:

```bash
cd backend
uv run --with pyarrow python scripts/build_reservoir_horizons.py --survey ../data/ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time.sgz
```

The backend serves the result at `GET /api/surveys/{id}/horizons` (float32 TWT grids, base64)
and the frontend exposes it as the **Ground Truth** / **Reservoir** toggle in the sidebar and
every viewer. The data is under the Equinor Volve data licence.

## Run the API Server

```bash
cd backend
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Environment variables (optional):
- `SEISMIC_DATA_DIR`: override `<repo_root>/data`
- `SEISMIC_EMBEDDINGS_DIR`: override `<repo_root>/embeddings`
- `SEISMIC_WEIGHTS_DIR`: override `<repo_root>/weights`

## Run Tests

```bash
cd backend
uv run pytest
```
