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
