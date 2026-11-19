"""Focused integration tests for the FastAPI seismic backend against real Volve SGZ + demo .npz."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.main import app

SURVEY_ID = "ST0202R08_PZ_PrSDM_CIP_gathers_in_PP_Time"


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as tc:
        yield tc


def test_health(client: TestClient):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["device"] in ("cuda", "mps", "cpu")
    assert data["mps_fallback"] is True
    assert data["weights_available"] is True


def test_surveys_and_rescan(client: TestClient):
    resp = client.get("/api/surveys")
    assert resp.status_code == 200
    surveys = resp.json()
    match = [s for s in surveys if s["id"] == SURVEY_ID]
    assert len(match) == 1
    s = match[0]
    assert s["is_4d"] is True
    assert s["shape"] == [193, 303, 1126, 100]
    assert s["default_offset_index"] == 50
    assert s["amplitude_clip_default"] > 0

    emb_map = {e["id"]: e for e in s["embedding_sets"]}
    assert "demo" in emb_map
    demo = emb_map["demo"]
    assert demo["has_patch_embeddings"] is True
    assert demo["has_offset_summary"] is True
    assert demo["patch_grid_shape"] == [13, 19, 71]
    assert demo["patch_size"] == 16
    assert len(demo["embedded_offsets"]) == 20
    assert set(demo["available_modes"]) == {
        "active_offset",
        "zero_offset",
        "offset_direction",
        "offset_curvature",
        "combined_trajectory",
    }
    assert set(demo["available_attributes"]) == {
        "offset_slope",
        "offset_fit_r2",
        "offset_axis_var_1",
        "offset_axis_var_2",
        "offset_axis_var_plane",
    }

    rescan_resp = client.post("/api/surveys/rescan")
    assert rescan_resp.status_code == 200
    assert any(item["id"] == SURVEY_ID for item in rescan_resp.json())


@pytest.mark.parametrize(
    "axis,index,expected_shape",
    [
        ("inline", 96, (303, 1126)),
        ("xline", 150, (193, 1126)),
        ("sample", 560, (193, 303)),
    ],
)
def test_section_all_axes(client: TestClient, axis: str, index: int, expected_shape: tuple[int, int]):
    resp = client.get(
        f"/api/surveys/{SURVEY_ID}/section",
        params={"axis": axis, "index": index, "offset_index": 50},
    )
    assert resp.status_code == 200
    assert resp.headers["X-Shape"] == f"{expected_shape[0]},{expected_shape[1]}"
    assert resp.headers["X-Axis"] == axis
    assert float(resp.headers["X-Offset-Value"]) == 50.0
    arr = np.frombuffer(resp.content, dtype="<f4").reshape(expected_shape)
    assert arr.shape == expected_shape
    assert np.isfinite(arr).all()


@pytest.mark.parametrize("line_axis", ["xline", "inline"])
def test_gather_panel_both_axes(client: TestClient, line_axis: str):
    resp = client.get(
        f"/api/surveys/{SURVEY_ID}/gather-panel",
        params={
            "line_axis": line_axis,
            "il_start_idx": 90,
            "xl_start_idx": 140,
            "increment": 5,
            "count": 4,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["X-Shape"] == "4,100,1126"
    locs = json.loads(resp.headers["X-Locations"])
    assert len(locs) == 4
    if line_axis == "xline":
        assert [l["xl_idx"] for l in locs] == [140, 145, 150, 155]
        assert all(l["il_idx"] == 90 for l in locs)
    else:
        assert [l["il_idx"] for l in locs] == [90, 95, 100, 105]
        assert all(l["xl_idx"] == 140 for l in locs)
    arr = np.frombuffer(resp.content, dtype="<f4").reshape(4, 100, 1126)
    assert np.isfinite(arr).all()


def test_gather_panel_truncates_at_boundary(client: TestClient):
    resp = client.get(
        f"/api/surveys/{SURVEY_ID}/gather-panel",
        params={
            "line_axis": "inline",
            "il_start_idx": 185,
            "xl_start_idx": 150,
            "increment": 5,
            "count": 10,
        },
    )
    assert resp.status_code == 200
    assert resp.headers["X-Shape"] == "2,100,1126"
    locs = json.loads(resp.headers["X-Locations"])
    assert [l["il_idx"] for l in locs] == [185, 190]


@pytest.mark.parametrize(
    "mode,whiten",
    [
        ("active_offset", False),
        ("active_offset", True),
        ("zero_offset", False),
        ("offset_direction", False),
        ("offset_curvature", True),
        ("combined_trajectory", False),
    ],
)
def test_similarity_all_modes_and_whiten(client: TestClient, mode: str, whiten: bool):
    resp = client.post(
        f"/api/surveys/{SURVEY_ID}/similarity",
        json={
            "embedding_set_id": "demo",
            "il_idx": 96,
            "xl_idx": 150,
            "sample_idx": 560,
            "offset_idx": 50,
            "mode": mode,
            "whiten": whiten,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == mode
    assert data["whiten"] is whiten
    assert data["grid_shape"] == [13, 19, 71]
    assert data["pick"]["patch_idx"] == [6, 9, 35]

    sim = np.frombuffer(base64.b64decode(data["similarity_b64"]), dtype="<f4").reshape(13, 19, 71)
    assert np.isfinite(sim).all()
    assert np.isclose(sim.min(), data["similarity_min"], atol=1e-5)
    assert np.isclose(sim.max(), data["similarity_max"], atol=1e-5)
    assert sim[6, 9, 35] > 0.99

    matches = data["top_matches"]
    assert len(matches) == 12
    assert all(m["patch_idx"] != [6, 9, 35] for m in matches)
    scores = [m["score"] for m in matches]
    assert scores == sorted(scores, reverse=True)

    traj = data["trajectory"]
    assert traj["available"] is True
    assert len(traj["offsets"]) == 20
    assert len(traj["c1"]) == 20
    assert len(traj["c2"]) == 20
    assert traj["slope"] is not None
    assert traj["r2"] is not None
    assert len(traj["axis_var"]) == 2


@pytest.mark.parametrize(
    "attribute",
    [
        "offset_slope",
        "offset_fit_r2",
        "offset_axis_var_1",
        "offset_axis_var_2",
        "offset_axis_var_plane",
    ],
)
def test_attribute_endpoint(client: TestClient, attribute: str):
    resp = client.get(
        f"/api/surveys/{SURVEY_ID}/attribute",
        params={"embedding_set_id": "demo", "attribute": attribute},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["attribute"] == attribute
    assert data["grid_shape"] == [13, 19, 71]
    vals = np.frombuffer(base64.b64decode(data["values_b64"]), dtype="<f4").reshape(13, 19, 71)
    assert np.isfinite(vals).all()
    assert np.isclose(vals.min(), data["value_min"], atol=1e-5)
    assert np.isclose(vals.max(), data["value_max"], atol=1e-5)


def test_horizons_endpoint(client: TestClient):
    surveys = client.get("/api/surveys").json()
    survey = next(s for s in surveys if s["id"] == SURVEY_ID)
    assert set(survey["horizon_ids"]) >= {"top_hugin_twt", "reservoir_top", "reservoir_base"}

    resp = client.get(f"/api/surveys/{SURVEY_ID}/horizons")
    assert resp.status_code == 200
    data = resp.json()
    by_id = {h["id"]: h for h in data["horizons"]}
    top = by_id["reservoir_top"]
    base = by_id["reservoir_base"]
    assert top["kind"] == "top" and base["kind"] == "base"
    assert top["grid_shape"] == [193, 303]

    top_twt = np.frombuffer(base64.b64decode(top["twt_b64"]), dtype="<f4").reshape(193, 303)
    base_twt = np.frombuffer(base64.b64decode(base["twt_b64"]), dtype="<f4").reshape(193, 303)
    defined = np.isfinite(top_twt)
    assert defined.any() and np.array_equal(defined, np.isfinite(base_twt))
    assert (base_twt[defined] >= top_twt[defined]).all()
    assert 2000.0 < top_twt[defined].min() < top_twt[defined].max() < 3500.0
    assert np.isclose(top_twt[defined].min(), top["twt_min"], atol=1e-3)

    picks = data["well_picks"]
    assert picks and all(0 <= p["il_idx"] < 193 and 0 <= p["xl_idx"] < 303 for p in picks)

    missing = client.get("/api/surveys/does-not-exist/horizons")
    assert missing.status_code == 404


def test_jobs_and_preprocess_stub(client: TestClient):
    resp = client.get("/api/jobs")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)

    prep = client.post(
        "/api/jobs/preprocess",
        json={"survey_id": SURVEY_ID, "params": {}},
    )
    assert prep.status_code == 200
    assert prep.json()["status"] == "unsupported"
