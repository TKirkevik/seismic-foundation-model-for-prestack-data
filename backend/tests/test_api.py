"""Focused integration tests for the FastAPI seismic backend against real Volve SGZ + demo .npz."""

from __future__ import annotations

import base64
import json
import threading
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format as npformat
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.latent_store import LatentStore
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


def _write_test_patch_npz(path: Path, offsets: list[float] | None = None) -> None:
    offs = np.asarray(offsets if offsets is not None else [0.0, 100.0], dtype=np.float64)
    emb = np.arange(len(offs) * 2 * 2 * 2 * 4, dtype=np.float16).reshape(len(offs), 2, 2, 2, 4)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as zf:
        with zf.open("embeddings.npy", "w") as fh:
            npformat.write_array(fh, emb, allow_pickle=False)
        for name, val in {
            "offsets": offs,
            "il_patch_start": np.array([0, 16], dtype=np.int64),
            "xl_patch_start": np.array([0, 16], dtype=np.int64),
            "t_patch_start": np.array([0, 16], dtype=np.int64),
            "patch_size": np.array(16, dtype=np.int64),
        }.items():
            with zf.open(f"{name}.npy", "w") as fh:
                npformat.write_array(fh, val, allow_pickle=False)


def test_release_set_removes_until_rescan_and_concurrent_safety(tmp_path: Path):
    npz_path = tmp_path / "foo_demo_patch_embeddings.npz"
    _write_test_patch_npz(npz_path)

    store = LatentStore(tmp_path)
    store.scan(["foo"])

    handle = store.get_set("foo", "demo")
    vol_before = handle.read_offset_volume(0)
    assert vol_before.shape == (2, 2, 2, 4)
    assert vol_before.dtype == np.float32
    assert not isinstance(vol_before, np.memmap)

    errors: list[Exception] = []
    stop_reading = threading.Event()

    def _reader() -> None:
        while not stop_reading.is_set():
            try:
                arr = handle.read_offset_volume(0)
                assert np.isfinite(arr).all()
            except RuntimeError as exc:
                if "closed" not in str(exc):
                    errors.append(exc)
                break
            except Exception as exc:
                errors.append(exc)
                break

    t = threading.Thread(target=_reader, daemon=True)
    t.start()
    store.release_set("foo", "demo")
    stop_reading.set()
    t.join(timeout=2)
    assert not errors

    # Snapshot read before close remains valid; handle rejects new reads; store hides set until rescan.
    assert np.isfinite(vol_before).all()
    with pytest.raises(KeyError):
        store.get_set("foo", "demo")
    with pytest.raises(RuntimeError, match="closed"):
        handle.read_offset_volume(0)
    with pytest.raises(RuntimeError, match="closed"):
        handle.read_zero_offset_volume(np.array([0]))

    store.scan(["foo"])
    reloaded = store.get_set("foo", "demo")
    assert reloaded is not handle
    assert reloaded.read_offset_volume(0).shape == (2, 2, 2, 4)
    store.close()


def test_longest_prefix_npz_pairing(tmp_path: Path):
    _write_test_patch_npz(tmp_path / "foo_demo_patch_embeddings.npz")
    _write_test_patch_npz(tmp_path / "foo_sub_demo_patch_embeddings.npz")
    _write_test_patch_npz(tmp_path / "foo_sub_patch_embeddings.npz")

    store = LatentStore(tmp_path)
    store.scan(["foo", "foo_sub"])

    foo_ids = {h.id for h in store.list_for_survey("foo")}
    foo_sub_ids = {h.id for h in store.list_for_survey("foo_sub")}
    assert foo_ids == {"demo"}
    assert foo_sub_ids == {"demo", "default"}
    store.close()


def test_duplicate_active_job_returns_409(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    import app.jobs as jobs_mod

    started = threading.Event()
    release = threading.Event()

    def _blocking_demo(*args, **kwargs):
        started.set()
        release.wait(timeout=5)
        raise RuntimeError("aborted test stub")

    monkeypatch.setattr(jobs_mod, "generate_demo_embeddings", _blocking_demo)

    resp1 = client.post(
        "/api/jobs/demo-embeddings",
        json={"survey_id": SURVEY_ID, "tag": "conflict_tag", "offset_stride": 5},
    )
    assert resp1.status_code == 200
    job_id = resp1.json()["id"]
    assert started.wait(timeout=2)

    resp2 = client.post(
        "/api/jobs/demo-embeddings",
        json={"survey_id": SURVEY_ID, "tag": "conflict_tag", "offset_stride": 5},
    )
    assert resp2.status_code == 409

    cancel_resp = client.post(f"/api/jobs/{job_id}/cancel")
    assert cancel_resp.status_code == 200
    assert cancel_resp.json()["status"] == "cancelled"
    release.set()


def test_config_resolves_relative_paths(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    monkeypatch.chdir(tmp_path)
    cfg = Settings(
        repo_root=Path("."),
        data_dir=Path("rel_data"),
        embeddings_dir=Path("rel_emb"),
        weights_dir=Path("rel_weights"),
        embedders_dir=Path("rel_embedders"),
    )
    assert cfg.repo_root == tmp_path.resolve()
    assert cfg.data_dir == (tmp_path / "rel_data").resolve()
    assert cfg.embeddings_dir == (tmp_path / "rel_emb").resolve()
    assert cfg.weights_dir == (tmp_path / "rel_weights").resolve()
    assert cfg.embedders_dir == (tmp_path / "rel_embedders").resolve()


def test_rescan_reuses_unchanged_survey_handles(client: TestClient):
    backend = app.state.backend
    before = backend.seismic_store.get_survey(SURVEY_ID)
    rescan_resp = client.post("/api/surveys/rescan")
    assert rescan_resp.status_code == 200
    after = backend.seismic_store.get_survey(SURVEY_ID)
    assert after is before
    assert after._closed is False


def test_has_distinct_abs_offsets_and_cube_stats_progress(tmp_path: Path):
    from app.jobs import _has_distinct_abs_offsets

    single_off = tmp_path / "single.npz"
    sym_off = tmp_path / "sym.npz"
    multi_off = tmp_path / "multi.npz"
    _write_test_patch_npz(single_off, offsets=[50.0])
    _write_test_patch_npz(sym_off, offsets=[-100.0, 100.0])
    _write_test_patch_npz(multi_off, offsets=[0.0, 100.0])

    assert _has_distinct_abs_offsets(single_off) is False
    assert _has_distinct_abs_offsets(sym_off) is False
    assert _has_distinct_abs_offsets(multi_off) is True


def test_multipick_points_polyline_and_concave_polygon(client: TestClient):
    # 1. Points on xline plane
    resp_pts = client.post(
        f"/api/surveys/{SURVEY_ID}/similarity",
        json={
            "embedding_set_id": "demo",
            "offset_idx": 50,
            "mode": "active_offset",
            "features": [
                {
                    "id": "pts-1",
                    "role": "include",
                    "plane_axis": "xline",
                    "plane_index": 150,
                    "geometry_mode": "points",
                    "picks": [
                        {"il_idx": 32, "xl_idx": 150, "sample_idx": 160},
                        {"il_idx": 128, "xl_idx": 150, "sample_idx": 480},
                    ],
                }
            ],
        },
    )
    assert resp_pts.status_code == 200
    d_pts = resp_pts.json()
    assert d_pts["included_patch_count"] == 2
    assert d_pts["excluded_patch_count"] == 0
    assert d_pts["overlap_resolved_count"] == 0
    sim_pts = np.frombuffer(base64.b64decode(d_pts["similarity_b64"]), dtype="<f4").reshape(13, 19, 71)
    assert sim_pts[2, 9, 10] > 0.99
    assert sim_pts[8, 9, 30] > 0.99
    match_patches = {tuple(m["patch_idx"]) for m in d_pts["top_matches"]}
    assert (2, 9, 10) not in match_patches
    assert (8, 9, 30) not in match_patches

    # 2. Polyline on inline plane: interpolated intermediate patch [6, 7, 20] must be included & masked
    resp_line = client.post(
        f"/api/surveys/{SURVEY_ID}/similarity",
        json={
            "embedding_set_id": "demo",
            "offset_idx": 50,
            "mode": "active_offset",
            "features": [
                {
                    "id": "line-1",
                    "role": "include",
                    "plane_axis": "inline",
                    "plane_index": 96,
                    "geometry_mode": "polyline",
                    "picks": [
                        {"il_idx": 96, "xl_idx": 80, "sample_idx": 320},
                        {"il_idx": 96, "xl_idx": 144, "sample_idx": 320},
                    ],
                }
            ],
        },
    )
    assert resp_line.status_code == 200
    d_line = resp_line.json()
    assert d_line["included_patch_count"] == 5  # xl patches 5, 6, 7, 8, 9
    sim_line = np.frombuffer(base64.b64decode(d_line["similarity_b64"]), dtype="<f4").reshape(13, 19, 71)
    assert sim_line[6, 7, 20] > 0.99
    line_matches = {tuple(m["patch_idx"]) for m in d_line["top_matches"]}
    assert (6, 7, 20) not in line_matches

    # 3. Concave U-shaped polygon with horizontal top and bottom edges
    resp_poly = client.post(
        f"/api/surveys/{SURVEY_ID}/similarity",
        json={
            "embedding_set_id": "demo",
            "offset_idx": 50,
            "mode": "active_offset",
            "features": [
                {
                    "id": "poly-1",
                    "role": "include",
                    "plane_axis": "inline",
                    "plane_index": 96,
                    "geometry_mode": "polygon",
                    "picks": [
                        {"il_idx": 96, "xl_idx": 64, "sample_idx": 320},
                        {"il_idx": 96, "xl_idx": 160, "sample_idx": 320},
                        {"il_idx": 96, "xl_idx": 160, "sample_idx": 480},
                        {"il_idx": 96, "xl_idx": 128, "sample_idx": 480},
                        {"il_idx": 96, "xl_idx": 128, "sample_idx": 384},
                        {"il_idx": 96, "xl_idx": 96, "sample_idx": 384},
                        {"il_idx": 96, "xl_idx": 96, "sample_idx": 480},
                        {"il_idx": 96, "xl_idx": 64, "sample_idx": 480},
                    ],
                }
            ],
        },
    )
    assert resp_poly.status_code == 200
    d_poly = resp_poly.json()
    sim_poly = np.frombuffer(base64.b64decode(d_poly["similarity_b64"]), dtype="<f4").reshape(13, 19, 71)
    # Interior of left leg: xl=80 (patch 5), sample=416 (patch 26)
    assert sim_poly[6, 5, 26] > 0.99
    # Top horizontal edge: xl=112 (patch 7), sample=320 (patch 20)
    assert sim_poly[6, 7, 20] > 0.99


def test_inclusion_exclusion_and_overlap_resolution(client: TestClient):
    # Shared-patch collision resolves in favor of exclude when len(include) > 1
    resp = client.post(
        f"/api/surveys/{SURVEY_ID}/similarity",
        json={
            "embedding_set_id": "demo",
            "offset_idx": 50,
            "mode": "active_offset",
            "exclusion_weight": 0.75,
            "features": [
                {
                    "id": "inc",
                    "role": "include",
                    "plane_axis": "inline",
                    "plane_index": 96,
                    "geometry_mode": "points",
                    "picks": [
                        {"il_idx": 96, "xl_idx": 80, "sample_idx": 320},
                        {"il_idx": 96, "xl_idx": 144, "sample_idx": 320},
                    ],
                },
                {
                    "id": "exc",
                    "role": "exclude",
                    "plane_axis": "inline",
                    "plane_index": 96,
                    "geometry_mode": "single",
                    "picks": [{"il_idx": 96, "xl_idx": 144, "sample_idx": 320}],
                },
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["overlap_resolved_count"] == 1
    assert data["included_patch_count"] == 1
    assert data["excluded_patch_count"] == 1
    assert data["exclusion_weight"] == 0.75

    sim = np.frombuffer(base64.b64decode(data["similarity_b64"]), dtype="<f4").reshape(13, 19, 71)
    # Excluded patch [6, 9, 20] is suppressed: sim_pos - 0.75 * 1.0 <= 0.25
    assert sim[6, 9, 20] <= 0.2501
    # Both picked patches are masked from top_matches
    match_patches = {tuple(m["patch_idx"]) for m in data["top_matches"]}
    assert (6, 5, 20) not in match_patches
    assert (6, 9, 20) not in match_patches

    # Self-cancellation guard when include == exclude
    resp_same = client.post(
        f"/api/surveys/{SURVEY_ID}/similarity",
        json={
            "embedding_set_id": "demo",
            "offset_idx": 50,
            "mode": "active_offset",
            "exclusion_weight": 1.0,
            "features": [
                {
                    "id": "inc",
                    "role": "include",
                    "plane_axis": "inline",
                    "plane_index": 96,
                    "geometry_mode": "single",
                    "picks": [{"il_idx": 96, "xl_idx": 80, "sample_idx": 320}],
                },
                {
                    "id": "exc",
                    "role": "exclude",
                    "plane_axis": "inline",
                    "plane_index": 96,
                    "geometry_mode": "single",
                    "picks": [{"il_idx": 96, "xl_idx": 80, "sample_idx": 320}],
                },
            ],
        },
    )
    assert resp_same.status_code == 200
    d_same = resp_same.json()
    assert d_same["overlap_resolved_count"] == 1
    assert d_same["included_patch_count"] == 1
    assert d_same["excluded_patch_count"] == 0
    sim_same = np.frombuffer(base64.b64decode(d_same["similarity_b64"]), dtype="<f4").reshape(13, 19, 71)
    assert sim_same[6, 5, 20] > 0.99


def test_multipick_validation_errors(client: TestClient):
    # Only exclude features (0 include picks) -> 400
    resp_no_inc = client.post(
        f"/api/surveys/{SURVEY_ID}/similarity",
        json={
            "embedding_set_id": "demo",
            "features": [
                {
                    "id": "exc-only",
                    "role": "exclude",
                    "plane_axis": "inline",
                    "plane_index": 96,
                    "geometry_mode": "single",
                    "picks": [{"il_idx": 96, "xl_idx": 80, "sample_idx": 320}],
                }
            ],
        },
    )
    assert resp_no_inc.status_code == 400

    # Non-coplanar pick -> 400
    resp_non_coplanar = client.post(
        f"/api/surveys/{SURVEY_ID}/similarity",
        json={
            "embedding_set_id": "demo",
            "features": [
                {
                    "id": "bad-plane",
                    "role": "include",
                    "plane_axis": "inline",
                    "plane_index": 96,
                    "geometry_mode": "single",
                    "picks": [{"il_idx": 97, "xl_idx": 80, "sample_idx": 320}],
                }
            ],
        },
    )
    assert resp_non_coplanar.status_code == 400

    # Out-of-bounds coordinate -> 400
    resp_oob = client.post(
        f"/api/surveys/{SURVEY_ID}/similarity",
        json={
            "embedding_set_id": "demo",
            "features": [
                {
                    "id": "oob",
                    "role": "include",
                    "plane_axis": "inline",
                    "plane_index": 999,
                    "geometry_mode": "single",
                    "picks": [{"il_idx": 999, "xl_idx": 80, "sample_idx": 320}],
                }
            ],
        },
    )
    assert resp_oob.status_code == 400


def test_combined_trajectory_multipick_and_trajectory_endpoint(client: TestClient):
    resp = client.post(
        f"/api/surveys/{SURVEY_ID}/similarity",
        json={
            "embedding_set_id": "demo",
            "offset_idx": 50,
            "mode": "combined_trajectory",
            "whiten": True,
            "focus_pick": {"il_idx": 96, "xl_idx": 144, "sample_idx": 320},
            "features": [
                {
                    "id": "inc-comb",
                    "role": "include",
                    "plane_axis": "inline",
                    "plane_index": 96,
                    "geometry_mode": "points",
                    "picks": [
                        {"il_idx": 96, "xl_idx": 80, "sample_idx": 320},
                        {"il_idx": 96, "xl_idx": 144, "sample_idx": 320},
                    ],
                }
            ],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["mode"] == "combined_trajectory"
    assert data["pick"]["patch_idx"] == [6, 9, 20]
    assert data["trajectory"]["available"] is True
    sim = np.frombuffer(base64.b64decode(data["similarity_b64"]), dtype="<f4").reshape(13, 19, 71)
    assert sim[6, 5, 20] > 0.99
    assert sim[6, 9, 20] > 0.99

    traj_resp = client.get(
        f"/api/surveys/{SURVEY_ID}/trajectory",
        params={
            "embedding_set_id": "demo",
            "il_idx": 96,
            "xl_idx": 144,
            "sample_idx": 320,
            "offset_idx": 50,
        },
    )
    assert traj_resp.status_code == 200
    t_data = traj_resp.json()
    assert t_data["pick"] == data["pick"]
    assert t_data["trajectory"] == data["trajectory"]

    traj_404 = client.get(
        f"/api/surveys/nonexistent/trajectory",
        params={"embedding_set_id": "demo", "il_idx": 0, "xl_idx": 0, "sample_idx": 0},
    )
    assert traj_404.status_code == 404

    traj_400 = client.get(
        f"/api/surveys/{SURVEY_ID}/trajectory",
        params={"embedding_set_id": "demo", "il_idx": 999, "xl_idx": 0, "sample_idx": 0},
    )
    assert traj_400.status_code == 400
