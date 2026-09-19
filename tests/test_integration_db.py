"""Integration tests against a real PostGIS database.

Everything here exercises SQL that a mock cannot verify: PostGIS geometry
round-trips, the ON CONFLICT upsert semantics, spatial predicates, and the
end-to-end detect -> fuse -> query path.

Skipped unless a database is reachable.  Point ``TEST_DATABASE_URL`` at one, or
run the compose stack:

    TEST_DATABASE_URL=postgresql+psycopg://sentinel:sentinel@localhost:5432/bobsentinel \
        pytest tests/test_integration_db.py
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta

import pytest
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.orm import sessionmaker

from bob_sentinel.fusion.interpolate import Fix, interpolate_fleet
from bob_sentinel.fusion.matcher import DetectionInput, match_detections, summarise
from bob_sentinel.ingest.aisstream import AISRecord
from bob_sentinel.ingest.eez import BANGLADESH_EEZ_APPROX, geojson_to_wkt
from bob_sentinel.models import AISPosition, Anomaly, Base, Detection, SARScene, Vessel
from bob_sentinel.repository import (
    insert_anomalies,
    insert_detections,
    upsert_ais_records,
    upsert_region,
    upsert_scene,
)

pytestmark = pytest.mark.integration

DATABASE_URL = os.environ.get("TEST_DATABASE_URL")

# Anchored to "now" rather than a fixed date: the API filters on rolling
# windows (30 days for detections, 60 minutes for AIS), so a hard-coded date
# would silently fall out of every window as the calendar advanced.
NOW = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=1)


@pytest.fixture(scope="module")
def engine():
    if not DATABASE_URL:
        pytest.skip("set TEST_DATABASE_URL to run integration tests")
    engine = create_engine(DATABASE_URL, future=True)
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT PostGIS_Version()"))
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"database unreachable: {exc}")
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)

    # /healthz queries the module-level engine directly, so point it at the
    # test database for the duration of the module.
    import bob_sentinel.db as db_module

    original = db_module._engine
    db_module._engine = engine
    yield engine
    db_module._engine = original
    Base.metadata.drop_all(engine)
    engine.dispose()


@pytest.fixture
def session(engine):
    factory = sessionmaker(bind=engine, expire_on_commit=False, future=True)
    session = factory()
    yield session
    session.rollback()
    # Leave a clean slate for the next test.
    for table in reversed(Base.metadata.sorted_tables):
        session.execute(text(f"TRUNCATE {table.name} RESTART IDENTITY CASCADE"))
    session.commit()
    session.close()


def ais(mmsi: int, ts: datetime, lon: float, lat: float, **kwargs) -> AISRecord:
    return AISRecord(mmsi=mmsi, ts=ts, lon=lon, lat=lat, **kwargs)


# --- schema ---------------------------------------------------------------


def test_schema_creates_every_table(engine) -> None:
    with engine.connect() as connection:
        tables = {
            row[0]
            for row in connection.execute(
                text("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            )
        }
    assert {"vessels", "ais_positions", "sar_scenes", "detections", "anomalies",
            "regions"} <= tables


def test_geometry_columns_are_registered_with_srid_4326(engine) -> None:
    with engine.connect() as connection:
        rows = dict(
            connection.execute(
                text("SELECT f_table_name || '.' || f_geometry_column, srid "
                     "FROM geometry_columns")
            ).all()
        )
    assert rows["ais_positions.geom"] == 4326
    assert rows["detections.geom"] == 4326
    assert rows["regions.geom"] == 4326


def test_spatial_indexes_exist(engine) -> None:
    """Without these, every ST_DWithin becomes a sequential scan."""
    with engine.connect() as connection:
        indexes = {
            row[0]
            for row in connection.execute(
                text("SELECT indexdef FROM pg_indexes WHERE schemaname='public'")
            )
        }
    assert any("gist" in i.lower() and "ais_positions" in i for i in indexes)
    assert any("gist" in i.lower() and "detections" in i for i in indexes)


# --- AIS persistence -------------------------------------------------------


def test_ais_records_round_trip_through_postgis(session) -> None:
    written, vessels = upsert_ais_records(
        session, [ais(405000123, NOW, 90.2, 21.5, sog_kn=8.4, cog_deg=142.0)]
    )
    session.commit()
    assert (written, vessels) == (1, 1)

    row = session.execute(
        text("SELECT ST_X(geom::geometry) AS lon, ST_Y(geom::geometry) AS lat "
             "FROM ais_positions")
    ).mappings().one()
    assert (row["lon"], row["lat"]) == (90.2, 21.5)


def test_replayed_positions_do_not_duplicate(session) -> None:
    """A reconnecting ingester replays frames; that must be a no-op."""
    records = [ais(405000123, NOW, 90.2, 21.5)]
    upsert_ais_records(session, records)
    session.commit()
    written, _ = upsert_ais_records(session, records)
    session.commit()
    assert written == 0
    assert session.scalar(select(func.count()).select_from(AISPosition)) == 1


def test_duplicate_timestamps_within_one_batch_are_collapsed(session) -> None:
    """ON CONFLICT cannot resolve two conflicting rows in one statement."""
    upsert_ais_records(
        session, [ais(1, NOW, 90.0, 21.0), ais(1, NOW, 90.0, 21.0)]
    )
    session.commit()
    assert session.scalar(select(func.count()).select_from(AISPosition)) == 1


def test_static_data_enriches_a_vessel_without_erasing_it(session) -> None:
    """COALESCE semantics: a later position report must not blank the name."""
    upsert_ais_records(
        session,
        [AISRecord(mmsi=1, ts=NOW, name="FV EXAMPLE", imo=9074729, kind="static")],
    )
    session.commit()
    upsert_ais_records(session, [ais(1, NOW + timedelta(minutes=1), 90.0, 21.0)])
    session.commit()

    vessel = session.get(Vessel, 1)
    assert vessel.name == "FV EXAMPLE"
    assert vessel.imo == 9074729
    assert vessel.last_seen == NOW + timedelta(minutes=1)


def test_first_and_last_seen_track_the_extremes(session) -> None:
    upsert_ais_records(
        session,
        [
            ais(1, NOW, 90.0, 21.0),
            ais(1, NOW - timedelta(hours=2), 89.9, 21.0),
            ais(1, NOW + timedelta(hours=2), 90.1, 21.0),
        ],
    )
    session.commit()
    vessel = session.get(Vessel, 1)
    assert vessel.first_seen == NOW - timedelta(hours=2)
    assert vessel.last_seen == NOW + timedelta(hours=2)


def test_static_only_records_create_a_vessel_but_no_position(session) -> None:
    written, vessels = upsert_ais_records(
        session, [AISRecord(mmsi=7, ts=NOW, name="NO FIX", kind="static")]
    )
    session.commit()
    assert (written, vessels) == (0, 1)
    assert session.get(Vessel, 7) is not None


# --- regions & spatial predicates -----------------------------------------


def test_eez_polygon_loads_and_answers_containment(session) -> None:
    region = upsert_region(
        session, name="Bangladesh EEZ (approximate)", kind="eez", mrgid=8481,
        geom_wkt=geojson_to_wkt(BANGLADESH_EEZ_APPROX),
    )
    session.commit()
    assert region.id is not None

    inside, outside = session.execute(
        text(
            """
            SELECT
              ST_Contains(geom::geometry, ST_SetSRID(ST_Point(90.5, 21.3), 4326)),
              ST_Contains(geom::geometry, ST_SetSRID(ST_Point(80.0, 15.0), 4326))
            FROM regions WHERE mrgid = 8481
            """
        )
    ).one()
    assert inside is True
    assert outside is False


def test_region_upsert_replaces_rather_than_duplicates(session) -> None:
    for _ in range(2):
        upsert_region(session, name="Bangladesh EEZ (approximate)", kind="eez",
                      mrgid=8481, geom_wkt=geojson_to_wkt(BANGLADESH_EEZ_APPROX))
        session.commit()
    assert session.scalar(text("SELECT count(*) FROM regions")) == 1


def test_st_dwithin_needs_a_geography_cast_to_mean_metres(session) -> None:
    """The geography cast is what makes a metre threshold mean metres.

    The geom columns are ``geometry`` in EPSG:4326, so a bare ST_DWithin
    compares *degrees* — a 400 "metre" gate would silently match anything
    within 400 degrees, i.e. the entire planet.  Any distance predicate added
    to this codebase must cast to geography first.
    """
    upsert_ais_records(
        session,
        [ais(1, NOW, 90.0, 21.0), ais(2, NOW, 90.005, 21.0)],  # ~520 m apart
    )
    session.commit()
    distance_m, near, far, degrees_trap = session.execute(
        text(
            """
            SELECT
              ST_Distance(a.geom::geography, b.geom::geography),
              ST_DWithin(a.geom::geography, b.geom::geography, 600),
              ST_DWithin(a.geom::geography, b.geom::geography, 400),
              ST_DWithin(a.geom::geometry, b.geom::geometry, 400)
            FROM ais_positions a, ais_positions b
            WHERE a.mmsi = 1 AND b.mmsi = 2
            """
        )
    ).one()
    assert distance_m == pytest.approx(520, abs=5)
    assert near is True
    assert far is False
    assert degrees_trap is True, "uncast ST_DWithin compares degrees, not metres"


# --- scenes & detections ---------------------------------------------------


def make_scene(session, **overrides) -> SARScene:
    fields = {
        "product_id": "abc-123",
        "product_name": "S1D_IW_GRDH_1SDV_20260103T114217_0001",
        "platform": "S1D",
        "product_type": "IW_GRDH_1S",
        "acquired_start": NOW,
        "acquired_end": NOW + timedelta(seconds=28),
        "footprint_wkt": "POLYGON((88 20,92 20,92 23,88 23,88 20))",
        "status": "catalogued",
    }
    fields.update(overrides)
    scene = upsert_scene(session, **fields)
    session.commit()
    return scene


def test_scene_upsert_is_idempotent_on_product_id(session) -> None:
    make_scene(session)
    scene = make_scene(session, status="complete")
    session.commit()
    assert session.scalar(select(func.count()).select_from(SARScene)) == 1
    assert scene.status == "complete"


def test_acquisition_midpoint_is_the_fusion_timestamp(session) -> None:
    scene = make_scene(session)
    assert scene.acquisition_midpoint == NOW + timedelta(seconds=14)


def test_detections_insert_and_reject_pixel_duplicates(session) -> None:
    scene = make_scene(session)
    rows = [
        {"lon": 90.5, "lat": 21.3, "row": 100, "col": 120, "snr_db": 14.0,
         "area_px": 6, "length_m": 30.0, "confidence": 0.5, "peak_db": 20.0},
    ]
    assert insert_detections(session, scene, rows) == 1
    session.commit()
    assert insert_detections(session, scene, rows) == 0, "same pixel inserted twice"
    session.commit()
    assert session.scalar(select(func.count()).select_from(Detection)) == 1


def test_deleting_a_scene_cascades_to_its_detections(session) -> None:
    scene = make_scene(session)
    insert_detections(session, scene, [
        {"lon": 90.5, "lat": 21.3, "row": 1, "col": 1, "snr_db": 10.0}
    ])
    session.commit()
    session.delete(session.get(SARScene, scene.id))
    session.commit()
    assert session.scalar(select(func.count()).select_from(Detection)) == 0


# --- anomalies -------------------------------------------------------------


def test_anomalies_deduplicate_on_their_natural_key(session) -> None:
    """A re-run of the scan must not multiply the same finding."""
    row = {
        "kind": "ais_gap", "mmsi": 1, "counterpart_mmsi": None, "start_ts": NOW,
        "end_ts": NOW + timedelta(hours=8), "lon": 90.0, "lat": 21.0,
        "score": 0.3, "details": {"duration_hours": 8.0},
    }
    assert insert_anomalies(session, [row]) == 1
    session.commit()
    assert insert_anomalies(session, [row]) == 0
    session.commit()
    assert session.scalar(select(func.count()).select_from(Anomaly)) == 1


def test_anomaly_details_round_trip_as_jsonb(session) -> None:
    insert_anomalies(session, [{
        "kind": "encounter", "mmsi": 1, "counterpart_mmsi": 2, "start_ts": NOW,
        "lon": 90.0, "lat": 21.0,
        "details": {"duration_hours": 4.5, "median_separation_km": 0.21},
    }])
    session.commit()
    anomaly = session.execute(select(Anomaly)).scalar_one()
    assert anomaly.details["median_separation_km"] == 0.21
    assert anomaly.counterpart_mmsi == 2


# --- end-to-end fusion -----------------------------------------------------


def test_full_fusion_path_produces_all_three_verdicts(session) -> None:
    """detect -> persist -> interpolate AIS -> match -> store verdict."""
    scene = make_scene(session)
    target = scene.acquisition_midpoint

    # A vessel broadcasting AIS right where SAR sees something, plus two
    # neighbours that establish local AIS coverage.
    upsert_ais_records(session, [
        ais(111, target - timedelta(minutes=2), 90.4990, 21.3, sog_kn=5.0, cog_deg=90.0),
        ais(111, target + timedelta(minutes=2), 90.5010, 21.3, sog_kn=5.0, cog_deg=90.0),
        ais(222, target - timedelta(minutes=2), 90.520, 21.31, sog_kn=5.0, cog_deg=90.0),
        ais(222, target + timedelta(minutes=2), 90.522, 21.31, sog_kn=5.0, cog_deg=90.0),
    ])
    insert_detections(session, scene, [
        {"lon": 90.5000, "lat": 21.3, "row": 10, "col": 10, "snr_db": 18.0,
         "length_m": 40.0, "confidence": 0.7},   # on top of MMSI 111 -> matched
        {"lon": 90.5400, "lat": 21.3, "row": 20, "col": 20, "snr_db": 15.0,
         "length_m": 25.0, "confidence": 0.6},   # AIS nearby, none matching -> dark
        {"lon": 88.6000, "lat": 20.6, "row": 30, "col": 30, "snr_db": 12.0,
         "length_m": 20.0, "confidence": 0.4},   # no AIS coverage -> indeterminate
    ])
    session.commit()

    # Rebuild tracks from the database, exactly as the worker does.
    rows = session.execute(text(
        "SELECT mmsi, ts, ST_X(geom::geometry) lon, ST_Y(geom::geometry) lat, "
        "sog_kn, cog_deg FROM ais_positions ORDER BY mmsi, ts"
    )).mappings()
    tracks: dict[int, list[Fix]] = {}
    for row in rows:
        tracks.setdefault(row["mmsi"], []).append(
            Fix(row["ts"], row["lon"], row["lat"], row["sog_kn"], row["cog_deg"])
        )

    positions = interpolate_fleet(tracks, target)
    assert len(positions) == 2
    assert all(p.method == "linear" and p.trusted for p in positions)

    detection_rows = session.execute(text(
        "SELECT id, ST_X(geom::geometry) lon, ST_Y(geom::geometry) lat, length_m "
        "FROM detections ORDER BY id"
    )).mappings()
    inputs = [DetectionInput(r["id"], r["lon"], r["lat"], r["length_m"]) for r in detection_rows]

    results = match_detections(inputs, positions, base_radius_m=500.0)
    assert summarise(results) == {"matched": 1, "dark": 1, "indeterminate": 1}

    for result in results:
        session.query(Detection).filter(Detection.id == result.detection_id).update(
            {"is_dark": result.is_dark, "matched_mmsi": result.matched_mmsi,
             "match_distance_m": result.distance_m,
             "match_notes": {**result.notes, "status": result.status}},
            synchronize_session=False,
        )
    session.commit()

    # The tri-state verdict must be queryable the way the API queries it.
    assert session.scalar(text("SELECT count(*) FROM detections WHERE is_dark IS true")) == 1
    assert session.scalar(text("SELECT count(*) FROM detections WHERE is_dark IS false")) == 1
    assert session.scalar(text("SELECT count(*) FROM detections WHERE is_dark IS NULL")) == 1

    matched = session.execute(
        select(Detection).where(Detection.is_dark.is_(False))
    ).scalar_one()
    assert matched.matched_mmsi == 111
    assert matched.match_distance_m < 500
    assert matched.match_notes["status"] == "matched"


def test_dark_detections_can_be_filtered_to_the_eez(session) -> None:
    """The operational query: dark detections inside the zone we police."""
    upsert_region(session, name="Bangladesh EEZ (approximate)", kind="eez",
                  mrgid=8481, geom_wkt=geojson_to_wkt(BANGLADESH_EEZ_APPROX))
    scene = make_scene(session)
    insert_detections(session, scene, [
        {"lon": 90.5, "lat": 21.3, "row": 1, "col": 1, "snr_db": 15.0},  # inside
        {"lon": 80.0, "lat": 15.0, "row": 2, "col": 2, "snr_db": 15.0},  # outside
    ])
    session.commit()
    session.execute(text("UPDATE detections SET is_dark = true"))
    session.commit()

    count = session.scalar(text(
        """
        SELECT count(*) FROM detections d
        JOIN regions r ON r.mrgid = 8481
        WHERE d.is_dark IS true
          AND ST_Contains(r.geom::geometry, d.geom::geometry)
        """
    ))
    assert count == 1


# --- API against real PostGIS ---------------------------------------------
#
# The mocked API tests in test_api.py never compile their statements against a
# real dialect, so they cannot catch invalid SQL construction.  These do.


@pytest.fixture
def api_client(engine, session):
    from fastapi.testclient import TestClient

    from bob_sentinel.config import Settings, get_settings
    from bob_sentinel.db import get_db
    from bob_sentinel.main import create_app

    app = create_app()
    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


def seed_fusion_fixture(session):
    """One matched, one dark and one indeterminate detection, plus an anomaly."""
    scene = make_scene(session)
    target = scene.acquisition_midpoint
    upsert_ais_records(session, [
        ais(111, target - timedelta(minutes=2), 90.4990, 21.3, sog_kn=5.0, cog_deg=90.0),
        ais(111, target + timedelta(minutes=2), 90.5010, 21.3, sog_kn=5.0, cog_deg=90.0),
        ais(222, target - timedelta(minutes=2), 90.520, 21.31, sog_kn=5.0, cog_deg=90.0),
        ais(222, target + timedelta(minutes=2), 90.522, 21.31, sog_kn=5.0, cog_deg=90.0),
    ])
    insert_detections(session, scene, [
        {"lon": 90.5000, "lat": 21.3, "row": 10, "col": 10, "snr_db": 18.0,
         "length_m": 40.0, "confidence": 0.7},
        {"lon": 90.5400, "lat": 21.3, "row": 20, "col": 20, "snr_db": 15.0,
         "length_m": 25.0, "confidence": 0.6},
        {"lon": 88.6000, "lat": 20.6, "row": 30, "col": 30, "snr_db": 12.0,
         "length_m": 20.0, "confidence": 0.4},
    ])
    session.execute(text(
        "UPDATE detections SET is_dark = CASE WHEN row = 10 THEN false "
        "WHEN row = 20 THEN true ELSE NULL END, "
        "matched_mmsi = CASE WHEN row = 10 THEN 111 ELSE NULL END"
    ))
    insert_anomalies(session, [{
        "kind": "ais_gap", "mmsi": 111, "start_ts": NOW,
        "end_ts": NOW + timedelta(hours=8), "lon": 90.4, "lat": 21.2,
        "score": 0.33, "details": {"duration_hours": 8.0},
    }])
    upsert_region(session, name="Bangladesh EEZ (approximate)", kind="eez",
                  mrgid=8481, geom_wkt=geojson_to_wkt(BANGLADESH_EEZ_APPROX))
    session.commit()
    return scene


def test_healthz_reports_a_live_postgis(api_client) -> None:
    payload = api_client.get("/healthz").json()
    assert payload["status"] == "ok"
    assert payload["components"]["database"]["ok"] is True
    assert "PostGIS" in payload["components"]["database"]["detail"]


def test_stats_counts_each_verdict(api_client, session) -> None:
    seed_fusion_fixture(session)
    stats = api_client.get("/api/stats").json()
    assert stats["detections"] == 3
    assert stats["dark_detections"] == 1
    assert stats["indeterminate_detections"] == 1
    assert stats["anomalies"] == 1
    assert stats["vessels"] == 2


def test_detections_endpoint_serialises_real_geometry(api_client, session) -> None:
    """Regression: an uncacheable geometry cast made this endpoint 500."""
    seed_fusion_fixture(session)
    response = api_client.get("/api/detections")
    assert response.status_code == 200

    features = response.json()["features"]
    assert len(features) == 3
    for feature in features:
        lon, lat = feature["geometry"]["coordinates"]
        assert 88.0 < lon < 93.0 and 20.0 < lat < 23.0, "coordinates are lon,lat"
    assert {f["properties"]["status"] for f in features} == {
        "matched", "dark", "indeterminate"
    }


@pytest.mark.parametrize(
    "status,expected", [("dark", 1), ("matched", 1), ("indeterminate", 1)]
)
def test_status_filters_hit_the_database(api_client, session, status, expected) -> None:
    seed_fusion_fixture(session)
    features = api_client.get(f"/api/detections?status={status}").json()["features"]
    assert len(features) == expected
    assert features[0]["properties"]["status"] == status


def test_anomalies_endpoint_serialises_real_geometry(api_client, session) -> None:
    seed_fusion_fixture(session)
    anomalies = api_client.get("/api/anomalies").json()
    assert len(anomalies) == 1
    assert anomalies[0]["kind"] == "ais_gap"
    assert anomalies[0]["lon"] == pytest.approx(90.4)
    assert anomalies[0]["details"]["duration_hours"] == 8.0


def test_latest_ais_returns_one_feature_per_vessel(api_client, session) -> None:
    seed_fusion_fixture(session)
    features = api_client.get("/api/ais/latest?minutes=1440").json()["features"]
    assert len(features) == 2, "DISTINCT ON should collapse to the newest per MMSI"
    assert {f["properties"]["mmsi"] for f in features} == {111, 222}


def test_track_endpoint_builds_a_linestring(api_client, session) -> None:
    seed_fusion_fixture(session)
    features = api_client.get("/api/ais/track/111?hours=24").json()["features"]
    assert [f["geometry"]["type"] for f in features] == ["LineString", "Point"]
    assert len(features[0]["geometry"]["coordinates"]) == 2


def test_regions_endpoint_returns_valid_geojson(api_client, session) -> None:
    seed_fusion_fixture(session)
    features = api_client.get("/api/regions").json()["features"]
    assert len(features) == 1
    geometry = features[0]["geometry"]
    assert geometry["type"] in {"Polygon", "MultiPolygon"}
    assert geometry["coordinates"], "simplification must not empty the polygon"
    assert features[0]["properties"]["mrgid"] == 8481


def test_scenes_endpoint_reports_detection_counts(api_client, session) -> None:
    seed_fusion_fixture(session)
    scenes = api_client.get("/api/scenes").json()
    assert len(scenes) == 1
    assert scenes[0]["detection_count"] == 3
    assert scenes[0]["platform"] == "S1D"
