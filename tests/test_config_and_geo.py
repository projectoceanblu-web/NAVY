"""Configuration wiring and GeoJSON/WKT conversion."""

from __future__ import annotations

import json

import pytest

from bob_sentinel.config import Settings
from bob_sentinel.ingest.eez import (
    BANGLADESH_EEZ_APPROX,
    BANGLADESH_MRGID,
    geojson_to_wkt,
)

# --- settings --------------------------------------------------------------


def test_database_url_is_assembled_from_parts() -> None:
    settings = Settings(
        postgres_user="u", postgres_password="p", postgres_host="db",
        postgres_port=5433, postgres_db="d", _env_file=None,
    )
    assert settings.sqlalchemy_url == "postgresql+psycopg://u:p@db:5433/d"


def test_explicit_database_url_wins() -> None:
    settings = Settings(database_url="postgresql+psycopg://x:y@h/z", _env_file=None)
    assert settings.sqlalchemy_url == "postgresql+psycopg://x:y@h/z"


def test_aisstream_bbox_is_lat_first() -> None:
    """aisstream reverses the GeoJSON axis order; aoi_bbox keeps it."""
    settings = Settings(_env_file=None)
    assert settings.aisstream_bbox() == [[[20.5, 88.0], [22.8, 92.7]]]
    assert settings.aoi_bbox == (88.0, 20.5, 92.7, 22.8)


def test_credentials_default_to_none_so_the_app_still_starts() -> None:
    """A missing GFW token must not stop AIS ingestion."""
    settings = Settings(_env_file=None)
    assert settings.cdse_username is None
    assert settings.aisstream_api_key is None
    assert settings.gfw_api_token is None


def test_aoi_defaults_to_the_bangladesh_eez() -> None:
    assert Settings(_env_file=None).aoi_mrgid == BANGLADESH_MRGID == 8481


def test_env_vars_override_defaults(monkeypatch) -> None:
    monkeypatch.setenv("MATCH_RADIUS_M", "1200")
    monkeypatch.setenv("CFAR_PFA", "1e-6")
    settings = Settings(_env_file=None)
    assert settings.match_radius_m == 1200.0
    assert settings.cfar_pfa == 1e-6


# --- geometry --------------------------------------------------------------


def test_polygon_converts_to_wkt() -> None:
    wkt = geojson_to_wkt({"type": "Polygon", "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]})
    assert wkt == "POLYGON((0.0 0.0,1.0 0.0,1.0 1.0,0.0 0.0))"


def test_multipolygon_converts_to_wkt() -> None:
    wkt = geojson_to_wkt(
        {"type": "MultiPolygon", "coordinates": [[[[0, 0], [1, 0], [1, 1], [0, 0]]]]}
    )
    assert wkt.startswith("MULTIPOLYGON(((")


def test_polygon_holes_are_preserved() -> None:
    """An inner ring matters: it can be an island excluded from the zone."""
    wkt = geojson_to_wkt(
        {
            "type": "Polygon",
            "coordinates": [
                [[0, 0], [10, 0], [10, 10], [0, 0]],
                [[2, 2], [3, 2], [3, 3], [2, 2]],
            ],
        }
    )
    assert wkt.count("(") == 3


def test_elevation_ordinates_are_ignored() -> None:
    wkt = geojson_to_wkt(
        {"type": "Polygon", "coordinates": [[[0, 0, 5], [1, 0, 5], [1, 1, 5], [0, 0, 5]]]}
    )
    assert "5" not in wkt.replace("0.0", "").replace("1.0", "")


def test_unsupported_geometry_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported geometry"):
        geojson_to_wkt({"type": "LineString", "coordinates": [[0, 0], [1, 1]]})


def test_fallback_outline_is_a_closed_ring_in_the_bay_of_bengal() -> None:
    ring = BANGLADESH_EEZ_APPROX["coordinates"][0]
    assert ring[0] == ring[-1], "polygon ring must close"
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    assert 88.0 <= min(lons) and max(lons) <= 93.0
    assert 20.0 <= min(lats) and max(lats) <= 23.0


def test_fallback_outline_is_marked_as_approximate() -> None:
    """It must never be mistaken for the authoritative VLIZ boundary."""
    import inspect

    from bob_sentinel.ingest import eez

    source = inspect.getsource(eez.load_fallback_bangladesh_eez)
    assert "APPROXIMATE" in source
    assert "NOT the authoritative" in source
    assert "(approximate)" in source, "the stored region name must say so too"


def test_geojson_round_trip_from_a_feature_collection(tmp_path) -> None:
    from bob_sentinel.ingest.eez import load_geojson_region

    payload = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "properties": {},
             "geometry": {"type": "Polygon",
                          "coordinates": [[[0, 0], [1, 0], [1, 1], [0, 0]]]}},
            {"type": "Feature", "properties": {},
             "geometry": {"type": "Polygon",
                          "coordinates": [[[5, 5], [6, 5], [6, 6], [5, 5]]]}},
        ],
    }
    path = tmp_path / "eez.geojson"
    path.write_text(json.dumps(payload))

    captured = {}

    class FakeSession:
        pass

    def fake_upsert(session, *, name, kind, geom_wkt, mrgid, source):
        captured.update(geom_wkt=geom_wkt, name=name, mrgid=mrgid)
        return "region"

    import bob_sentinel.ingest.eez as eez_module

    original = eez_module.upsert_region
    eez_module.upsert_region = fake_upsert
    try:
        load_geojson_region(FakeSession(), path, name="Test EEZ", mrgid=8481)
    finally:
        eez_module.upsert_region = original

    # Both features must be dissolved into a single MULTIPOLYGON.
    assert captured["geom_wkt"].startswith("MULTIPOLYGON(")
    assert captured["geom_wkt"].count("((") == 2
    assert captured["mrgid"] == 8481
