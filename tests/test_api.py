"""API surface tests.

These run without a database: the DB dependency is overridden with a fake
session so route wiring, filtering and serialisation can be checked in
isolation.  Queries against real PostGIS are covered by the integration tests.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine

from bob_sentinel.config import Settings, get_settings
from bob_sentinel.db import get_db
from bob_sentinel.main import create_app

NOW = datetime(2026, 1, 3, 11, 0, tzinfo=UTC)


class FakeResult:
    def __init__(self, rows):
        self._rows = rows

    def mappings(self):
        return self._rows

    def __iter__(self):
        return iter(self._rows)

    def all(self):
        return self._rows

    def scalar(self):
        return self._rows[0] if self._rows else None


class FakeSession:
    """Records the SQL it is asked to run and replays canned rows."""

    def __init__(self, rows=None, scalars=None):
        self.rows = rows or []
        self.scalars = scalars or {}
        self.statements = []

    def execute(self, statement, params=None):
        self.statements.append((str(statement), params))
        return FakeResult(self.rows)

    def scalar(self, statement):
        self.statements.append((str(statement), None))
        return self.scalars.get("default", 0)


@pytest.fixture
def client():
    app = create_app()
    session = FakeSession()

    app.dependency_overrides[get_db] = lambda: session
    app.dependency_overrides[get_settings] = lambda: Settings(_env_file=None)
    test_client = TestClient(app)
    test_client.session = session  # type: ignore[attr-defined]
    yield test_client
    app.dependency_overrides.clear()


# --- documentation & health ------------------------------------------------


def test_openapi_exposes_every_route(client) -> None:
    paths = client.get("/openapi.json").json()["paths"]
    assert set(paths) >= {
        "/healthz", "/api/stats", "/api/ais/latest", "/api/ais/track/{mmsi}",
        "/api/detections", "/api/scenes", "/api/anomalies", "/api/regions",
    }


def test_api_description_carries_the_dual_use_caveat(client) -> None:
    """The limits of this tool must travel with the tool."""
    description = client.get("/openapi.json").json()["info"]["description"]
    assert "not evidence" in description.lower()
    assert "indeterminate" in description.lower()


def test_healthz_reports_unconfigured_credentials(client) -> None:
    payload = client.get("/healthz").json()
    assert payload["status"] in {"ok", "degraded"}
    assert set(payload["components"]) == {"database", "cdse", "aisstream", "gfw"}
    assert payload["components"]["gfw"]["ok"] is False
    assert "GFW_API_TOKEN" in payload["components"]["gfw"]["detail"]


def test_healthz_answers_even_when_the_database_is_down(client, monkeypatch) -> None:
    """Liveness must not depend on the thing it is reporting on.

    The unreachable engine is injected rather than assumed: this test used to
    rely on no database happening to listen on the default port, so it passed
    on a dev box and failed in CI, where PostGIS is up. A test about a
    database being down has to put it down itself.
    """
    import bob_sentinel.db as db_module

    unreachable = create_engine(
        # Port 1 is reserved and never listening, so this refuses immediately
        # instead of hanging the suite on a connect timeout.
        "postgresql+psycopg://nobody:nobody@127.0.0.1:1/nonexistent",
        connect_args={"connect_timeout": 1},
    )
    monkeypatch.setattr(db_module, "_engine", unreachable)

    response = client.get("/healthz")
    assert response.status_code == 200
    payload = response.json()
    assert payload["components"]["database"]["ok"] is False
    assert payload["status"] == "degraded"
    # The failure is reported, not swallowed.
    assert payload["components"]["database"]["detail"]


# --- AIS -------------------------------------------------------------------


def test_latest_ais_returns_geojson(client) -> None:
    client.session.rows = [
        {"mmsi": 405000123, "ts": NOW, "lon": 90.2, "lat": 21.5, "sog_kn": 8.4,
         "cog_deg": 142.0, "name": "FV EXAMPLE", "ship_type": 30}
    ]
    payload = client.get("/api/ais/latest?minutes=60").json()
    assert payload["type"] == "FeatureCollection"
    feature = payload["features"][0]
    assert feature["geometry"]["coordinates"] == [90.2, 21.5], "GeoJSON is lon,lat"
    assert feature["properties"]["mmsi"] == 405000123


def test_latest_ais_validates_its_window(client) -> None:
    assert client.get("/api/ais/latest?minutes=0").status_code == 422
    assert client.get("/api/ais/latest?minutes=999999").status_code == 422


def test_track_returns_a_linestring_and_an_endpoint(client) -> None:
    client.session.rows = [
        {"ts": NOW, "lon": 90.0, "lat": 21.0, "sog_kn": 8.0, "cog_deg": 90.0},
        {"ts": NOW, "lon": 90.1, "lat": 21.0, "sog_kn": 8.0, "cog_deg": 90.0},
    ]
    features = client.get("/api/ais/track/405000123").json()["features"]
    assert [f["geometry"]["type"] for f in features] == ["LineString", "Point"]
    assert len(features[0]["geometry"]["coordinates"]) == 2


def test_track_for_an_unknown_vessel_is_404(client) -> None:
    client.session.rows = []
    assert client.get("/api/ais/track/1").status_code == 404


# --- detections ------------------------------------------------------------


def test_detection_status_filter_is_validated(client) -> None:
    assert client.get("/api/detections?status=dark").status_code == 200
    assert client.get("/api/detections?status=guilty").status_code == 422


@pytest.mark.parametrize(
    "status,sql_fragment",
    [("dark", "IS true"), ("matched", "IS false"), ("indeterminate", "IS NULL")],
)
def test_status_filter_maps_to_the_right_sql(client, status, sql_fragment) -> None:
    """The tri-state verdict must survive into the query itself."""
    client.get(f"/api/detections?status={status}")
    compiled = " ".join(s for s, _ in client.session.statements)
    assert sql_fragment.lower() in compiled.lower()


def test_gfw_tile_without_a_token_is_503_not_500(client) -> None:
    """An unconfigured optional integration is a service state, not a crash."""
    response = client.get("/api/gfw/tile/5/23/13")
    assert response.status_code == 503
    assert "GFW_API_TOKEN" in response.json()["detail"]
