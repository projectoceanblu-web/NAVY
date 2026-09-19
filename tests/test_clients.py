"""CDSE and GFW clients, driven against mocked HTTP transports."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import httpx
import pytest

from bob_sentinel.clients.gfw import (
    BANGLADESH_EEZ_ID,
    EVENT_DATASETS,
    SAR_PRESENCE,
    GFWClient,
    GFWError,
    format_date_range,
)
from bob_sentinel.sar.cdse import (
    ACTIVE_PLATFORMS,
    CDSEClient,
    CDSEError,
    Product,
    build_odata_filter,
)

START = datetime(2026, 1, 1, tzinfo=UTC)
END = datetime(2026, 1, 7, tzinfo=UTC)


def mock_client(handler) -> httpx.Client:
    return httpx.Client(transport=httpx.MockTransport(handler))


# --- CDSE: filter construction --------------------------------------------


def test_filter_targets_grd_iw_products() -> None:
    f = build_odata_filter(start=START, end=END)
    assert "Collection/Name eq 'SENTINEL-1'" in f
    assert "'IW_GRDH_1S'" in f
    assert "ContentDate/Start ge 2026-01-01T00:00:00.000Z" in f


def test_filter_defaults_to_the_active_constellation() -> None:
    """S1A concluded operations 2026-06-30; only S1C/S1D still acquire."""
    assert ACTIVE_PLATFORMS == ("S1C", "S1D")
    f = build_odata_filter(start=START, end=END)
    assert "startswith(Name,'S1C_')" in f
    assert "startswith(Name,'S1D_')" in f
    assert "S1A" not in f


def test_filter_includes_the_spatial_intersection() -> None:
    wkt = "POLYGON((88 20,92 20,92 23,88 23,88 20))"
    assert f"OData.CSC.Intersects(area=geography'SRID=4326;{wkt}')" in build_odata_filter(
        start=START, end=END, wkt=wkt
    )


def test_naive_datetimes_are_treated_as_utc() -> None:
    assert "2026-01-01T00:00:00.000Z" in build_odata_filter(
        start=datetime(2026, 1, 1), end=datetime(2026, 1, 7)
    )


# --- CDSE: auth ------------------------------------------------------------


def test_password_grant_is_used_and_token_is_cached() -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(httpx.QueryParams(request.content.decode())))
        return httpx.Response(200, json={"access_token": "tok-1", "refresh_token": "r-1",
                                         "expires_in": 600})

    client = CDSEClient("user", "pw", client=mock_client(handler))
    assert client.token == "tok-1"
    assert client.token == "tok-1", "a valid token must not be re-fetched"
    assert len(calls) == 1
    assert calls[0]["grant_type"] == "password"
    assert calls[0]["client_id"] == "cdse-public"


def test_expired_token_is_refreshed_with_the_refresh_token() -> None:
    """Access tokens live ~10 minutes; long downloads outlive them."""
    grants = []

    def handler(request: httpx.Request) -> httpx.Response:
        data = dict(httpx.QueryParams(request.content.decode()))
        grants.append(data["grant_type"])
        return httpx.Response(200, json={"access_token": f"tok-{len(grants)}",
                                         "refresh_token": "r-1", "expires_in": 600})

    client = CDSEClient("user", "pw", client=mock_client(handler))
    assert client.token == "tok-1"
    client._expires_at = 0  # simulate expiry
    assert client.token == "tok-2"
    assert grants == ["password", "refresh_token"]


def test_rejected_refresh_falls_back_to_a_full_login() -> None:
    grants = []

    def handler(request: httpx.Request) -> httpx.Response:
        data = dict(httpx.QueryParams(request.content.decode()))
        grants.append(data["grant_type"])
        if data["grant_type"] == "refresh_token":
            return httpx.Response(400, json={"error": "invalid_grant"})
        return httpx.Response(200, json={"access_token": "fresh", "refresh_token": "r",
                                         "expires_in": 600})

    client = CDSEClient("user", "pw", client=mock_client(handler))
    assert client.token == "fresh"
    client._expires_at = 0
    assert client.token == "fresh"
    assert grants == ["password", "refresh_token", "password"]


def test_missing_credentials_fail_fast() -> None:
    with pytest.raises(CDSEError, match="credentials missing"):
        CDSEClient("", "")


def test_bad_credentials_surface_the_server_error() -> None:
    client = CDSEClient(
        "u", "p", client=mock_client(lambda r: httpx.Response(401, text="unauthorized"))
    )
    with pytest.raises(CDSEError, match="401"):
        _ = client.token


# --- CDSE: search & products ----------------------------------------------


def odata_product(name="S1D_IW_GRDH_1SDV_20260103T114217_0001", pid="abc-123"):
    return {
        "Id": pid,
        "Name": name,
        "Footprint": "geography'SRID=4326;POLYGON((88 20,92 20,92 23,88 23,88 20))'",
        "ContentDate": {"Start": "2026-01-03T11:42:17.000Z", "End": "2026-01-03T11:42:45.000Z"},
        "ContentLength": 1_700_000_000,
        "Online": True,
        "Attributes": [{"Name": "productType", "Value": "IW_GRDH_1S"}],
    }


def test_product_parses_footprint_and_platform() -> None:
    product = Product.from_odata(odata_product())
    assert product.platform == "S1D"
    assert product.product_type == "IW_GRDH_1S"
    assert product.footprint_wkt.startswith("POLYGON((")
    assert "geography" not in product.footprint_wkt
    assert product.acquired_start.tzinfo is not None


def test_search_follows_pagination_and_honours_the_limit() -> None:
    pages = [
        {"value": [odata_product(pid=f"p{i}") for i in range(2)],
         "@odata.nextLink": "https://catalogue.dataspace.copernicus.eu/odata/v1/Products?skip=2"},
        {"value": [odata_product(pid=f"p{i}") for i in range(2, 4)]},
    ]
    seen = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if "token" in str(request.url):
            return httpx.Response(200, json={"access_token": "t", "expires_in": 600})
        page = pages[min(seen["n"], len(pages) - 1)]
        seen["n"] += 1
        return httpx.Response(200, json=page)

    client = CDSEClient("u", "p", client=mock_client(handler))
    assert len(client.search(start=START, end=END, limit=3, page_size=2)) == 3


def test_search_failure_is_reported() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "token" in str(request.url):
            return httpx.Response(200, json={"access_token": "t", "expires_in": 600})
        return httpx.Response(500, text="boom")

    client = CDSEClient("u", "p", client=mock_client(handler))
    with pytest.raises(CDSEError, match="500"):
        client.search(start=START, end=END)


def test_download_writes_atomically(tmp_path) -> None:
    """A partial download must never be mistaken for a finished product."""
    def handler(request: httpx.Request) -> httpx.Response:
        if "token" in str(request.url):
            return httpx.Response(200, json={"access_token": "t", "expires_in": 600})
        return httpx.Response(200, content=b"PK\x03\x04payload")

    client = CDSEClient("u", "p", client=mock_client(handler))
    product = Product.from_odata(odata_product())
    path = client.download(product, tmp_path)

    assert path.exists() and path.read_bytes() == b"PK\x03\x04payload"
    assert not list(tmp_path.glob("*.part")), "a .part file was left behind"

    # Second call is a no-op rather than a re-download.
    assert client.download(product, tmp_path) == path


# --- GFW -------------------------------------------------------------------


def test_date_range_formatting_and_cap() -> None:
    assert format_date_range(date(2026, 1, 1), date(2026, 1, 6)) == "2026-01-01,2026-01-06"
    with pytest.raises(GFWError, match="caps it at 366"):
        format_date_range(date(2024, 1, 1), date(2026, 1, 6))
    with pytest.raises(GFWError, match="precedes"):
        format_date_range(date(2026, 2, 1), date(2026, 1, 1))


def test_sar_report_requests_dark_detections_over_the_eez() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = request.url
        captured["body"] = json.loads(request.content)
        captured["auth"] = request.headers["Authorization"]
        return httpx.Response(200, json={"entries": []})

    with GFWClient("tok", client=mock_client(handler)) as client:
        client.sar_report(start=date(2026, 1, 1), end=date(2026, 1, 6))

    params = captured["url"].params
    assert params["datasets[0]"] == SAR_PRESENCE
    assert params["filters[0]"] == "matched='false'", "dark-detection filter missing"
    assert captured["body"]["region"] == {"dataset": "public-eez-areas",
                                          "id": BANGLADESH_EEZ_ID}
    assert captured["auth"] == "Bearer tok"
    assert str(captured["url"]).startswith("https://gateway.api.globalfishingwatch.org/v3/")


def test_sar_report_can_request_all_detections() -> None:
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["params"] = request.url.params
        return httpx.Response(200, json={})

    with GFWClient("tok", client=mock_client(handler)) as client:
        client.sar_report(start=date(2026, 1, 1), end=date(2026, 1, 6), dark_only=False)
    assert "filters[0]" not in captured["params"]


def test_events_rejects_unknown_types() -> None:
    with GFWClient("tok", client=mock_client(lambda r: httpx.Response(200, json={}))) as c:
        with pytest.raises(GFWError, match="unknown event types"):
            c.events(types=["TELEPORT"])


def test_event_datasets_cover_the_documented_types() -> None:
    assert set(EVENT_DATASETS) == {"ENCOUNTER", "FISHING", "PORT_VISIT", "LOITERING", "GAP"}
    assert all(v.startswith("public-global-") for v in EVENT_DATASETS.values())


@pytest.mark.parametrize(
    "status,pattern",
    [(401, "rejected the token"), (429, "rate limit"), (500, "failed")],
)
def test_http_errors_are_translated(status: int, pattern: str) -> None:
    with GFWClient("tok", client=mock_client(lambda r: httpx.Response(status, text="x"))) as c:
        with pytest.raises(GFWError, match=pattern):
            c.list_datasets()


def test_missing_token_fails_fast() -> None:
    with pytest.raises(GFWError, match="GFW_API_TOKEN"):
        GFWClient("")


def test_tile_url_never_embeds_the_token() -> None:
    """A token in a browser tile URL is a published token."""
    with GFWClient("secret-token", client=mock_client(lambda r: httpx.Response(200))) as c:
        assert "secret-token" not in c.tile_url(5, 23, 13)
