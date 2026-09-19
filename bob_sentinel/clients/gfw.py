"""Global Fishing Watch API v3 client.

Used as an independent cross-check on our own detections: GFW publishes its
own Sentinel-1 SAR presence layer with an AIS-match flag, so ``matched='false'``
is their dark-detection view of the same water we are looking at.

Scope and limits worth knowing before relying on it:

* Non-commercial use only (academic, NGO, public-good).
* Rate limits apply per *user* across all tokens, max 5 tokens per user.
* The SAR layer runs from 2017 to roughly 5 days ago — it is not live.
* Per-detection extras (length_m, presence_score, matching_score) are only in
  the Data Download Portal, not this API.  Reports are gridded counts.
* GFW reported SAR freshness disruption in Aug 2026 tied to Sentinel-1A's
  retirement, so treat a thin result as possibly upstream, not as empty seas.

Dataset IDs for loitering and gaps follow the v3 naming convention but are
worth confirming against ``GET /v3/datasets`` for your token before relying
on them — :meth:`GFWClient.list_datasets` exists for exactly that.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import date, datetime
from typing import Any

import httpx

log = logging.getLogger(__name__)

BASE_URL = "https://gateway.api.globalfishingwatch.org/v3"

# --- dataset identifiers ---------------------------------------------------
SAR_PRESENCE = "public-global-sar-presence:latest"
AIS_PRESENCE = "public-global-presence:latest"
FISHING_EFFORT = "public-global-fishing-effort:latest"
VESSEL_IDENTITY = "public-global-vessel-identity:latest"

EVENT_DATASETS = {
    "ENCOUNTER": "public-global-encounters-events:latest",
    "FISHING": "public-global-fishing-events:latest",
    "PORT_VISIT": "public-global-port-visits-events:latest",
    "LOITERING": "public-global-loitering-events:latest",
    "GAP": "public-global-gaps-events:latest",
}

EEZ_REGION_DATASET = "public-eez-areas"
BANGLADESH_EEZ_ID = 8481

#: The API rejects windows longer than this.
MAX_DATE_RANGE_DAYS = 366


class GFWError(RuntimeError):
    pass


def format_date_range(start: date | datetime, end: date | datetime) -> str:
    """``date-range`` is ``YYYY-MM-DD,YYYY-MM-DD`` and capped at 366 days."""
    start_d = start.date() if isinstance(start, datetime) else start
    end_d = end.date() if isinstance(end, datetime) else end
    if end_d < start_d:
        raise GFWError("date-range end precedes its start")
    span = (end_d - start_d).days
    if span > MAX_DATE_RANGE_DAYS:
        raise GFWError(
            f"date-range spans {span} days; the API caps it at {MAX_DATE_RANGE_DAYS}"
        )
    return f"{start_d.isoformat()},{end_d.isoformat()}"


class GFWClient:
    """Minimal v3 client covering the endpoints this MVP uses."""

    def __init__(
        self,
        token: str,
        *,
        base_url: str = BASE_URL,
        client: httpx.Client | None = None,
        timeout: float = 60.0,
    ) -> None:
        if not token:
            raise GFWError("GFW_API_TOKEN is not set")
        self.base_url = base_url.rstrip("/")
        # Auth is applied per request rather than baked into the client's
        # default headers, so an injected client (tests, a shared pool, a
        # proxy-aware client) cannot silently drop authentication.
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/json",
            "User-Agent": "bob-sentinel",
        }
        self._client = client or httpx.Client(timeout=timeout)

    # -- plumbing -----------------------------------------------------------

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        headers = {**self._headers, **kwargs.pop("headers", {})}
        response = self._client.request(method, url, headers=headers, **kwargs)
        if response.status_code == 401:
            raise GFWError("GFW rejected the token (401) — expired or malformed")
        if response.status_code == 429:
            raise GFWError(
                "GFW rate limit hit (429); limits are per user across all tokens"
            )
        if response.status_code >= 400:
            raise GFWError(
                f"GFW {method} {path} failed ({response.status_code}): {response.text[:300]}"
            )
        if not response.content:
            return None
        try:
            return response.json()
        except ValueError:
            return response.content

    # -- datasets -----------------------------------------------------------

    def list_datasets(self) -> Any:
        """Enumerate datasets this token can see — use to confirm v3 IDs."""
        return self._request("GET", "/datasets")

    # -- 4Wings (gridded SAR / AIS presence) --------------------------------

    def sar_report(
        self,
        *,
        start: date | datetime,
        end: date | datetime,
        region_id: int = BANGLADESH_EEZ_ID,
        region_dataset: str = EEZ_REGION_DATASET,
        dark_only: bool = True,
        spatial_resolution: str = "HIGH",
        temporal_resolution: str = "DAILY",
        response_format: str = "JSON",
    ) -> Any:
        """Gridded SAR detection counts over a region.

        ``dark_only`` adds ``filters[0]=matched='false'`` — GFW's SAR
        detections that could not be reconciled with an AIS broadcast.
        """
        params: list[tuple[str, str]] = [
            ("spatial-resolution", spatial_resolution),
            ("temporal-resolution", temporal_resolution),
            ("datasets[0]", SAR_PRESENCE),
            ("date-range", format_date_range(start, end)),
            ("format", response_format),
        ]
        if dark_only:
            params.append(("filters[0]", "matched='false'"))
        body = {"region": {"dataset": region_dataset, "id": region_id}}
        return self._request("POST", "/4wings/report", params=params, json=body)

    def presence_report(
        self,
        *,
        dataset: str,
        start: date | datetime,
        end: date | datetime,
        region_id: int = BANGLADESH_EEZ_ID,
        region_dataset: str = EEZ_REGION_DATASET,
        spatial_resolution: str = "HIGH",
        temporal_resolution: str = "DAILY",
        response_format: str = "JSON",
    ) -> Any:
        """Gridded activity for any 4Wings dataset (AIS presence, fishing effort)."""
        params: list[tuple[str, str]] = [
            ("spatial-resolution", spatial_resolution),
            ("temporal-resolution", temporal_resolution),
            ("datasets[0]", dataset),
            ("date-range", format_date_range(start, end)),
            ("format", response_format),
        ]
        body = {"region": {"dataset": region_dataset, "id": region_id}}
        return self._request("POST", "/4wings/report", params=params, json=body)

    def stats(
        self,
        *,
        datasets: Sequence[str] = (SAR_PRESENCE,),
        start: date | datetime,
        end: date | datetime,
    ) -> Any:
        params = [("datasets[0]", datasets[0]), ("date-range", format_date_range(start, end))]
        for i, dataset in enumerate(datasets[1:], start=1):
            params.append((f"datasets[{i}]", dataset))
        return self._request("GET", "/4wings/stats", params=params)

    def tile_url(self, z: int, x: int, y: int, *, dataset: str = SAR_PRESENCE) -> str:
        """Heatmap tile URL.

        Returned rather than fetched so the API layer can proxy it and keep
        the token server-side — a token embedded in a Leaflet URL is a
        published token.
        """
        return f"{self.base_url}/4wings/tile/heatmap/{z}/{x}/{y}?datasets[0]={dataset}"

    def get_tile(self, z: int, x: int, y: int, *, dataset: str = SAR_PRESENCE,
                 date_range: str | None = None) -> bytes:
        params = [("datasets[0]", dataset)]
        if date_range:
            params.append(("date-range", date_range))
        url = f"{self.base_url}/4wings/tile/heatmap/{z}/{x}/{y}"
        response = self._client.get(url, params=params, headers=self._headers)
        if response.status_code >= 400:
            raise GFWError(f"GFW tile {z}/{x}/{y} failed ({response.status_code})")
        return response.content

    # -- vessels ------------------------------------------------------------

    def search_vessels(self, query: str, *, limit: int = 10) -> Any:
        params = [
            ("query", query),
            ("datasets[0]", VESSEL_IDENTITY),
            ("includes[0]", "MATCH_CRITERIA"),
            ("includes[1]", "OWNERSHIP"),
            ("limit", str(limit)),
        ]
        return self._request("GET", "/vessels/search", params=params)

    def get_vessel(self, vessel_id: str) -> Any:
        return self._request(
            "GET", f"/vessels/{vessel_id}", params=[("datasets[0]", VESSEL_IDENTITY)]
        )

    # -- events -------------------------------------------------------------

    def events(
        self,
        *,
        types: Iterable[str] = ("ENCOUNTER", "LOITERING", "GAP"),
        start: date | datetime | None = None,
        end: date | datetime | None = None,
        region_id: int | None = BANGLADESH_EEZ_ID,
        region_dataset: str = EEZ_REGION_DATASET,
        limit: int = 100,
        offset: int = 0,
    ) -> Any:
        """Fetch encounter / loitering / gap / port-visit events."""
        types = list(types)
        unknown = [t for t in types if t not in EVENT_DATASETS]
        if unknown:
            raise GFWError(f"unknown event types {unknown}; expected {sorted(EVENT_DATASETS)}")

        body: dict[str, Any] = {
            "datasets": [EVENT_DATASETS[t] for t in types],
        }
        if start and end:
            start_d = start.date() if isinstance(start, datetime) else start
            end_d = end.date() if isinstance(end, datetime) else end
            body["startDate"] = start_d.isoformat()
            body["endDate"] = end_d.isoformat()
        if region_id is not None:
            body["region"] = {"dataset": region_dataset, "id": region_id}
        params = [("limit", str(limit)), ("offset", str(offset))]
        return self._request("POST", "/events", params=params, json=body)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> GFWClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
