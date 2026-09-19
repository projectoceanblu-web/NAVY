"""Live Global Fishing Watch endpoints.

These serve **real, current** data rather than anything stored locally: every
request is proxied to the GFW v3 API with the token held server-side, and the
response is normalised into GeoJSON the map can draw directly.

Why GFW is the live source for the hosted deployment:

* It is a REST API, so it works inside a serverless function. The aisstream
  feed is a persistent WebSocket and the Sentinel-1 pipeline needs GDAL plus
  minutes of CPU per scene, so neither can run on Vercel — they populate the
  database from a long-running host instead.
* Its SAR layer is Sentinel-1 derived and carries an AIS-match flag, so
  ``matched='false'`` is a real dark-detection view of the same water this
  project targets.

Responses are cached briefly in-process: GFW rate-limits per user across all
tokens, and a map that polls every minute must not spend that budget.
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, date, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query

from bob_sentinel.clients.gfw import (
    AIS_PRESENCE,
    EVENT_DATASETS,
    FISHING_EFFORT,
    SAR_PRESENCE,
    GFWClient,
    GFWError,
)
from bob_sentinel.config import Settings, get_settings
from bob_sentinel.schemas import GeoJSONFeature, GeoJSONFeatureCollection

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/live", tags=["live"])

#: GFW's SAR layer lags real time by roughly 5 days; asking for "today" returns
#: an empty grid and looks like a bug rather than a latency characteristic.
SAR_LAG_DAYS = 5

_CACHE: dict[str, tuple[float, Any]] = {}
_CACHE_TTL_S = 300.0


def _cached(key: str, producer):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and now - hit[0] < _CACHE_TTL_S:
        return hit[1]
    value = producer()
    _CACHE[key] = (now, value)
    return value


def _client(settings: Settings) -> GFWClient:
    if not settings.gfw_api_token:
        raise HTTPException(
            503,
            "GFW_API_TOKEN is not configured — the live layers need a free "
            "Global Fishing Watch API token.",
        )
    return GFWClient(settings.gfw_api_token)


def _window(days: int, lag_days: int = 0) -> tuple[date, date]:
    end = datetime.now(UTC).date() - timedelta(days=lag_days)
    return end - timedelta(days=days), end


def _num(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if result == result else None


def _first(mapping: dict, *names: str) -> Any:
    """Pick the first present key — GFW spells fields differently per endpoint."""
    for name in names:
        if name in mapping and mapping[name] is not None:
            return mapping[name]
    return None


# --- events ---------------------------------------------------------------


def _flatten_events(payload: Any) -> list[dict]:
    if isinstance(payload, dict):
        for key in ("entries", "events", "data", "results"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
    return payload if isinstance(payload, list) else []


def _event_feature(event: dict) -> GeoJSONFeature | None:
    position = event.get("position") or event.get("coordinates") or {}
    lat = _num(_first(position, "lat", "latitude")) if isinstance(position, dict) else None
    lon = _num(_first(position, "lon", "lng", "longitude")) if isinstance(position, dict) else None
    if lat is None or lon is None:
        lat = _num(_first(event, "lat", "latitude"))
        lon = _num(_first(event, "lon", "lng", "longitude"))
    if lat is None or lon is None:
        return None

    vessel = event.get("vessel") or {}
    return GeoJSONFeature(
        geometry={"type": "Point", "coordinates": [lon, lat]},
        properties={
            "id": event.get("id"),
            "kind": (event.get("type") or "").lower() or "event",
            "start": _first(event, "start", "startDate", "timestamp"),
            "end": _first(event, "end", "endDate"),
            "vessel_name": vessel.get("name") if isinstance(vessel, dict) else None,
            "vessel_flag": vessel.get("flag") if isinstance(vessel, dict) else None,
            "mmsi": vessel.get("ssvid") if isinstance(vessel, dict) else None,
            "source": "Global Fishing Watch",
        },
    )


@router.get("/events", response_model=GeoJSONFeatureCollection)
def live_events(
    settings: Settings = Depends(get_settings),
    days: int = Query(30, ge=1, le=366),
    types: str = Query("ENCOUNTER,LOITERING,FISHING", description="Comma-separated"),
    limit: int = Query(300, ge=1, le=1000),
) -> GeoJSONFeatureCollection:
    """Real GFW vessel events inside the Bangladesh EEZ.

    Encounters and loitering are the transhipment signatures; fishing events
    are GFW's own apparent-fishing inferences.
    """
    wanted = [t.strip().upper() for t in types.split(",") if t.strip()]
    unknown = [t for t in wanted if t not in EVENT_DATASETS]
    if unknown:
        raise HTTPException(422, f"unknown event types {unknown}")

    start, end = _window(days)
    key = f"events:{','.join(sorted(wanted))}:{days}:{limit}"

    def fetch():
        with _client(settings) as client:
            return client.events(
                types=wanted, start=start, end=end,
                region_id=settings.aoi_mrgid, limit=limit,
            )

    try:
        payload = _cached(key, fetch)
    except GFWError as exc:
        raise HTTPException(502, f"Global Fishing Watch: {exc}") from exc

    features = [f for f in (_event_feature(e) for e in _flatten_events(payload)) if f]
    log.info("live events: %d features for %s", len(features), wanted)
    return GeoJSONFeatureCollection(features=features)


# --- gridded activity ------------------------------------------------------


def _expand(item: Any) -> list[dict]:
    """Expand one report entry into detection rows.

    The live v3 shape keys the rows by dataset id, one level deeper than an
    entry list suggests::

        {"entries": [{"public-global-sar-presence:v4.0": [{lat, lon, ...}]}]}

    So an entry that is a dict of lists is a container, not a row — treating
    it as a row is how this silently produced an empty layer.
    """
    if not isinstance(item, dict):
        return []
    if isinstance(item.get("entries"), list):
        return [x for x in item["entries"] if isinstance(x, dict)]

    nested: list[dict] = []
    for value in item.values():
        if isinstance(value, list):
            nested.extend(x for x in value if isinstance(x, dict))
    if nested:
        return nested

    # A bare row: only accept it if it actually carries a position.
    return [item] if ("lat" in item or "latitude" in item) else []


def _flatten_report(payload: Any) -> list[dict]:
    """Normalise a GFW report payload into a flat list of detection rows."""
    if isinstance(payload, list):
        return [row for item in payload for row in _expand(item)]
    if isinstance(payload, dict):
        for key in ("entries", "data", "result"):
            value = payload.get(key)
            if isinstance(value, list):
                return [row for item in value for row in _expand(item)]
            if isinstance(value, dict):
                rows = _expand(value)
                if rows:
                    return rows
    return []


def _text(value: Any) -> str | None:
    """GFW returns empty strings rather than nulls for unknown identity."""
    if isinstance(value, str):
        return value.strip() or None
    return value if value is not None else None


def _grid_feature(cell: dict, value_keys: tuple[str, ...]) -> GeoJSONFeature | None:
    lat = _num(_first(cell, "lat", "latitude", "cell_lat"))
    lon = _num(_first(cell, "lon", "lng", "longitude", "cell_lon"))
    if lat is None or lon is None:
        return None

    name = _text(cell.get("shipName"))
    mmsi = _text(cell.get("mmsi"))
    return GeoJSONFeature(
        geometry={"type": "Point", "coordinates": [lon, lat]},
        properties={
            "value": _num(_first(cell, *value_keys)),
            "date": _first(cell, "date", "timestamp", "time"),
            # Identity fields come back blank for detections GFW could not
            # reconcile with an AIS broadcast — which is the whole point here.
            "vessel_name": name,
            "mmsi": mmsi,
            "flag": _text(cell.get("flag")),
            "vessel_type": _text(cell.get("vesselType")),
            "geartype": _text(cell.get("geartype")),
            "identified": bool(name or mmsi),
            "first_seen": _text(cell.get("entryTimestamp")),
            "last_seen": _text(cell.get("exitTimestamp")),
            "source": "Global Fishing Watch",
        },
    )


def _report_layer(
    settings: Settings, *, dataset: str, days: int, lag_days: int,
    dark_only: bool | None, value_keys: tuple[str, ...], cache_key: str,
) -> GeoJSONFeatureCollection:
    start, end = _window(days, lag_days=lag_days)

    def fetch():
        with _client(settings) as client:
            params: dict[str, Any] = {
                "start": start, "end": end,
                "region_id": settings.aoi_mrgid,
                "spatial_resolution": "HIGH",
                "temporal_resolution": "ENTIRE",
            }
            if dataset == SAR_PRESENCE:
                return client.sar_report(dark_only=bool(dark_only), **params)
            return client.presence_report(dataset=dataset, **params)

    try:
        payload = _cached(cache_key, fetch)
    except GFWError as exc:
        raise HTTPException(502, f"Global Fishing Watch: {exc}") from exc

    features = [
        f for f in (_grid_feature(c, value_keys) for c in _flatten_report(payload)) if f
    ]
    return GeoJSONFeatureCollection(features=features)


@router.get("/sar", response_model=GeoJSONFeatureCollection)
def live_sar(
    settings: Settings = Depends(get_settings),
    days: int = Query(30, ge=1, le=366),
    dark_only: bool = Query(True, description="Only detections GFW could not match to AIS"),
) -> GeoJSONFeatureCollection:
    """Real GFW Sentinel-1 SAR detections over the Bangladesh EEZ.

    ``dark_only`` applies ``matched='false'`` — vessels GFW detected in radar
    but could not reconcile with any AIS broadcast.
    """
    return _report_layer(
        settings, dataset=SAR_PRESENCE, days=days, lag_days=SAR_LAG_DAYS,
        dark_only=dark_only, value_keys=("detections", "value", "hours", "count"),
        cache_key=f"sar:{days}:{dark_only}",
    )


@router.get("/fishing", response_model=GeoJSONFeatureCollection)
def live_fishing(
    settings: Settings = Depends(get_settings),
    days: int = Query(30, ge=1, le=366),
) -> GeoJSONFeatureCollection:
    """Real GFW apparent fishing effort (AIS-derived) over the EEZ."""
    return _report_layer(
        settings, dataset=FISHING_EFFORT, days=days, lag_days=0, dark_only=None,
        value_keys=("hours", "value", "fishingHours", "count"),
        cache_key=f"fishing:{days}",
    )


@router.get("/presence", response_model=GeoJSONFeatureCollection)
def live_presence(
    settings: Settings = Depends(get_settings),
    days: int = Query(7, ge=1, le=366),
) -> GeoJSONFeatureCollection:
    """Real GFW AIS vessel presence over the EEZ."""
    return _report_layer(
        settings, dataset=AIS_PRESENCE, days=days, lag_days=0, dark_only=None,
        value_keys=("hours", "value", "count"),
        cache_key=f"presence:{days}",
    )


@router.get("/diagnostics")
def live_diagnostics(settings: Settings = Depends(get_settings)) -> dict:
    """What the upstream API actually returned, for troubleshooting a blank map.

    GFW's payload shapes vary between endpoints and versions; this reports the
    raw envelope keys and one sample row so a mismatch is obvious rather than
    silently rendering an empty layer.
    """
    out: dict[str, Any] = {
        "gfw_token_configured": bool(settings.gfw_api_token),
        "aoi_mrgid": settings.aoi_mrgid,
        "sar_lag_days": SAR_LAG_DAYS,
    }
    if not settings.gfw_api_token:
        return out

    start, end = _window(30, lag_days=SAR_LAG_DAYS)
    with _client(settings) as client:
        for name, call in (
            ("sar_report", lambda: client.sar_report(
                start=start, end=end, region_id=settings.aoi_mrgid,
                temporal_resolution="ENTIRE")),
            ("events", lambda: client.events(
                types=["ENCOUNTER", "LOITERING"], start=start, end=end,
                region_id=settings.aoi_mrgid, limit=5)),
        ):
            try:
                payload = call()
                rows = (_flatten_report(payload) if name == "sar_report"
                        else _flatten_events(payload))
                sample = rows[0] if rows else None
                if isinstance(sample, dict):
                    # Echo scalars only: a nested container here once made this
                    # response a megabyte, which defeats a diagnostic endpoint.
                    sample = {
                        k: v
                        for k, v in sample.items()
                        if not isinstance(v, (list, dict))
                    }
                out[name] = {
                    "ok": True,
                    "envelope_type": type(payload).__name__,
                    "envelope_keys": sorted(payload)[:12] if isinstance(payload, dict) else None,
                    "row_count": len(rows),
                    "sample_row_keys": sorted(rows[0])[:20] if rows else None,
                    "sample_row": sample,
                }
            except Exception as exc:  # noqa: BLE001 — diagnostics must not raise
                out[name] = {"ok": False, "error": str(exc)[:400]}
    return out
