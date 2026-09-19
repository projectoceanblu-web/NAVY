"""Live AIS collection inside a serverless request.

aisstream.io is a push feed, and a serverless platform has no long-lived
process to hold the socket open. This endpoint bridges the two: it opens the
WebSocket, drains it for a bounded number of seconds, writes what arrived to
PostGIS, and returns.

Each call therefore adds a fresh slice of **real** vessel positions. Calling it
repeatedly (the map does, on load and on refresh) accumulates genuine tracks
over time rather than replaying anything canned.

This is a deliberate trade-off, not the recommended production design: a
long-running `python -m bob_sentinel.workers.ais_ingest` against the same
database gives continuous coverage. This exists so the hosted map is live
without requiring a second always-on host.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from bob_sentinel.config import Settings, get_settings
from bob_sentinel.db import get_db
from bob_sentinel.ingest.aisstream import AISRecord, AISStreamClient
from bob_sentinel.repository import upsert_ais_records

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api/ais", tags=["ais"])

#: Leave headroom inside the function's wall-clock budget for the DB write.
MAX_COLLECT_S = 20


@router.post("/collect")
def collect_live_ais(
    db: Session = Depends(get_db),
    settings: Settings = Depends(get_settings),
    seconds: int = Query(8, ge=1, le=MAX_COLLECT_S),
    max_records: int = Query(1500, ge=1, le=20000),
    lat_min: float | None = Query(None, ge=-90, le=90),
    lat_max: float | None = Query(None, ge=-90, le=90),
    lon_min: float | None = Query(None, ge=-180, le=180),
    lon_max: float | None = Query(None, ge=-180, le=180),
) -> dict:
    """Drain the live AIS feed for `seconds`, persist it, and report the yield.

    The bounding box defaults to the configured AOI; the query parameters
    override it so coverage elsewhere can be checked without redeploying.
    """
    if not settings.aisstream_api_key:
        raise HTTPException(503, "AISSTREAM_API_KEY is not configured")

    override = [lat_min, lat_max, lon_min, lon_max]
    if any(v is not None for v in override):
        if any(v is None for v in override):
            raise HTTPException(422, "supply all four of lat_min/lat_max/lon_min/lon_max")
        bboxes = [[[lat_min, lon_min], [lat_max, lon_max]]]
    else:
        bboxes = settings.aisstream_bbox()

    started = datetime.now(UTC)
    try:
        records = asyncio.run(
            _drain(settings.aisstream_api_key, bboxes, seconds, max_records)
        )
    except Exception as exc:  # noqa: BLE001 — surface upstream trouble as 502
        log.exception("AIS collection failed")
        raise HTTPException(502, f"aisstream: {exc}") from exc

    written, vessels = upsert_ais_records(db, records)
    db.commit()

    result = {
        "collected": len(records),
        "positions_written": written,
        "vessels_seen": vessels,
        "window_seconds": seconds,
        "bbox": bboxes[0],
        "started": started.isoformat(),
        "source": "aisstream.io (live)",
    }
    if not records:
        # An empty drain is not an error and not an absence of vessels. The
        # feed is crowd-contributed from shore stations, and as of 2026-09 it
        # has no contributing receiver anywhere in the Bay of Bengal: a global
        # subscription yields ~1700 messages in 15 s while 5-25N / 78-100E
        # yields zero. Saying so is the honest answer; silently drawing an
        # empty vessel layer would imply empty water.
        result["coverage"] = (
            "No AIS messages in this bounding box during the sample window. "
            "aisstream is crowd-contributed from shore receivers and has no "
            "known coverage in the Bay of Bengal; the nearest contributing "
            "stations are around the Malacca Strait. This is a gap in the "
            "feed, not an absence of vessels — which is exactly why an "
            "unmatched SAR detection here is reported as 'indeterminate' "
            "rather than 'dark'."
        )
    return result


async def _drain(api_key: str, bboxes, seconds: int, max_records: int) -> list[AISRecord]:
    """Collect records until the time budget or the record cap is reached."""
    client = AISStreamClient(api_key, bboxes)
    collected: list[AISRecord] = []

    def on_record(record: AISRecord) -> None:
        collected.append(record)
        if len(collected) >= max_records:
            client.stop()

    async def run() -> None:
        try:
            await asyncio.wait_for(client.run(on_record), timeout=seconds)
        except TimeoutError:
            client.stop()

    await run()
    log.info("collected %d AIS records in %ss", len(collected), seconds)
    return collected
