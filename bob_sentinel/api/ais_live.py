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
) -> dict:
    """Drain the live AIS feed for `seconds`, persist it, and report the yield."""
    if not settings.aisstream_api_key:
        raise HTTPException(503, "AISSTREAM_API_KEY is not configured")

    started = datetime.now(UTC)
    try:
        records = asyncio.run(
            _drain(settings.aisstream_api_key, settings.aisstream_bbox(),
                   seconds, max_records)
        )
    except Exception as exc:  # noqa: BLE001 — surface upstream trouble as 502
        log.exception("AIS collection failed")
        raise HTTPException(502, f"aisstream: {exc}") from exc

    written, vessels = upsert_ais_records(db, records)
    db.commit()
    return {
        "collected": len(records),
        "positions_written": written,
        "vessels_seen": vessels,
        "window_seconds": seconds,
        "started": started.isoformat(),
        "source": "aisstream.io (live)",
    }


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
