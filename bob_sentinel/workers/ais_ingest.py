"""AIS ingestion worker: aisstream.io -> PostGIS.

Records are buffered and flushed in batches — one INSERT per position would
put the database under thousands of round-trips a minute for no benefit.  The
flush runs in a worker thread so the sync DB driver never blocks the event loop
that is reading the socket.

Run with:  python -m bob_sentinel.workers.ais_ingest
"""

from __future__ import annotations

import asyncio
import logging
import signal
from datetime import UTC, datetime

from bob_sentinel.config import get_settings
from bob_sentinel.db import session_scope
from bob_sentinel.ingest.aisstream import AISRecord, AISStreamClient
from bob_sentinel.main import configure_logging
from bob_sentinel.repository import upsert_ais_records

log = logging.getLogger("bob_sentinel.ais_ingest")

BATCH_SIZE = 500
FLUSH_INTERVAL_S = 10.0


class BatchWriter:
    """Buffers AIS records and flushes on size or age, whichever comes first."""

    def __init__(self, batch_size: int = BATCH_SIZE, interval_s: float = FLUSH_INTERVAL_S):
        self.batch_size = batch_size
        self.interval_s = interval_s
        self._buffer: list[AISRecord] = []
        self._lock = asyncio.Lock()
        self._last_flush = datetime.now(UTC)
        self.total_positions = 0
        self.total_records = 0

    async def add(self, record: AISRecord) -> None:
        async with self._lock:
            self._buffer.append(record)
            self.total_records += 1
            due = (
                len(self._buffer) >= self.batch_size
                or (datetime.now(UTC) - self._last_flush).total_seconds()
                >= self.interval_s
            )
        if due:
            await self.flush()

    async def flush(self) -> None:
        async with self._lock:
            batch, self._buffer = self._buffer, []
            self._last_flush = datetime.now(UTC)
        if not batch:
            return
        try:
            written, vessels = await asyncio.to_thread(self._write, batch)
        except Exception as exc:  # noqa: BLE001 — a bad batch must not kill ingest
            log.exception("failed to flush %d AIS records: %s", len(batch), exc)
            return
        self.total_positions += written
        log.info(
            "flushed %d records (%d positions, %d vessels); session total %d positions",
            len(batch), written, vessels, self.total_positions,
        )

    @staticmethod
    def _write(batch: list[AISRecord]) -> tuple[int, int]:
        with session_scope() as session:
            return upsert_ais_records(session, batch)


async def periodic_flush(writer: BatchWriter, stop: asyncio.Event) -> None:
    """Flush on a timer so a quiet feed still lands its buffered records."""
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), timeout=writer.interval_s)
        except TimeoutError:
            await writer.flush()
    await writer.flush()


async def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    if not settings.aisstream_api_key:
        raise SystemExit("AISSTREAM_API_KEY is not set — see .env.example")

    writer = BatchWriter()
    stop = asyncio.Event()
    client = AISStreamClient(settings.aisstream_api_key, settings.aisstream_bbox())

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda: (stop.set(), client.stop()))
        except NotImplementedError:  # pragma: no cover - Windows
            pass

    log.info(
        "ingesting AIS for %s bbox lat[%.2f, %.2f] lon[%.2f, %.2f]",
        settings.aoi_name, settings.aoi_lat_min, settings.aoi_lat_max,
        settings.aoi_lon_min, settings.aoi_lon_max,
    )
    flusher = asyncio.create_task(periodic_flush(writer, stop))
    try:
        await client.run(writer.add)
    finally:
        stop.set()
        await flusher
        log.info("ingest finished: %d records seen", writer.total_records)


if __name__ == "__main__":
    asyncio.run(main())
