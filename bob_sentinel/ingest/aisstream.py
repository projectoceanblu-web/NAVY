"""aisstream.io WebSocket ingester.

Protocol notes that cost time if you learn them the hard way:

* The subscription JSON must be sent **within 3 seconds** of the socket
  opening, or the server drops the connection.
* Bounding boxes are ``[[lat, lon], [lat, lon]]`` — latitude first, the
  opposite of GeoJSON order.
* Frames arrive as **binary** WebSocket messages whose payload is UTF-8 JSON,
  so decode bytes before parsing.
* Re-sending a subscription *replaces* the previous one; it does not merge.
* The service is community-run with no SLA, so reconnection with backoff is
  mandatory rather than optional.

Parsing is kept free of I/O (:func:`parse_message`) so it can be tested
against recorded frames without a socket or a database.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

log = logging.getLogger(__name__)

STREAM_URL = "wss://stream.aisstream.io/v0/stream"

#: The server closes the socket if the subscription is not sent in time.
SUBSCRIBE_DEADLINE_S = 3.0

DEFAULT_MESSAGE_TYPES = ("PositionReport", "ShipStaticData")

#: AIS sentinel values that mean "not available" rather than a real reading.
_SOG_UNAVAILABLE = 102.3
_COG_UNAVAILABLE = 360.0
_HEADING_UNAVAILABLE = 511


class AISStreamError(RuntimeError):
    pass


@dataclass
class AISRecord:
    """A normalised AIS observation (position and/or static data)."""

    mmsi: int
    ts: datetime
    lon: float | None = None
    lat: float | None = None
    sog_kn: float | None = None
    cog_deg: float | None = None
    heading_deg: float | None = None
    nav_status: int | None = None
    rot: float | None = None
    name: str | None = None
    imo: int | None = None
    call_sign: str | None = None
    ship_type: int | None = None
    length_m: float | None = None
    width_m: float | None = None
    destination: str | None = None
    kind: str = "position"

    @property
    def has_position(self) -> bool:
        return self.lon is not None and self.lat is not None


def build_subscription(
    api_key: str,
    bboxes: Sequence[Sequence[Sequence[float]]],
    *,
    mmsi_filter: Iterable[str | int] | None = None,
    message_types: Sequence[str] | None = DEFAULT_MESSAGE_TYPES,
) -> dict[str, Any]:
    """Build the subscription payload.

    Keys are case-sensitive on the server (``APIKey``, not ``apiKey``) — a
    common cause of the "subscription accepted but no data" symptom.
    """
    if not api_key:
        raise AISStreamError("AISSTREAM_API_KEY is not set")
    if not bboxes:
        raise AISStreamError("at least one bounding box is required")

    payload: dict[str, Any] = {"APIKey": api_key, "BoundingBoxes": [
        [list(map(float, corner)) for corner in bbox] for bbox in bboxes
    ]}
    if mmsi_filter:
        mmsis = [str(m) for m in mmsi_filter]
        if len(mmsis) > 50:
            raise AISStreamError(
                f"FiltersShipMMSI accepts at most 50 vessels, got {len(mmsis)}"
            )
        payload["FiltersShipMMSI"] = mmsis
    if message_types:
        payload["FilterMessageTypes"] = list(message_types)
    return payload


def _parse_time(value: str | None) -> datetime:
    """aisstream stamps look like '2026-01-03 11:42:17.123456789 +0000 UTC'."""
    if not value:
        return datetime.now(UTC)
    cleaned = value.replace(" UTC", "").strip()
    try:
        head, _, tail = cleaned.rpartition(" ")
        if head and (tail.startswith("+") or tail.startswith("-")) and len(tail) == 5:
            # Trim fractional seconds to microseconds for fromisoformat.
            if "." in head:
                base, _, frac = head.partition(".")
                head = f"{base}.{frac[:6]}"
            return datetime.fromisoformat(f"{head}{tail[:3]}:{tail[3:]}")
        return datetime.fromisoformat(cleaned).replace(tzinfo=UTC)
    except ValueError:
        log.debug("unparseable AIS timestamp %r; using receipt time", value)
        return datetime.now(UTC)


def _clean(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip().strip("@").strip()
        return stripped or None
    return value


def parse_message(raw: str | bytes | dict) -> AISRecord | None:
    """Normalise one aisstream envelope into an :class:`AISRecord`.

    Returns ``None`` for frames that carry no usable content (unknown message
    types, keepalives, malformed JSON) rather than raising — a single bad frame
    must never take down a long-running ingester.
    """
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        envelope = json.loads(raw) if isinstance(raw, str) else raw
    except (UnicodeDecodeError, json.JSONDecodeError):
        log.debug("dropping undecodable AIS frame")
        return None
    if not isinstance(envelope, dict):
        return None

    meta = envelope.get("MetaData") or {}
    message_type = envelope.get("MessageType")
    body = (envelope.get("Message") or {}).get(message_type, {}) or {}

    mmsi = meta.get("MMSI") or meta.get("MMSI_String") or body.get("UserID")
    try:
        mmsi = int(mmsi)
    except (TypeError, ValueError):
        return None

    record = AISRecord(mmsi=mmsi, ts=_parse_time(meta.get("time_utc")))

    if message_type == "PositionReport":
        record.kind = "position"
        record.lat = _coerce_float(body.get("Latitude", meta.get("latitude")))
        record.lon = _coerce_float(body.get("Longitude", meta.get("longitude")))
        sog = _coerce_float(body.get("Sog"))
        cog = _coerce_float(body.get("Cog"))
        heading = body.get("TrueHeading")
        record.sog_kn = None if sog is None or sog >= _SOG_UNAVAILABLE else sog
        record.cog_deg = None if cog is None or cog >= _COG_UNAVAILABLE else cog
        record.heading_deg = (
            None if heading is None or int(heading) >= _HEADING_UNAVAILABLE else float(heading)
        )
        record.nav_status = _coerce_int(body.get("NavigationalStatus"))
        record.rot = _coerce_float(body.get("RateOfTurn"))
    elif message_type == "ShipStaticData":
        record.kind = "static"
        imo_field = body.get("ImoNumber")
        # Some feeds nest the IMO as {"IMO": 9074729}; most send a bare integer.
        if isinstance(imo_field, dict):
            imo_field = imo_field.get("IMO")
        record.imo = _coerce_int(imo_field)
        record.call_sign = _clean(body.get("CallSign"))
        record.name = _clean(body.get("Name") or meta.get("ShipName"))
        record.ship_type = _coerce_int(body.get("Type"))
        record.destination = _clean(body.get("Destination"))
        dimension = body.get("Dimension") or {}
        bow, stern = _coerce_float(dimension.get("A")), _coerce_float(dimension.get("B"))
        port, starboard = _coerce_float(dimension.get("C")), _coerce_float(dimension.get("D"))
        if bow is not None and stern is not None:
            record.length_m = bow + stern
        if port is not None and starboard is not None:
            record.width_m = port + starboard
        record.lat = _coerce_float(meta.get("latitude"))
        record.lon = _coerce_float(meta.get("longitude"))
    else:
        # Unsubscribed or future message type — keep identity, drop the body.
        record.kind = "other"
        record.lat = _coerce_float(meta.get("latitude"))
        record.lon = _coerce_float(meta.get("longitude"))

    if record.name is None:
        record.name = _clean(meta.get("ShipName"))
    if record.lat is not None and not (-90.0 <= record.lat <= 90.0):
        record.lat = None
    if record.lon is not None and not (-180.0 <= record.lon <= 180.0):
        record.lon = None
    return record


def _coerce_float(value: Any) -> float | None:
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return result if result == result else None  # drop NaN


def _coerce_int(value: Any) -> int | None:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


class AISStreamClient:
    """Reconnecting consumer of the aisstream.io feed."""

    def __init__(
        self,
        api_key: str,
        bboxes: Sequence[Sequence[Sequence[float]]],
        *,
        message_types: Sequence[str] | None = DEFAULT_MESSAGE_TYPES,
        url: str = STREAM_URL,
        max_backoff_s: float = 60.0,
    ) -> None:
        self.subscription = build_subscription(
            api_key, bboxes, message_types=message_types
        )
        self.url = url
        self.max_backoff_s = max_backoff_s
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    async def run(
        self,
        on_record: Callable[[AISRecord], Any],
        *,
        connect_factory=None,
    ) -> None:
        """Consume the stream until :meth:`stop`, reconnecting with backoff.

        ``connect_factory`` is injectable so tests can drive the loop with a
        fake socket instead of a live connection.
        """
        if connect_factory is None:
            import websockets

            def connect_factory(url: str):
                # permessage-deflate is on by default in `websockets` and
                # materially reduces bandwidth on this feed.
                return websockets.connect(url, ping_interval=20, ping_timeout=20)

        attempt = 0
        while not self._stop.is_set():
            try:
                async with connect_factory(self.url) as socket:
                    await asyncio.wait_for(
                        socket.send(json.dumps(self.subscription)),
                        timeout=SUBSCRIBE_DEADLINE_S,
                    )
                    log.info("subscribed to aisstream at %s", self.url)
                    attempt = 0
                    async for frame in socket:
                        if self._stop.is_set():
                            break
                        record = parse_message(frame)
                        if record is not None:
                            result = on_record(record)
                            if asyncio.iscoroutine(result):
                                await result
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - keep the ingester alive
                if self._stop.is_set():
                    break
                attempt += 1
                delay = min(2.0**attempt, self.max_backoff_s)
                delay *= 0.5 + random.random()  # jitter: avoid a thundering herd
                log.warning(
                    "aisstream connection failed (%s); reconnecting in %.1fs",
                    exc,
                    delay,
                )
                try:
                    await asyncio.wait_for(self._stop.wait(), timeout=delay)
                except TimeoutError:
                    pass
        log.info("aisstream ingester stopped")
