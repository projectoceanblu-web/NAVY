"""aisstream.io framing, parsing and resilience."""

from __future__ import annotations

import asyncio
import json

import pytest

from bob_sentinel.ingest.aisstream import (
    AISStreamClient,
    AISStreamError,
    build_subscription,
    parse_message,
)


def position_frame(**overrides):
    body = {
        "Latitude": 21.5, "Longitude": 90.2, "Sog": 8.4, "Cog": 142.0,
        "TrueHeading": 140, "NavigationalStatus": 0, "RateOfTurn": -2,
    }
    body.update(overrides)
    return json.dumps(
        {
            "MessageType": "PositionReport",
            "MetaData": {
                "MMSI": 405000123, "ShipName": "TEST VESSEL@@@",
                "latitude": 21.5, "longitude": 90.2,
                "time_utc": "2026-01-03 11:42:17.123456789 +0000 UTC",
            },
            "Message": {"PositionReport": body},
        }
    )


# --- subscription ----------------------------------------------------------


def test_subscription_uses_the_servers_exact_key_casing() -> None:
    """Lowercase keys are silently accepted and then send no data."""
    payload = build_subscription("KEY", [[[20.5, 88.0], [22.8, 92.7]]])
    assert payload["APIKey"] == "KEY"
    assert "BoundingBoxes" in payload
    assert payload["FilterMessageTypes"] == ["PositionReport", "ShipStaticData"]


def test_bounding_box_is_lat_lon_not_lon_lat() -> None:
    """aisstream reverses the GeoJSON convention; getting this wrong = no data."""
    payload = build_subscription("KEY", [[[20.5, 88.0], [22.8, 92.7]]])
    (lat_min, lon_min), (lat_max, lon_max) = payload["BoundingBoxes"][0]
    assert (lat_min, lat_max) == (20.5, 22.8)
    assert (lon_min, lon_max) == (88.0, 92.7)


def test_missing_key_is_rejected_early() -> None:
    with pytest.raises(AISStreamError, match="AISSTREAM_API_KEY"):
        build_subscription("", [[[0, 0], [1, 1]]])


def test_empty_bbox_is_rejected() -> None:
    with pytest.raises(AISStreamError, match="bounding box"):
        build_subscription("KEY", [])


def test_mmsi_filter_cap_is_enforced() -> None:
    with pytest.raises(AISStreamError, match="at most 50"):
        build_subscription("KEY", [[[0, 0], [1, 1]]], mmsi_filter=range(51))


# --- parsing ---------------------------------------------------------------


def test_position_report_is_parsed() -> None:
    record = parse_message(position_frame())
    assert record is not None
    assert record.mmsi == 405000123
    assert (record.lon, record.lat) == (90.2, 21.5)
    assert record.sog_kn == 8.4
    assert record.name == "TEST VESSEL", "AIS pads names with '@'"
    assert record.ts.year == 2026 and record.ts.microsecond == 123456


def test_binary_frames_are_decoded() -> None:
    """The server sends binary frames carrying UTF-8 JSON."""
    assert parse_message(position_frame().encode("utf-8")) is not None


def test_unavailable_sentinels_become_none() -> None:
    """102.3 kn / 360 deg / 511 mean 'not available', not a real reading."""
    record = parse_message(position_frame(Sog=102.3, Cog=360.0, TrueHeading=511))
    assert record is not None
    assert record.sog_kn is None
    assert record.cog_deg is None
    assert record.heading_deg is None


def test_static_data_yields_dimensions() -> None:
    frame = json.dumps(
        {
            "MessageType": "ShipStaticData",
            "MetaData": {"MMSI": 405000123, "time_utc": "2026-01-03 11:42:17 +0000 UTC"},
            "Message": {
                "ShipStaticData": {
                    "Name": "FV EXAMPLE", "CallSign": "S2AB", "Type": 30,
                    "ImoNumber": 9074729, "Destination": "CHATTOGRAM",
                    "Dimension": {"A": 20, "B": 10, "C": 4, "D": 4},
                }
            },
        }
    )
    record = parse_message(frame)
    assert record is not None
    assert record.kind == "static"
    assert record.length_m == 30 and record.width_m == 8
    assert record.imo == 9074729


@pytest.mark.parametrize(
    "junk",
    [b"\xff\xfenot json", "{not json", "[]", "null", json.dumps({"MessageType": "X"})],
)
def test_malformed_frames_return_none_instead_of_raising(junk) -> None:
    """A single bad frame must never take down a long-running ingester."""
    assert parse_message(junk) is None


def test_out_of_range_coordinates_are_discarded() -> None:
    record = parse_message(position_frame(Latitude=91.0, Longitude=181.0))
    assert record is not None
    assert record.lat is None and record.lon is None
    assert record.has_position is False


def test_unparseable_timestamp_falls_back_to_receipt_time() -> None:
    frame = json.dumps(
        {
            "MessageType": "PositionReport",
            "MetaData": {"MMSI": 1, "time_utc": "not a timestamp"},
            "Message": {"PositionReport": {"Latitude": 21.0, "Longitude": 90.0}},
        }
    )
    record = parse_message(frame)
    assert record is not None and record.ts is not None


# --- connection handling ---------------------------------------------------


class FakeSocket:
    """Minimal async context manager mimicking a websockets connection."""

    def __init__(self, frames, fail_first=False):
        self.frames = frames
        self.fail_first = fail_first
        self.sent: list[str] = []

    async def __aenter__(self):
        if self.fail_first:
            self.fail_first = False
            raise ConnectionError("simulated drop")
        return self

    async def __aexit__(self, *exc):
        return False

    async def send(self, payload):
        self.sent.append(payload)

    def __aiter__(self):
        async def gen():
            for frame in self.frames:
                yield frame
        return gen()


@pytest.mark.asyncio
async def test_subscription_is_sent_before_any_frame_is_read() -> None:
    socket = FakeSocket([position_frame()])
    client = AISStreamClient("KEY", [[[20.5, 88.0], [22.8, 92.7]]])
    seen = []

    def on_record(record):
        seen.append(record)
        client.stop()

    await asyncio.wait_for(
        client.run(on_record, connect_factory=lambda url: socket), timeout=5
    )
    assert socket.sent, "no subscription was sent"
    assert json.loads(socket.sent[0])["APIKey"] == "KEY"
    assert len(seen) == 1


@pytest.mark.asyncio
async def test_connection_failure_triggers_a_reconnect() -> None:
    """aisstream has no SLA, so dropping out must be survivable."""
    socket = FakeSocket([position_frame()], fail_first=True)
    client = AISStreamClient("KEY", [[[20.5, 88.0], [22.8, 92.7]]], max_backoff_s=0.05)
    seen = []

    def on_record(record):
        seen.append(record)
        client.stop()

    await asyncio.wait_for(
        client.run(on_record, connect_factory=lambda url: socket), timeout=10
    )
    assert len(seen) == 1, "the client did not recover from a dropped connection"


@pytest.mark.asyncio
async def test_async_callbacks_are_awaited() -> None:
    socket = FakeSocket([position_frame(), position_frame()])
    client = AISStreamClient("KEY", [[[20.5, 88.0], [22.8, 92.7]]])
    seen = []

    async def on_record(record):
        await asyncio.sleep(0)
        seen.append(record)
        if len(seen) == 2:
            client.stop()

    await asyncio.wait_for(
        client.run(on_record, connect_factory=lambda url: socket), timeout=5
    )
    assert len(seen) == 2
