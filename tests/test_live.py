"""Parsing of live Global Fishing Watch payloads.

The envelope shapes here are copied from real v3 responses observed against
the deployed service. They exist because the first deployment silently served
empty layers: GFW nests detection rows one level deeper than an ``entries``
list suggests, and a parser that treats the container as a row produces zero
features without raising anything.
"""

from __future__ import annotations

import pytest

from bob_sentinel.api.live import (
    _event_feature,
    _expand,
    _flatten_events,
    _flatten_report,
    _grid_feature,
    _text,
)

VALUE_KEYS = ("detections", "value", "hours", "count")

#: Verbatim shape of a live /v3/4wings/report response for the SAR layer.
REAL_SAR_REPORT = {
    "entries": [
        {
            "public-global-sar-presence:v4.0": [
                {
                    "lat": 21.16, "lon": 90.71, "detections": 1,
                    "date": "2026-08-15,2026-09-14",
                    "shipName": "", "mmsi": "", "flag": "", "vesselType": "",
                    "geartype": "", "callsign": "", "imo": "", "vesselId": "",
                    "entryTimestamp": "2026-08-15T23:47:34Z",
                    "exitTimestamp": "2026-09-13T23:56:20Z",
                },
                {
                    "lat": 21.57, "lon": 89.74, "detections": 3,
                    "date": "2026-08-15,2026-09-14",
                    "shipName": "FV EXAMPLE", "mmsi": "405000123",
                    "flag": "BGD", "vesselType": "fishing",
                    "geartype": "trawlers", "callsign": "S2AB",
                    "entryTimestamp": "", "exitTimestamp": "",
                },
            ]
        }
    ],
    "limit": None,
    "metadata": {},
    "nextOffset": None,
    "offset": 0,
    "total": 2,
}


def test_dataset_keyed_rows_are_flattened() -> None:
    """Regression: rows keyed by dataset id must not be read as a single row."""
    rows = _flatten_report(REAL_SAR_REPORT)
    assert len(rows) == 2
    assert {r["lat"] for r in rows} == {21.16, 21.57}


def test_every_flattened_row_becomes_a_feature() -> None:
    features = [
        f for f in (_grid_feature(r, VALUE_KEYS) for r in _flatten_report(REAL_SAR_REPORT)) if f
    ]
    assert len(features) == 2
    assert features[0].geometry["coordinates"] == [90.71, 21.16], "GeoJSON is lon,lat"


def test_blank_identity_marks_a_detection_unidentified() -> None:
    """GFW sends empty strings, not nulls, for detections it could not match."""
    dark, matched = (
        _grid_feature(r, VALUE_KEYS) for r in _flatten_report(REAL_SAR_REPORT)
    )
    assert dark.properties["identified"] is False
    assert dark.properties["vessel_name"] is None
    assert dark.properties["mmsi"] is None

    assert matched.properties["identified"] is True
    assert matched.properties["vessel_name"] == "FV EXAMPLE"
    assert matched.properties["flag"] == "BGD"


def test_detection_count_is_read_as_the_value() -> None:
    features = [_grid_feature(r, VALUE_KEYS) for r in _flatten_report(REAL_SAR_REPORT)]
    assert [f.properties["value"] for f in features] == [1.0, 3.0]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"entries": []},
        {"entries": [{}]},
        [],
        None,
        {"entries": [{"some-dataset:v1": []}]},
    ],
)
def test_empty_payloads_yield_no_rows(payload) -> None:
    assert _flatten_report(payload) == []


def test_bare_rows_without_a_position_are_rejected() -> None:
    """A metadata dict must not be mistaken for a detection."""
    assert _expand({"limit": 100, "offset": 0}) == []
    assert _expand({"lat": 21.0, "lon": 90.0}) == [{"lat": 21.0, "lon": 90.0}]


def test_rows_without_coordinates_produce_no_feature() -> None:
    assert _grid_feature({"detections": 4}, VALUE_KEYS) is None


def test_text_normalises_gfw_blanks() -> None:
    assert _text("") is None
    assert _text("   ") is None
    assert _text("BGD") == "BGD"
    assert _text(None) is None


# --- events ----------------------------------------------------------------


REAL_EVENTS = {
    "entries": [
        {
            "id": "evt-1",
            "type": "ENCOUNTER",
            "start": "2026-09-01T04:00:00Z",
            "end": "2026-09-01T09:30:00Z",
            "position": {"lat": 21.3, "lon": 90.5},
            "vessel": {"name": "FV EXAMPLE", "flag": "BGD", "ssvid": "405000123"},
        }
    ]
}


def test_events_are_flattened_and_geolocated() -> None:
    rows = _flatten_events(REAL_EVENTS)
    assert len(rows) == 1
    feature = _event_feature(rows[0])
    assert feature is not None
    assert feature.geometry["coordinates"] == [90.5, 21.3]
    assert feature.properties["kind"] == "encounter"
    assert feature.properties["vessel_name"] == "FV EXAMPLE"
    assert feature.properties["mmsi"] == "405000123"


def test_event_without_a_position_is_dropped() -> None:
    assert _event_feature({"id": "x", "type": "LOITERING"}) is None


def test_event_position_can_be_top_level() -> None:
    feature = _event_feature({"type": "GAP", "lat": 20.9, "lon": 91.1})
    assert feature is not None
    assert feature.geometry["coordinates"] == [91.1, 20.9]
