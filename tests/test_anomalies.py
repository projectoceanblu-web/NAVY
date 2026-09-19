"""AIS behavioural heuristics.

Each test pairs a positive case with the innocent case it must not flag —
a heuristic that fires on everything is worse than no heuristic at all.
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from bob_sentinel.fusion.anomalies import (
    ENCOUNTER,
    GAP,
    IDENTITY,
    LOITERING,
    SPEED_JUMP,
    _valid_imo,
    analyse_tracks,
    check_identity,
    detect_encounters,
    detect_gaps,
    detect_loitering,
    detect_speed_jumps,
)
from bob_sentinel.fusion.interpolate import Fix


def well_reported(t0, count=20, minutes=20):
    return [
        Fix(t0 + timedelta(minutes=minutes * i), 90.0 + 0.001 * i, 21.0, 6.0, 90.0)
        for i in range(count)
    ]


# --- gaps ------------------------------------------------------------------


def test_gap_after_good_reception_is_flagged(t0) -> None:
    fixes = well_reported(t0)
    fixes.append(Fix(fixes[-1].ts + timedelta(hours=8), 91.5, 21.4, 6.0, 90.0))
    events = detect_gaps(1, fixes)
    assert len(events) == 1
    assert events[0].kind == GAP
    assert events[0].details["duration_hours"] == pytest.approx(8.0)
    assert events[0].details["resumed_km_away"] > 100


def test_gap_on_a_poorly_heard_vessel_is_not_flagged(t0) -> None:
    """Otherwise every patchy receiver becomes an accusation.

    This is the GFW rule: a silence is only meaningful if the vessel was being
    heard reliably beforehand.
    """
    sparse = [Fix(t0 + timedelta(hours=2 * i), 90.0, 21.0, 6.0, 90.0) for i in range(3)]
    sparse.append(Fix(sparse[-1].ts + timedelta(hours=10), 91.0, 21.0, 6.0, 90.0))
    assert detect_gaps(2, sparse) == []


def test_short_silence_is_not_a_gap(t0) -> None:
    fixes = well_reported(t0)
    fixes.append(Fix(fixes[-1].ts + timedelta(minutes=45), 90.1, 21.0, 6.0, 90.0))
    assert detect_gaps(3, fixes) == []


# --- speed jumps -----------------------------------------------------------


def test_impossible_speed_is_flagged(t0) -> None:
    fixes = [Fix(t0, 90.0, 21.0, 5.0, 90.0), Fix(t0 + timedelta(minutes=5), 92.0, 21.0, 5.0, 90.0)]
    events = detect_speed_jumps(1, fixes)
    assert len(events) == 1
    assert events[0].kind == SPEED_JUMP
    assert events[0].details["implied_sog_kn"] > 1000


def test_a_fast_but_plausible_vessel_is_not_flagged(t0) -> None:
    """A 30 kn patrol boat is fast, not impossible."""
    fixes = [Fix(t0, 90.0, 21.0, 30.0, 90.0), Fix(t0 + timedelta(hours=1), 90.53, 21.0, 30.0, 90.0)]
    assert detect_speed_jumps(1, fixes) == []


# --- loitering -------------------------------------------------------------


def test_sustained_drifting_is_loitering(t0) -> None:
    fixes = [
        Fix(t0 + timedelta(minutes=15 * i), 90.0 + 0.0001 * i, 21.0, 0.4, 0.0)
        for i in range(24)
    ]
    events = detect_loitering(1, fixes)
    assert len(events) == 1
    assert events[0].kind == LOITERING
    assert events[0].details["duration_hours"] > 3


def test_a_transiting_vessel_is_not_loitering(t0) -> None:
    fixes = [
        Fix(t0 + timedelta(minutes=15 * i), 90.0 + 0.05 * i, 21.0, 12.0, 90.0)
        for i in range(24)
    ]
    assert detect_loitering(1, fixes) == []


def test_a_brief_stop_is_not_loitering(t0) -> None:
    fixes = [Fix(t0 + timedelta(minutes=15 * i), 90.0, 21.0, 0.3, 0.0) for i in range(5)]
    assert detect_loitering(1, fixes) == []


# --- encounters ------------------------------------------------------------


def test_two_slow_close_vessels_are_an_encounter(t0) -> None:
    a = [Fix(t0 + timedelta(minutes=15 * i), 90.0, 21.0, 0.3, 0.0) for i in range(16)]
    b = [Fix(t0 + timedelta(minutes=15 * i), 90.002, 21.0, 0.3, 0.0) for i in range(16)]
    events = detect_encounters({4: a, 5: b})
    assert len(events) == 1
    assert events[0].kind == ENCOUNTER
    assert {events[0].mmsi, events[0].counterpart_mmsi} == {4, 5}
    assert events[0].details["median_separation_km"] < 0.5


def test_vessels_merely_passing_are_not_an_encounter(t0) -> None:
    """Close approach at speed is traffic, not transhipment."""
    a = [
        Fix(t0 + timedelta(minutes=15 * i), 90.0 + 0.05 * i, 21.0, 12.0, 90.0)
        for i in range(16)
    ]
    b = [
        Fix(t0 + timedelta(minutes=15 * i), 90.0 + 0.05 * i, 21.001, 12.0, 90.0)
        for i in range(16)
    ]
    assert detect_encounters({1: a, 2: b}) == []


def test_slow_but_distant_vessels_are_not_an_encounter(t0) -> None:
    a = [Fix(t0 + timedelta(minutes=15 * i), 90.0, 21.0, 0.3, 0.0) for i in range(16)]
    b = [Fix(t0 + timedelta(minutes=15 * i), 90.5, 21.0, 0.3, 0.0) for i in range(16)]
    assert detect_encounters({1: a, 2: b}) == []


# --- identity --------------------------------------------------------------


@pytest.mark.parametrize("mmsi", [12345, 999000123, 1])
def test_malformed_mmsi_is_flagged(mmsi: int) -> None:
    events = check_identity(mmsi)
    assert len(events) == 1
    assert events[0].kind == IDENTITY


def test_valid_mmsi_and_imo_pass(t0) -> None:
    assert check_identity(405000123, imo=9074729) == []


def test_imo_check_digit() -> None:
    assert _valid_imo(9074729) is True
    assert _valid_imo(9074720) is False
    assert _valid_imo(123) is False


# --- orchestration ---------------------------------------------------------


def test_analyse_tracks_runs_every_heuristic(t0) -> None:
    gapped = well_reported(t0)
    gapped.append(Fix(gapped[-1].ts + timedelta(hours=8), 91.5, 21.4, 6.0, 90.0))
    loitering = [
        Fix(t0 + timedelta(minutes=15 * i), 90.5 + 0.0001 * i, 21.0, 0.4, 0.0)
        for i in range(24)
    ]
    events = analyse_tracks({1: gapped, 2: loitering})
    assert {e.kind for e in events} >= {GAP, LOITERING}
    # Events come back in chronological order for an analyst timeline.
    assert [e.start_ts for e in events] == sorted(e.start_ts for e in events)


def test_single_fix_tracks_are_skipped(t0) -> None:
    assert analyse_tracks({1: [Fix(t0, 90.0, 21.0)]}) == []
