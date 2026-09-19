"""Matching and the dark-vessel verdict.

The tri-state verdict is the safety-critical part of this system: calling an
'indeterminate' detection 'dark' is how a screening tool turns into a false
accusation, so most of these tests are about that boundary.
"""

from __future__ import annotations

import pytest

from bob_sentinel.fusion.interpolate import InterpolatedPosition
from bob_sentinel.fusion.matcher import (
    DARK,
    INDETERMINATE,
    MATCHED,
    DetectionInput,
    coverage_is_adequate,
    match_detections,
    match_gate_m,
    summarise,
)


def position(mmsi, lon, lat, *, t0, gap=60.0, sog=5.0, bracketed=True, method="linear"):
    return InterpolatedPosition(
        mmsi=mmsi, ts=t0, lon=lon, lat=lat, method=method,
        gap_s=gap, bracketed=bracketed, sog_kn=sog, cog_deg=90.0,
    )


def test_detection_on_top_of_ais_is_matched(t0) -> None:
    results = match_detections(
        [DetectionInput(1, 90.0005, 21.0, 40.0)], [position(111, 90.0, 21.0, t0=t0)]
    )
    assert results[0].status == MATCHED
    assert results[0].matched_mmsi == 111
    assert results[0].is_dark is False
    assert results[0].distance_m < 100


def test_unmatched_detection_with_nearby_ais_is_dark(t0) -> None:
    """AIS was clearly being received here, so a silent target is notable."""
    positions = [position(m, 90.0 + 0.01 * i, 21.0, t0=t0) for i, m in enumerate([1, 2, 3])]
    results = match_detections([DetectionInput(9, 90.1, 21.0, 30.0)], positions)
    assert results[0].status == DARK
    assert results[0].is_dark is True


def test_unmatched_detection_without_coverage_is_indeterminate(t0) -> None:
    """The critical distinction: a coverage hole is not a dark vessel."""
    results = match_detections(
        [DetectionInput(9, 95.0, 25.0, 30.0)], [position(1, 90.0, 21.0, t0=t0)]
    )
    assert results[0].status == INDETERMINATE
    assert results[0].is_dark is None, "SQL NULL is the right value for 'unknown'"
    assert "coverage" in results[0].notes["reason"]


def test_no_ais_at_all_never_yields_a_dark_call(t0) -> None:
    """With an empty AIS feed every detection is unknowable, never accusable."""
    results = match_detections(
        [DetectionInput(i, 90.0 + i * 0.1, 21.0) for i in range(5)], []
    )
    assert {r.status for r in results} == {INDETERMINATE}
    assert all(r.is_dark is None for r in results)


def test_one_ais_vessel_is_claimed_only_once(t0) -> None:
    """Two detections must not both be explained by the same transponder."""
    positions = [position(111, 90.0, 21.0, t0=t0)]
    detections = [
        DetectionInput(1, 90.0002, 21.0, 20.0),
        DetectionInput(2, 90.0004, 21.0, 20.0),
    ]
    results = match_detections(detections, positions)
    matched = [r for r in results if r.status == MATCHED]
    assert len(matched) == 1
    assert matched[0].detection_id == 1  # the closer one wins


def test_untrusted_positions_are_excluded_by_default(t0) -> None:
    """A dead-reckoned guess should not be able to explain away a detection."""
    stale = position(111, 90.0, 21.0, t0=t0, gap=7200, bracketed=False,
                     method="dead_reckoning")
    results = match_detections([DetectionInput(1, 90.0005, 21.0)], [stale])
    assert results[0].status != MATCHED

    relaxed = match_detections(
        [DetectionInput(1, 90.0005, 21.0)], [stale], require_trusted=False
    )
    assert relaxed[0].status == MATCHED


def test_gate_widens_with_speed_and_gap(t0) -> None:
    slow = position(1, 90.0, 21.0, t0=t0, gap=0.0, sog=0.0)
    fast = position(2, 90.0, 21.0, t0=t0, gap=0.0, sog=15.0)
    stale = position(3, 90.0, 21.0, t0=t0, gap=900.0, sog=0.0)

    assert match_gate_m(slow, base_radius_m=500) == pytest.approx(500)
    assert match_gate_m(fast, base_radius_m=500) > match_gate_m(slow, base_radius_m=500)
    assert match_gate_m(stale, base_radius_m=500) > match_gate_m(slow, base_radius_m=500)


def test_gate_accounts_for_vessel_length(t0) -> None:
    p = position(1, 90.0, 21.0, t0=t0, gap=0.0, sog=0.0)
    assert match_gate_m(p, base_radius_m=500, detection_length_m=300) == pytest.approx(800)


def test_gate_is_capped(t0) -> None:
    """An unbounded gate would match a detection to the whole sea."""
    p = position(1, 90.0, 21.0, t0=t0, gap=1e6, sog=40.0)
    assert match_gate_m(p, base_radius_m=500, max_gate_m=5000) == 5000


def test_coverage_check_respects_its_radius(t0) -> None:
    positions = [position(1, 90.0, 21.0, t0=t0)]
    near = DetectionInput(1, 90.2, 21.0)   # ~21 km away
    far = DetectionInput(2, 92.0, 21.0)    # ~208 km away
    assert coverage_is_adequate(positions, near, neighbourhood_km=50) is True
    assert coverage_is_adequate(positions, far, neighbourhood_km=50) is False


def test_summarise_counts_every_status(t0) -> None:
    positions = [position(1, 90.0, 21.0, t0=t0)]
    detections = [
        DetectionInput(1, 90.0002, 21.0),   # matched
        DetectionInput(2, 90.05, 21.0),     # dark (coverage nearby)
        DetectionInput(3, 95.0, 25.0),      # indeterminate
    ]
    assert summarise(match_detections(detections, positions)) == {
        MATCHED: 1, DARK: 1, INDETERMINATE: 1,
    }


def test_results_are_returned_in_input_order(t0) -> None:
    positions = [position(1, 90.0, 21.0, t0=t0)]
    detections = [DetectionInput(i, 90.0 + i * 0.05, 21.0) for i in range(4)]
    results = match_detections(detections, positions)
    assert [r.detection_id for r in results] == [0, 1, 2, 3]
