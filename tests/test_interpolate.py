"""AIS interpolation: geodesy, method selection, and refusal to over-reach."""

from __future__ import annotations

from datetime import timedelta

import pytest

from bob_sentinel.fusion.interpolate import (
    Fix,
    azimuth_displacement_m,
    haversine_m,
    initial_bearing_deg,
    interpolate_fleet,
    interpolate_track,
    project,
    slerp,
)


def test_haversine_matches_known_distance() -> None:
    # One degree of latitude is ~111.2 km everywhere.
    assert haversine_m(90.0, 21.0, 90.0, 22.0) == pytest.approx(111_195, rel=0.001)


def test_bearing_due_east() -> None:
    assert initial_bearing_deg(90.0, 0.0, 91.0, 0.0) == pytest.approx(90.0, abs=0.01)


def test_slerp_midpoint_is_equidistant() -> None:
    lon, lat = slerp(88.0, 20.0, 92.0, 22.0, 0.5)
    assert haversine_m(88.0, 20.0, lon, lat) == pytest.approx(
        haversine_m(lon, lat, 92.0, 22.0), rel=1e-6
    )


def test_slerp_crosses_the_antimeridian() -> None:
    """Naive lon averaging would land at 0 degrees — the far side of the planet."""
    lon, lat = slerp(179.0, 0.0, -179.0, 0.0, 0.5)
    assert abs(lon) > 179.0


def test_project_moves_the_expected_distance() -> None:
    lon, lat = project(90.0, 21.0, 90.0, 10_000.0)
    assert haversine_m(90.0, 21.0, lon, lat) == pytest.approx(10_000, rel=1e-4)


def test_linear_interpolation_between_fixes(t0) -> None:
    fixes = [
        Fix(t0, 90.0, 21.0, 10.0, 90.0),
        Fix(t0 + timedelta(minutes=10), 90.05, 21.0, 10.0, 90.0),
    ]
    position = interpolate_track(1, fixes, t0 + timedelta(minutes=5))
    assert position is not None
    assert position.method == "linear"
    assert position.bracketed is True
    assert position.trusted is True
    assert position.lon == pytest.approx(90.025, abs=1e-4)


def test_exact_hit_is_reported_as_exact(t0) -> None:
    position = interpolate_track(1, [Fix(t0, 90.0, 21.0)], t0)
    assert position is not None
    assert position.method == "exact"
    assert position.gap_s == 0.0


def test_dead_reckoning_projects_along_course(t0) -> None:
    """A single fix is extrapolated along its own SOG/COG."""
    fix = Fix(t0, 90.0, 21.0, 10.0, 90.0)  # 10 kn due east
    position = interpolate_track(1, [fix], t0 + timedelta(minutes=5))
    assert position is not None
    assert position.method == "dead_reckoning"
    assert position.bracketed is False
    assert position.trusted is False, "unbracketed estimates must not be trusted"
    expected_m = 10.0 * 0.514444 * 300
    assert haversine_m(90.0, 21.0, position.lon, position.lat) == pytest.approx(
        expected_m, rel=0.01
    )


def test_dead_reckoning_refuses_to_over_extrapolate(t0) -> None:
    """Beyond the limit, the estimate describes the model, not the vessel."""
    fix = Fix(t0, 90.0, 21.0, 10.0, 90.0)
    assert interpolate_track(1, [fix], t0 + timedelta(hours=3)) is None


def test_stationary_fix_without_speed_stays_put(t0) -> None:
    position = interpolate_track(1, [Fix(t0, 90.0, 21.0)], t0 + timedelta(minutes=5))
    assert position is not None
    assert (position.lon, position.lat) == (90.0, 21.0)


def test_empty_track_returns_none(t0) -> None:
    assert interpolate_track(1, [], t0) is None


def test_gap_is_distance_to_nearest_real_fix(t0) -> None:
    fixes = [
        Fix(t0, 90.0, 21.0),
        Fix(t0 + timedelta(minutes=60), 90.1, 21.0),
    ]
    position = interpolate_track(1, fixes, t0 + timedelta(minutes=50))
    assert position is not None
    assert position.gap_s == pytest.approx(600.0)  # nearer to the later fix
    assert position.trusted is True


def test_unsorted_fixes_are_handled(t0) -> None:
    fixes = [
        Fix(t0 + timedelta(minutes=10), 90.05, 21.0),
        Fix(t0, 90.0, 21.0),
    ]
    position = interpolate_track(1, fixes, t0 + timedelta(minutes=5))
    assert position is not None
    assert position.lon == pytest.approx(90.025, abs=1e-4)


def test_interpolate_fleet_drops_untenable_tracks(t0) -> None:
    tracks = {
        1: [Fix(t0, 90.0, 21.0), Fix(t0 + timedelta(minutes=10), 90.05, 21.0)],
        2: [Fix(t0 - timedelta(days=1), 91.0, 21.0, 10.0, 90.0)],  # far too stale
        3: [],
    }
    positions = interpolate_fleet(tracks, t0 + timedelta(minutes=5))
    assert [p.mmsi for p in positions] == [1]


def test_azimuth_displacement_exceeds_a_fixed_gate() -> None:
    """The reason the match gate must widen for fast movers.

    Sentinel-1 geometry smears a 10 kn radial target ~575 m along-track —
    already past a fixed 500 m gate, so ignoring it manufactures dark vessels.
    """
    assert azimuth_displacement_m(10.0, 0.0, 0.0) == pytest.approx(575, abs=25)
    assert azimuth_displacement_m(10.0, 0.0, 0.0) > 500.0
    # A purely cross-track heading produces no Doppler offset.
    assert azimuth_displacement_m(10.0, 90.0, 0.0) == pytest.approx(0.0, abs=1e-6)
