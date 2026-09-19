"""Interpolating AIS tracks to the SAR acquisition instant.

A SAR scene is a single instant; AIS is a ragged, per-vessel time series
(Class A reports every 2-10 s under way, Class B every 30 s, and far less when
coverage is thin).  To ask "was a vessel here when the satellite looked?" every
track must be evaluated at the acquisition time.

Two estimators, chosen per vessel by what the data supports:

*linear*
    The target instant falls between two fixes: interpolate position directly.
    Great-circle interpolation, so it stays correct across long gaps.
*dead reckoning*
    The instant falls outside the track (or only one fix exists): project the
    nearest fix forward along its course at its speed over ground.

Both report the gap to the nearest real observation, because that gap — not
the interpolation method — is what decides whether a "no AIS here" verdict is
evidence of a dark vessel or merely evidence of thin AIS coverage.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

EARTH_RADIUS_M = 6_371_008.8
KNOTS_TO_MPS = 0.514444


@dataclass(frozen=True)
class Fix:
    """One AIS observation reduced to what interpolation needs."""

    ts: datetime
    lon: float
    lat: float
    sog_kn: float | None = None
    cog_deg: float | None = None


@dataclass(frozen=True)
class InterpolatedPosition:
    mmsi: int
    ts: datetime
    lon: float
    lat: float
    method: str                 # "exact" | "linear" | "dead_reckoning"
    gap_s: float                # seconds to the nearest real fix
    bracketed: bool             # True if real fixes exist both sides
    sog_kn: float | None = None
    cog_deg: float | None = None

    @property
    def trusted(self) -> bool:
        """Cheap default: bracketed fixes within 30 minutes."""
        return self.bracketed and self.gap_s <= 1800.0


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = p2 - p1
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def initial_bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """Forward azimuth from point 1 to point 2, degrees true."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    x = math.sin(dlambda) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dlambda)
    return math.degrees(math.atan2(x, y)) % 360.0


def slerp(lon1: float, lat1: float, lon2: float, lat2: float, frac: float) -> tuple[float, float]:
    """Spherical linear interpolation along the great circle between two points.

    Linear interpolation of lon/lat degrees is wrong over long gaps and breaks
    entirely across the antimeridian; this does not.
    """
    if frac <= 0.0:
        return lon1, lat1
    if frac >= 1.0:
        return lon2, lat2

    p1, l1 = math.radians(lat1), math.radians(lon1)
    p2, l2 = math.radians(lat2), math.radians(lon2)
    d = haversine_m(lon1, lat1, lon2, lat2) / EARTH_RADIUS_M
    if d < 1e-12:
        return lon1, lat1

    a = math.sin((1 - frac) * d) / math.sin(d)
    b = math.sin(frac * d) / math.sin(d)
    x = a * math.cos(p1) * math.cos(l1) + b * math.cos(p2) * math.cos(l2)
    y = a * math.cos(p1) * math.sin(l1) + b * math.cos(p2) * math.sin(l2)
    z = a * math.sin(p1) + b * math.sin(p2)
    lat = math.degrees(math.atan2(z, math.hypot(x, y)))
    lon = math.degrees(math.atan2(y, x))
    return _wrap_lon(lon), lat


def project(lon: float, lat: float, bearing_deg: float, distance_m: float) -> tuple[float, float]:
    """Move ``distance_m`` from a point along a constant bearing."""
    if distance_m == 0.0:
        return lon, lat
    d = distance_m / EARTH_RADIUS_M
    theta = math.radians(bearing_deg)
    p1, l1 = math.radians(lat), math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(theta))
    l2 = l1 + math.atan2(
        math.sin(theta) * math.sin(d) * math.cos(p1),
        math.cos(d) - math.sin(p1) * math.sin(p2),
    )
    return _wrap_lon(math.degrees(l2)), math.degrees(p2)


def _wrap_lon(lon: float) -> float:
    return (lon + 180.0) % 360.0 - 180.0


def interpolate_track(
    mmsi: int,
    fixes: Sequence[Fix],
    target: datetime,
    *,
    max_extrapolation_s: float = 1800.0,
) -> InterpolatedPosition | None:
    """Estimate where ``mmsi`` was at ``target``.

    Returns ``None`` when the track has no fixes, or when the only fixes lie
    further than ``max_extrapolation_s`` outside the target — beyond that,
    dead reckoning says more about the model than about the vessel.
    """
    if not fixes:
        return None
    ordered = sorted(fixes, key=lambda f: f.ts)

    before: Fix | None = None
    after: Fix | None = None
    for fix in ordered:
        if fix.ts <= target:
            before = fix
        else:
            after = fix
            break

    if before is not None and before.ts == target:
        return InterpolatedPosition(
            mmsi=mmsi, ts=target, lon=before.lon, lat=before.lat,
            method="exact", gap_s=0.0, bracketed=True,
            sog_kn=before.sog_kn, cog_deg=before.cog_deg,
        )

    if before is not None and after is not None:
        span = (after.ts - before.ts).total_seconds()
        frac = 0.0 if span <= 0 else (target - before.ts).total_seconds() / span
        lon, lat = slerp(before.lon, before.lat, after.lon, after.lat, frac)
        gap = min(
            (target - before.ts).total_seconds(), (after.ts - target).total_seconds()
        )
        sog = _mean_or_none(before.sog_kn, after.sog_kn)
        cog = before.cog_deg if frac < 0.5 else after.cog_deg
        if cog is None:
            cog = initial_bearing_deg(before.lon, before.lat, after.lon, after.lat)
        return InterpolatedPosition(
            mmsi=mmsi, ts=target, lon=lon, lat=lat, method="linear",
            gap_s=gap, bracketed=True, sog_kn=sog, cog_deg=cog,
        )

    # Only one side available -> dead reckoning from the nearest fix.
    anchor = before or after
    assert anchor is not None
    delta_s = (target - anchor.ts).total_seconds()
    if abs(delta_s) > max_extrapolation_s:
        return None

    lon, lat = anchor.lon, anchor.lat
    if anchor.sog_kn and anchor.cog_deg is not None:
        distance = anchor.sog_kn * KNOTS_TO_MPS * delta_s
        bearing = anchor.cog_deg if distance >= 0 else (anchor.cog_deg + 180.0) % 360.0
        lon, lat = project(lon, lat, bearing, abs(distance))
    return InterpolatedPosition(
        mmsi=mmsi, ts=target, lon=lon, lat=lat, method="dead_reckoning",
        gap_s=abs(delta_s), bracketed=False,
        sog_kn=anchor.sog_kn, cog_deg=anchor.cog_deg,
    )


def _mean_or_none(a: float | None, b: float | None) -> float | None:
    values = [v for v in (a, b) if v is not None]
    return sum(values) / len(values) if values else None


def interpolate_fleet(
    tracks: dict[int, Sequence[Fix]],
    target: datetime,
    *,
    max_extrapolation_s: float = 1800.0,
) -> list[InterpolatedPosition]:
    """Interpolate every track to ``target``, dropping those that cannot be."""
    results = []
    for mmsi, fixes in tracks.items():
        position = interpolate_track(
            mmsi, fixes, target, max_extrapolation_s=max_extrapolation_s
        )
        if position is not None:
            results.append(position)
    return results


def azimuth_displacement_m(sog_kn: float, cog_deg: float, heading_deg: float,
                           *, slant_range_m: float = 850_000.0,
                           platform_speed_mps: float = 7_600.0) -> float:
    """Along-track smear of a moving target in a SAR image.

    A vessel's radial velocity Doppler-shifts its return, and the SAR processor
    maps that shift to an along-track position offset of
    ``R * v_radial / V_platform``.  For Sentinel-1 (R~850 km, V~7.6 km/s) a
    10 kn radial component displaces a target by roughly 575 m — larger than
    the default 500 m match gate, which is exactly why fast vessels appear
    "dark" if this is ignored.  Callers use this to widen the gate for
    fast movers rather than to correct the geometry outright.
    """
    radial_kn = sog_kn * math.cos(math.radians(cog_deg - heading_deg))
    return abs(slant_range_m * radial_kn * KNOTS_TO_MPS / platform_speed_mps)
