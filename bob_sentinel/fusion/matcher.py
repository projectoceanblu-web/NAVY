"""Matching SAR detections to interpolated AIS positions.

The dark-vessel verdict is the whole point of the system, and it is a verdict
about *evidence*, not about guilt.  Three outcomes, not two:

``matched``
    A SAR detection sits within the gate of an interpolated AIS position.
``dark``
    No AIS position is near it, **and** AIS coverage at that place and time was
    good enough that we would have expected to see one.
``indeterminate``
    No AIS match, but coverage was too thin to conclude anything.  Reporting
    these as "dark" is the single easiest way to manufacture false accusations,
    so they are kept separate.

The gate is adaptive: it starts at ``base_radius_m`` and widens for fast
movers, whose along-track Doppler smear can exceed a fixed gate outright
(see :func:`~bob_sentinel.fusion.interpolate.azimuth_displacement_m`), and for
positions whose interpolation had to span a long AIS gap.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field

from bob_sentinel.fusion.interpolate import (
    InterpolatedPosition,
    azimuth_displacement_m,
    haversine_m,
)

log = logging.getLogger(__name__)

MATCHED = "matched"
DARK = "dark"
INDETERMINATE = "indeterminate"

#: Drift allowance per second of AIS gap, in metres.  10 kn ~ 5.1 m/s; this is
#: deliberately below that so the gate does not balloon on stale tracks.
_DRIFT_M_PER_S = 2.0


@dataclass
class DetectionInput:
    """The minimum a detection must carry to be matched."""

    id: int | None
    lon: float
    lat: float
    length_m: float | None = None


@dataclass
class MatchResult:
    detection_id: int | None
    lon: float
    lat: float
    status: str
    matched_mmsi: int | None = None
    distance_m: float | None = None
    gate_m: float | None = None
    notes: dict = field(default_factory=dict)

    @property
    def is_dark(self) -> bool | None:
        """Tri-state, mirroring ``detections.is_dark``.

        ``None`` for indeterminate — SQL NULL is exactly the right semantics
        for "we do not know", and it keeps these rows out of ``WHERE is_dark``.
        """
        if self.status == DARK:
            return True
        if self.status == MATCHED:
            return False
        return None


def match_gate_m(
    position: InterpolatedPosition,
    *,
    base_radius_m: float = 500.0,
    detection_length_m: float | None = None,
    max_gate_m: float = 5000.0,
) -> float:
    """Gate radius for one interpolated position.

    Widened by: AIS position age (drift), vessel size (a 300 m hull's centroid
    and its AIS antenna are not the same point), and Doppler azimuth smear.
    """
    gate = base_radius_m
    gate += _DRIFT_M_PER_S * max(position.gap_s, 0.0)
    if detection_length_m:
        gate += detection_length_m
    if position.sog_kn and position.sog_kn > 2.0:
        # Worst case: the full SOG is radial.  Heading is usually unknown at
        # this point, so assume the geometry that hurts most.
        gate += azimuth_displacement_m(position.sog_kn, 0.0, 0.0)
    return float(min(gate, max_gate_m))


def _bbox_prefilter(
    detection: DetectionInput, positions: Sequence[InterpolatedPosition], radius_m: float
) -> list[InterpolatedPosition]:
    """Cheap degree-box filter before the expensive haversine pass."""
    dlat = radius_m / 111_320.0
    scale = max(math.cos(math.radians(detection.lat)), 1e-6)
    dlon = dlat / scale
    return [
        p
        for p in positions
        if abs(p.lat - detection.lat) <= dlat and abs(p.lon - detection.lon) <= dlon
    ]


def coverage_is_adequate(
    positions: Sequence[InterpolatedPosition],
    detection: DetectionInput,
    *,
    neighbourhood_km: float = 50.0,
    min_vessels: int = 1,
) -> bool:
    """Was AIS reception good enough here to trust a negative?

    If no vessel at all was reporting AIS within ``neighbourhood_km`` of the
    detection, the absence of a match says nothing about this vessel — it says
    the feed was blind there.  aisstream is terrestrial/crowd-contributed and
    genuinely does go thin offshore, so this check matters in the Bay of Bengal.
    """
    radius_m = neighbourhood_km * 1000.0
    nearby = _bbox_prefilter(detection, positions, radius_m)
    count = sum(
        1
        for p in nearby
        if haversine_m(detection.lon, detection.lat, p.lon, p.lat) <= radius_m
    )
    return count >= min_vessels


def match_detections(
    detections: Sequence[DetectionInput],
    positions: Sequence[InterpolatedPosition],
    *,
    base_radius_m: float = 500.0,
    require_trusted: bool = True,
    coverage_km: float = 50.0,
) -> list[MatchResult]:
    """Assign each detection to its nearest acceptable AIS position.

    Greedy nearest-neighbour: detections are matched strongest-constraint
    first (smallest distance wins), and an MMSI is consumed once matched so two
    detections cannot both claim the same vessel.
    """
    usable = [
        p for p in positions if (p.trusted or not require_trusted)
    ]
    if require_trusted and len(usable) < len(positions):
        log.debug(
            "%d/%d interpolated positions dropped as untrusted",
            len(positions) - len(usable),
            len(positions),
        )

    # Build every (detection, position) pair inside its own gate.
    pairs: list[tuple[float, int, int, float]] = []  # distance, det_idx, pos_idx, gate
    for di, detection in enumerate(detections):
        max_gate = max(
            (
                match_gate_m(p, base_radius_m=base_radius_m,
                            detection_length_m=detection.length_m)
                for p in usable
            ),
            default=base_radius_m,
        )
        for index, candidate in _index_prefilter(detection, usable, max_gate):
            gate = match_gate_m(
                candidate, base_radius_m=base_radius_m,
                detection_length_m=detection.length_m,
            )
            distance = haversine_m(detection.lon, detection.lat, candidate.lon, candidate.lat)
            if distance <= gate:
                pairs.append((distance, di, index, gate))

    pairs.sort(key=lambda t: t[0])
    matched_det: dict[int, tuple[int, float, float]] = {}
    claimed_mmsi: set[int] = set()
    for distance, di, pi, gate in pairs:
        if di in matched_det or usable[pi].mmsi in claimed_mmsi:
            continue
        matched_det[di] = (pi, distance, gate)
        claimed_mmsi.add(usable[pi].mmsi)

    results: list[MatchResult] = []
    for di, detection in enumerate(detections):
        if di in matched_det:
            pi, distance, gate = matched_det[di]
            position = usable[pi]
            results.append(
                MatchResult(
                    detection_id=detection.id,
                    lon=detection.lon,
                    lat=detection.lat,
                    status=MATCHED,
                    matched_mmsi=position.mmsi,
                    distance_m=round(distance, 1),
                    gate_m=round(gate, 1),
                    notes={
                        "interpolation": position.method,
                        "ais_gap_s": round(position.gap_s, 1),
                    },
                )
            )
            continue

        adequate = coverage_is_adequate(
            positions, detection, neighbourhood_km=coverage_km
        )
        results.append(
            MatchResult(
                detection_id=detection.id,
                lon=detection.lon,
                lat=detection.lat,
                status=DARK if adequate else INDETERMINATE,
                notes={
                    "reason": (
                        "no AIS within gate despite local AIS coverage"
                        if adequate
                        else f"no AIS vessel reporting within {coverage_km:g} km — "
                             "cannot distinguish a dark vessel from a coverage hole"
                    ),
                    "ais_positions_considered": len(usable),
                },
            )
        )
    return results


def _index_prefilter(
    detection: DetectionInput, positions: Sequence[InterpolatedPosition], radius_m: float
) -> list[tuple[int, InterpolatedPosition]]:
    dlat = radius_m / 111_320.0
    scale = max(math.cos(math.radians(detection.lat)), 1e-6)
    dlon = dlat / scale
    return [
        (i, p)
        for i, p in enumerate(positions)
        if abs(p.lat - detection.lat) <= dlat and abs(p.lon - detection.lon) <= dlon
    ]


def summarise(results: Sequence[MatchResult]) -> dict[str, int]:
    summary = {MATCHED: 0, DARK: 0, INDETERMINATE: 0}
    for result in results:
        summary[result.status] = summary.get(result.status, 0) + 1
    return summary
