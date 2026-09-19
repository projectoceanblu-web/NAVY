"""AIS behavioural heuristics: gaps, speed jumps, loitering, encounters.

These are screening heuristics, not findings.  Each one has an innocent
explanation that is usually the correct one — AIS receivers fail, transponders
are switched off in bad weather, vessels drift while fishing gear soaks, and
two hulls can pass close in a shipping lane.  Every anomaly carries the numbers
that produced it so an analyst can dismiss it quickly.

The gap rule follows Global Fishing Watch's template: only call a silence a
"gap" if the vessel was reporting reliably beforehand, otherwise every patchy
receiver becomes an accusation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from bob_sentinel.fusion.interpolate import Fix, haversine_m

# --- kinds -----------------------------------------------------------------
GAP = "ais_gap"
SPEED_JUMP = "impossible_speed"
LOITERING = "loitering"
ENCOUNTER = "encounter"
IDENTITY = "identity_anomaly"

#: Faster than this over the ground is not a vessel; it is bad data or two
#: vessels sharing an MMSI.  The fastest commercial ships top out near 45 kn.
MAX_PLAUSIBLE_SOG_KN = 60.0

#: GFW's rule of thumb: a gap only counts if reception was good beforehand.
MIN_POSITIONS_BEFORE_GAP = 14
POSITIONS_LOOKBACK = timedelta(hours=12)


@dataclass
class AnomalyEvent:
    kind: str
    mmsi: int
    start_ts: datetime
    end_ts: datetime | None = None
    lon: float | None = None
    lat: float | None = None
    counterpart_mmsi: int | None = None
    score: float | None = None
    details: dict | None = None

    def to_row(self) -> dict:
        return {
            "kind": self.kind,
            "mmsi": self.mmsi,
            "counterpart_mmsi": self.counterpart_mmsi,
            "start_ts": self.start_ts,
            "end_ts": self.end_ts,
            "lon": self.lon,
            "lat": self.lat,
            "score": self.score,
            "details": self.details or {},
        }


def _speed_kn(a: Fix, b: Fix) -> float:
    seconds = (b.ts - a.ts).total_seconds()
    if seconds <= 0:
        return 0.0
    metres = haversine_m(a.lon, a.lat, b.lon, b.lat)
    return (metres / seconds) / 0.514444


def detect_gaps(
    mmsi: int,
    fixes: Sequence[Fix],
    *,
    min_gap: timedelta = timedelta(hours=6),
    min_prior_positions: int = MIN_POSITIONS_BEFORE_GAP,
) -> list[AnomalyEvent]:
    """Flag silences that follow a well-reported period.

    The ``min_prior_positions`` precondition is what separates "this vessel
    went dark" from "our receiver never heard it well in the first place".
    """
    ordered = sorted(fixes, key=lambda f: f.ts)
    events: list[AnomalyEvent] = []
    for i in range(len(ordered) - 1):
        current, following = ordered[i], ordered[i + 1]
        duration = following.ts - current.ts
        if duration < min_gap:
            continue
        window_start = current.ts - POSITIONS_LOOKBACK
        prior = sum(1 for f in ordered[: i + 1] if f.ts >= window_start)
        if prior < min_prior_positions:
            continue
        distance_km = (
            haversine_m(current.lon, current.lat, following.lon, following.lat) / 1000.0
        )
        events.append(
            AnomalyEvent(
                kind=GAP,
                mmsi=mmsi,
                start_ts=current.ts,
                end_ts=following.ts,
                lon=current.lon,
                lat=current.lat,
                score=min(duration.total_seconds() / (24 * 3600), 1.0),
                details={
                    "duration_hours": round(duration.total_seconds() / 3600, 2),
                    "prior_positions_12h": prior,
                    "resumed_km_away": round(distance_km, 1),
                    "resumed_lon": following.lon,
                    "resumed_lat": following.lat,
                },
            )
        )
    return events


def detect_speed_jumps(
    mmsi: int, fixes: Sequence[Fix], *, max_sog_kn: float = MAX_PLAUSIBLE_SOG_KN
) -> list[AnomalyEvent]:
    """Flag consecutive fixes implying an impossible speed.

    Usually a duplicated/spoofed MMSI, a bad decode, or two receivers reporting
    the same vessel with skewed clocks — all worth surfacing.
    """
    ordered = sorted(fixes, key=lambda f: f.ts)
    events: list[AnomalyEvent] = []
    for a, b in zip(ordered, ordered[1:], strict=False):
        implied = _speed_kn(a, b)
        if implied <= max_sog_kn:
            continue
        events.append(
            AnomalyEvent(
                kind=SPEED_JUMP,
                mmsi=mmsi,
                start_ts=a.ts,
                end_ts=b.ts,
                lon=b.lon,
                lat=b.lat,
                score=min(implied / (max_sog_kn * 4), 1.0),
                details={
                    "implied_sog_kn": round(implied, 1),
                    "threshold_kn": max_sog_kn,
                    "separation_km": round(haversine_m(a.lon, a.lat, b.lon, b.lat) / 1000, 1),
                    "interval_s": round((b.ts - a.ts).total_seconds(), 1),
                    "hypothesis": "duplicated MMSI, spoofed identity, or bad decode",
                },
            )
        )
    return events


def detect_loitering(
    mmsi: int,
    fixes: Sequence[Fix],
    *,
    max_sog_kn: float = 2.0,
    min_duration: timedelta = timedelta(hours=3),
    max_radius_km: float = 10.0,
) -> list[AnomalyEvent]:
    """Flag sustained slow movement within a small radius.

    Loitering is how both fishing and transhipment look from orbit; it is a
    prompt to look, never a conclusion.
    """
    ordered = sorted(fixes, key=lambda f: f.ts)
    events: list[AnomalyEvent] = []
    start = 0
    while start < len(ordered):
        end = start
        while end + 1 < len(ordered):
            nxt = ordered[end + 1]
            sog = nxt.sog_kn if nxt.sog_kn is not None else _speed_kn(ordered[end], nxt)
            if sog > max_sog_kn:
                break
            drift_m = haversine_m(
                ordered[start].lon, ordered[start].lat, nxt.lon, nxt.lat
            )
            if drift_m > max_radius_km * 1000:
                break
            end += 1
        duration = ordered[end].ts - ordered[start].ts
        if end > start and duration >= min_duration:
            lons = [f.lon for f in ordered[start : end + 1]]
            lats = [f.lat for f in ordered[start : end + 1]]
            events.append(
                AnomalyEvent(
                    kind=LOITERING,
                    mmsi=mmsi,
                    start_ts=ordered[start].ts,
                    end_ts=ordered[end].ts,
                    lon=sum(lons) / len(lons),
                    lat=sum(lats) / len(lats),
                    score=min(duration.total_seconds() / (12 * 3600), 1.0),
                    details={
                        "duration_hours": round(duration.total_seconds() / 3600, 2),
                        "positions": end - start + 1,
                        "max_sog_kn": max_sog_kn,
                    },
                )
            )
            start = end + 1
        else:
            start += 1
    return events


def detect_encounters(
    tracks: dict[int, Sequence[Fix]],
    *,
    max_separation_km: float = 0.5,
    min_duration: timedelta = timedelta(hours=2),
    max_sog_kn: float = 2.0,
    time_tolerance: timedelta = timedelta(minutes=15),
) -> list[AnomalyEvent]:
    """Flag two vessels slow and close together for a sustained period.

    Mirrors the GFW encounter definition; at sea this is the signature of
    transhipment, which is how catch gets laundered out of an EEZ.
    """
    events: list[AnomalyEvent] = []
    mmsis = sorted(tracks)
    for i, first in enumerate(mmsis):
        for second in mmsis[i + 1 :]:
            close = _close_samples(
                tracks[first], tracks[second], max_separation_km, max_sog_kn, time_tolerance
            )
            for window in _contiguous_windows(close, min_duration, time_tolerance):
                start_ts, end_ts, lon, lat, separation = window
                events.append(
                    AnomalyEvent(
                        kind=ENCOUNTER,
                        mmsi=first,
                        counterpart_mmsi=second,
                        start_ts=start_ts,
                        end_ts=end_ts,
                        lon=lon,
                        lat=lat,
                        score=min((end_ts - start_ts).total_seconds() / (6 * 3600), 1.0),
                        details={
                            "duration_hours": round((end_ts - start_ts).total_seconds() / 3600, 2),
                            "median_separation_km": round(separation, 3),
                            "max_sog_kn": max_sog_kn,
                        },
                    )
                )
    return events


def _close_samples(
    track_a: Sequence[Fix],
    track_b: Sequence[Fix],
    max_separation_km: float,
    max_sog_kn: float,
    tolerance: timedelta,
) -> list[tuple[datetime, float, float, float]]:
    """Time-aligned samples where both vessels are slow and close."""
    b_sorted = sorted(track_b, key=lambda f: f.ts)
    samples = []
    for fix_a in sorted(track_a, key=lambda f: f.ts):
        partner = _nearest_in_time(b_sorted, fix_a.ts, tolerance)
        if partner is None:
            continue
        if (fix_a.sog_kn or 0.0) > max_sog_kn or (partner.sog_kn or 0.0) > max_sog_kn:
            continue
        separation_km = (
            haversine_m(fix_a.lon, fix_a.lat, partner.lon, partner.lat) / 1000.0
        )
        if separation_km <= max_separation_km:
            samples.append(
                (
                    fix_a.ts,
                    (fix_a.lon + partner.lon) / 2,
                    (fix_a.lat + partner.lat) / 2,
                    separation_km,
                )
            )
    return samples


def _nearest_in_time(fixes: Sequence[Fix], when: datetime, tolerance: timedelta) -> Fix | None:
    best, best_delta = None, tolerance.total_seconds()
    for fix in fixes:
        delta = abs((fix.ts - when).total_seconds())
        if delta <= best_delta:
            best, best_delta = fix, delta
    return best


def _contiguous_windows(
    samples: Sequence[tuple[datetime, float, float, float]],
    min_duration: timedelta,
    tolerance: timedelta,
):
    """Group samples into runs broken by gaps larger than ``max_gap``."""
    if not samples:
        return
    # A run breaks when consecutive samples are further apart than a generous
    # multiple of the alignment tolerance.
    max_gap = max(tolerance * 4, timedelta(hours=1))
    run = [samples[0]]
    for sample in samples[1:]:
        if sample[0] - run[-1][0] > max_gap:
            yield from _emit_window(run, min_duration)
            run = [sample]
        else:
            run.append(sample)
    yield from _emit_window(run, min_duration)


def _emit_window(run: Sequence[tuple[datetime, float, float, float]], min_duration: timedelta):
    if len(run) < 2:
        return
    duration = run[-1][0] - run[0][0]
    if duration < min_duration:
        return
    lons = sorted(s[1] for s in run)
    lats = sorted(s[2] for s in run)
    separations = sorted(s[3] for s in run)
    mid = len(run) // 2
    yield (run[0][0], run[-1][0], lons[mid], lats[mid], separations[mid])


def check_identity(
    mmsi: int, *, imo: int | None = None, name: str | None = None
) -> list[AnomalyEvent]:
    """Structural validation of an AIS identity.

    A valid MMSI is 9 digits whose first three are a real MID (Maritime
    Identification Digits, 201-775).  Values outside that are placeholders,
    test transmissions, or deliberate spoofs.
    """
    problems = []
    text = str(mmsi)
    if len(text) != 9:
        problems.append(f"MMSI is {len(text)} digits, not 9")
    else:
        mid = int(text[:3])
        if not 201 <= mid <= 775:
            problems.append(f"MID {mid} is outside the assigned range 201-775")
    if imo is not None and not _valid_imo(imo):
        problems.append(f"IMO {imo} fails its check-digit test")
    if not problems:
        return []
    return [
        AnomalyEvent(
            kind=IDENTITY,
            mmsi=mmsi,
            start_ts=datetime.now(UTC),
            score=0.5,
            details={"problems": problems, "name": name, "imo": imo},
        )
    ]


def _valid_imo(imo: int) -> bool:
    """IMO numbers are 7 digits; the last is a weighted-sum check digit."""
    text = str(imo)
    if len(text) != 7 or not text.isdigit():
        return False
    total = sum(
        int(digit) * weight
        for digit, weight in zip(text[:6], range(7, 1, -1), strict=True)
    )
    return total % 10 == int(text[6])


def analyse_tracks(
    tracks: dict[int, Sequence[Fix]],
    *,
    include_encounters: bool = True,
) -> list[AnomalyEvent]:
    """Run every per-vessel heuristic, plus pairwise encounters."""
    events: list[AnomalyEvent] = []
    for mmsi, fixes in tracks.items():
        if len(fixes) < 2:
            continue
        events.extend(detect_gaps(mmsi, fixes))
        events.extend(detect_speed_jumps(mmsi, fixes))
        events.extend(detect_loitering(mmsi, fixes))
    if include_encounters:
        events.extend(detect_encounters(tracks))
    events.sort(key=lambda e: (e.start_ts, e.kind))
    return events
