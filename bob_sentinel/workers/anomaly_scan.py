"""Periodic AIS behaviour scan: gaps, speed jumps, loitering, encounters.

Run with:  python -m bob_sentinel.workers.anomaly_scan --hours 48
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import UTC, datetime, timedelta

from sqlalchemy import text

from bob_sentinel.config import get_settings
from bob_sentinel.db import session_scope
from bob_sentinel.fusion.anomalies import analyse_tracks, check_identity
from bob_sentinel.fusion.interpolate import Fix
from bob_sentinel.main import configure_logging
from bob_sentinel.repository import insert_anomalies

log = logging.getLogger("bob_sentinel.anomaly_scan")


def load_tracks(session, since: datetime) -> dict[int, list[Fix]]:
    rows = session.execute(
        text(
            """
            SELECT mmsi, ts, ST_X(geom::geometry) AS lon, ST_Y(geom::geometry) AS lat,
                   sog_kn, cog_deg
            FROM ais_positions
            WHERE ts >= :since
            ORDER BY mmsi, ts
            """
        ),
        {"since": since},
    ).mappings()
    tracks: dict[int, list[Fix]] = defaultdict(list)
    for row in rows:
        tracks[row["mmsi"]].append(
            Fix(row["ts"], row["lon"], row["lat"], row["sog_kn"], row["cog_deg"])
        )
    return dict(tracks)


def scan(hours: int, *, include_encounters: bool = True) -> dict[str, int]:
    since = datetime.now(UTC) - timedelta(hours=hours)
    with session_scope() as session:
        tracks = load_tracks(session, since)
        log.info("scanning %d vessel tracks since %s", len(tracks), since.isoformat())

        events = analyse_tracks(tracks, include_encounters=include_encounters)

        # Identity checks read from the vessel table, not the track.
        identities = session.execute(
            text("SELECT mmsi, imo, name FROM vessels WHERE last_seen >= :since"),
            {"since": since},
        ).mappings()
        for row in identities:
            events.extend(check_identity(row["mmsi"], imo=row["imo"], name=row["name"]))

        written = insert_anomalies(session, [e.to_row() for e in events])

    counts: dict[str, int] = defaultdict(int)
    for event in events:
        counts[event.kind] += 1
    log.info("found %d anomalies (%d new): %s", len(events), written, dict(counts))
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="BoB Sentinel AIS anomaly scan")
    parser.add_argument("--hours", type=int, default=48, help="Look-back window")
    parser.add_argument("--no-encounters", action="store_true",
                        help="Skip the O(n^2) pairwise encounter pass")
    args = parser.parse_args()

    configure_logging(get_settings().log_level)
    scan(args.hours, include_encounters=not args.no_encounters)


if __name__ == "__main__":
    main()
