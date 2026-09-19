#!/usr/bin/env python3
"""End-to-end demo with synthetic data — no credentials required.

Generates a small synthetic SAR scene and a matching AIS picture, then runs the
real pipeline over them: CFAR detection, geolocation, AIS interpolation to the
acquisition instant, matching, dark-vessel flagging and anomaly scanning.

Useful for proving the stack is wired correctly before any account exists, and
for exercising the API without waiting for a Sentinel-1 overpass.

    python scripts/demo_pipeline.py

The synthetic vessels are fictional. Nothing here is a real observation.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import text  # noqa: E402

from bob_sentinel.db import get_engine, session_scope  # noqa: E402
from bob_sentinel.fusion.anomalies import analyse_tracks  # noqa: E402
from bob_sentinel.fusion.interpolate import Fix, interpolate_fleet  # noqa: E402
from bob_sentinel.fusion.matcher import (  # noqa: E402
    DetectionInput,
    match_detections,
    summarise,
)
from bob_sentinel.ingest.aisstream import AISRecord  # noqa: E402
from bob_sentinel.ingest.eez import load_fallback_bangladesh_eez  # noqa: E402
from bob_sentinel.main import configure_logging  # noqa: E402
from bob_sentinel.models import Base, Detection, SARScene  # noqa: E402
from bob_sentinel.repository import (  # noqa: E402
    get_region,
    insert_anomalies,
    insert_detections,
    upsert_ais_records,
    upsert_scene,
)
from bob_sentinel.sar.cfar import CFARParams  # noqa: E402
from bob_sentinel.sar.pipeline import detect_in_scene  # noqa: E402

log = logging.getLogger("demo")

ACQUIRED = datetime.now(UTC).replace(microsecond=0) - timedelta(hours=1)

# Fictional vessels placed in the Bay of Bengal.
#   (lon, lat, broadcasts_ais)
SCENARIO = [
    (90.400, 21.300, True),    # cooperative: SAR + AIS -> matched
    (90.460, 21.320, True),    # cooperative: establishes local AIS coverage
    (90.520, 21.300, False),   # SAR only, AIS nearby        -> dark
    (88.700, 20.650, False),   # SAR only, no AIS for miles  -> indeterminate
]

# The demo uses coarser pixels than a real IW GRDH product (10 m).  At true
# GRDH spacing the scenario's ~2 degree extent would be a 20,000 px scene —
# realistic, but a slow and pointless thing to synthesise for a smoke test.
PIXEL_DEG = 0.0005          # ~52 m near 21 N
MARGIN_DEG = 0.15           # keeps every target clear of the scene edge

# Scene bounds are derived from the scenario so the two cannot drift apart.
ORIGIN_LON = min(lon for lon, _, _ in SCENARIO) - MARGIN_DEG
ORIGIN_LAT = max(lat for _, lat, _ in SCENARIO) + MARGIN_DEG
WIDTH_PX = int(
    round((max(lon for lon, _, _ in SCENARIO) + MARGIN_DEG - ORIGIN_LON) / PIXEL_DEG)
)
HEIGHT_PX = int(
    round((ORIGIN_LAT - (min(lat for _, lat, _ in SCENARIO) - MARGIN_DEG)) / PIXEL_DEG)
)


def build_synthetic_scene(path: Path) -> None:
    """Rayleigh sea clutter with bright targets at the scenario positions."""
    import rasterio
    from rasterio.transform import from_origin

    rng = np.random.default_rng(2026)
    data = rng.rayleigh(scale=1.0, size=(HEIGHT_PX, WIDTH_PX)).astype(np.float32)

    for lon, lat, _ in SCENARIO:
        col = int(round((lon - ORIGIN_LON) / PIXEL_DEG))
        row = int(round((ORIGIN_LAT - lat) / PIXEL_DEG))
        if not (0 <= row < HEIGHT_PX - 4 and 0 <= col < WIDTH_PX - 3):
            raise SystemExit(
                f"scenario target {lon},{lat} falls outside the derived scene "
                "bounds — this is a bug in the demo geometry"
            )
        data[row : row + 4, col : col + 3] += 45.0

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        path, "w", driver="GTiff", height=HEIGHT_PX, width=WIDTH_PX, count=1,
        dtype="float32", crs="EPSG:4326",
        transform=from_origin(ORIGIN_LON, ORIGIN_LAT, PIXEL_DEG, PIXEL_DEG),
    ) as dst:
        dst.write(data, 1)
    log.info("wrote synthetic scene %s (%d x %d px)", path.name, WIDTH_PX, HEIGHT_PX)


def seed_ais(session) -> int:
    """AIS tracks for the cooperative vessels, bracketing the acquisition.

    One vessel is also given a long silence so the anomaly scan has something
    real to find.
    """
    records: list[AISRecord] = []
    for mmsi_index, (lon, lat, broadcasts) in enumerate(SCENARIO):
        if not broadcasts:
            continue
        mmsi = 405000100 + mmsi_index
        # A fix every 4 minutes for 40 minutes either side of the acquisition.
        for step in range(-10, 11):
            ts = ACQUIRED + timedelta(minutes=4 * step)
            records.append(
                AISRecord(
                    mmsi=mmsi, ts=ts,
                    lon=lon + step * 0.0008, lat=lat,
                    sog_kn=4.5, cog_deg=90.0, name=f"DEMO VESSEL {mmsi_index}",
                )
            )
    # A deliberate silence after a well-reported period: 20 fixes at 15-minute
    # intervals (satisfying the >=14-positions-in-12h precondition), then an
    # 8-hour gap, then a reappearance far away.  detect_gaps needs > 6 h.
    gap_mmsi = 405000199
    last_fix_ts = ACQUIRED - timedelta(hours=6)
    for step in range(20):
        last_fix_ts = ACQUIRED - timedelta(hours=6) + timedelta(minutes=15 * step)
        records.append(
            AISRecord(
                mmsi=gap_mmsi, ts=last_fix_ts,
                lon=91.2 + 0.002 * step, lat=21.6, sog_kn=7.0, cog_deg=90.0,
                name="DEMO GOES DARK",
            )
        )
    records.append(
        AISRecord(mmsi=gap_mmsi, ts=last_fix_ts + timedelta(hours=8),
                  lon=92.0, lat=21.9, sog_kn=7.0, cog_deg=90.0)
    )

    written, vessels = upsert_ais_records(session, records)
    log.info("seeded %d AIS positions across %d vessels", written, vessels)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/work"))
    parser.add_argument("--keep", action="store_true",
                        help="Keep existing rows instead of starting clean")
    args = parser.parse_args()

    configure_logging("INFO")
    engine = get_engine()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS postgis"))
    Base.metadata.create_all(engine)

    scene_path = args.data_dir / "demo_scene.tif"
    build_synthetic_scene(scene_path)

    with session_scope() as session:
        if not args.keep:
            for table in reversed(Base.metadata.sorted_tables):
                session.execute(text(f"TRUNCATE {table.name} RESTART IDENTITY CASCADE"))

        if get_region(session, mrgid=8481) is None:
            load_fallback_bangladesh_eez(session)
        seed_ais(session)

        scene = upsert_scene(
            session,
            product_id="DEMO-0001",
            product_name="DEMO_S1D_IW_GRDH_1SDV_SYNTHETIC",
            platform="S1D", product_type="IW_GRDH_1S",
            acquired_start=ACQUIRED,
            acquired_end=ACQUIRED + timedelta(seconds=28),
            footprint_wkt=(
                f"POLYGON(({ORIGIN_LON} {ORIGIN_LAT},"
                f"{ORIGIN_LON + WIDTH_PX * PIXEL_DEG} {ORIGIN_LAT},"
                f"{ORIGIN_LON + WIDTH_PX * PIXEL_DEG} {ORIGIN_LAT - HEIGHT_PX * PIXEL_DEG},"
                f"{ORIGIN_LON} {ORIGIN_LAT - HEIGHT_PX * PIXEL_DEG},"
                f"{ORIGIN_LON} {ORIGIN_LAT}))"
            ),
            local_path=str(scene_path), status="processing",
            detector="ca-cfar", detector_params={"model": "gamma", "pfa": 1e-9},
        )
        scene_id = scene.id

    params = CFARParams(pfa=1e-9, guard=8, train=24, model="gamma")
    detections = detect_in_scene(scene_path, params=params, tile=512, overlap=128)
    log.info("CFAR found %d candidate vessels", len(detections))

    with session_scope() as session:
        scene = session.get(SARScene, scene_id)
        insert_detections(session, scene, [d.to_dict() for d in detections])
        target = scene.acquisition_midpoint

        rows = session.execute(text(
            "SELECT mmsi, ts, ST_X(geom::geometry) lon, ST_Y(geom::geometry) lat,"
            " sog_kn, cog_deg FROM ais_positions ORDER BY mmsi, ts"
        )).mappings()
        tracks: dict[int, list[Fix]] = {}
        for row in rows:
            tracks.setdefault(row["mmsi"], []).append(
                Fix(row["ts"], row["lon"], row["lat"], row["sog_kn"], row["cog_deg"])
            )

        positions = interpolate_fleet(tracks, target)
        log.info("interpolated %d AIS tracks to %s", len(positions), target.isoformat())

        detection_rows = session.execute(text(
            "SELECT id, ST_X(geom::geometry) lon, ST_Y(geom::geometry) lat, length_m"
            " FROM detections WHERE scene_id = :s ORDER BY id"
        ), {"s": scene_id}).mappings()
        inputs = [
            DetectionInput(r["id"], r["lon"], r["lat"], r["length_m"])
            for r in detection_rows
        ]
        results = match_detections(inputs, positions, base_radius_m=500.0)
        for result in results:
            session.query(Detection).filter(Detection.id == result.detection_id).update(
                {"is_dark": result.is_dark, "matched_mmsi": result.matched_mmsi,
                 "match_distance_m": result.distance_m,
                 "match_notes": {**result.notes, "status": result.status,
                                 "gate_m": result.gate_m}},
                synchronize_session=False,
            )
        session.query(SARScene).filter(SARScene.id == scene_id).update(
            {"status": "complete"}, synchronize_session=False
        )

        events = analyse_tracks(tracks)
        insert_anomalies(session, [e.to_row() for e in events])

    summary = summarise(results)
    print("\n" + "=" * 62)
    print("  BoB Sentinel demo complete (synthetic data)")
    print("=" * 62)
    print(f"  SAR detections     : {len(detections)}")
    print(f"  Matched to AIS     : {summary['matched']}")
    print(f"  Dark               : {summary['dark']}")
    print(f"  Indeterminate      : {summary['indeterminate']}")
    print(f"  AIS anomalies      : {len(events)}"
          f" ({', '.join(sorted({e.kind for e in events})) or 'none'})")
    print("=" * 62)
    print("  Start the API and open http://localhost:8000 to see the map.")
    print("  These are fictional vessels — not real observations.\n")


if __name__ == "__main__":
    main()
