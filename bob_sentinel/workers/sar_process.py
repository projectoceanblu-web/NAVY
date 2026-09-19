"""SAR worker: search CDSE, download, detect, fuse against AIS, persist.

Run with:
    python -m bob_sentinel.workers.sar_process --days 7 --limit 2
    python -m bob_sentinel.workers.sar_process --scene /data/scenes/S1D_...zip
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from datetime import datetime, timedelta
from pathlib import Path

from sqlalchemy import text

from bob_sentinel.clients.gfw import BANGLADESH_EEZ_ID  # noqa: F401  (documented default)
from bob_sentinel.config import get_settings
from bob_sentinel.db import session_scope
from bob_sentinel.fusion.interpolate import Fix, interpolate_fleet
from bob_sentinel.fusion.matcher import DetectionInput, match_detections, summarise
from bob_sentinel.main import configure_logging
from bob_sentinel.models import Detection, SARScene
from bob_sentinel.repository import get_region, insert_detections, upsert_scene
from bob_sentinel.sar.cdse import CDSEClient, Product, recent_window
from bob_sentinel.sar.cfar import CFARParams
from bob_sentinel.sar.pipeline import detect_in_scene

log = logging.getLogger("bob_sentinel.sar_process")


def aoi_wkt(session) -> str | None:
    """The EEZ polygon as WKT, for the CDSE spatial filter.

    Simplified first: CDSE rejects an over-long ``$filter``, and a
    several-thousand-vertex EEZ boundary will blow that budget.
    """
    settings = get_settings()
    region = get_region(session, mrgid=settings.aoi_mrgid)
    if region is None:
        log.warning("no AOI region loaded; run scripts/load_eez.py first")
        return None
    return session.execute(
        text(
            "SELECT ST_AsText(ST_ConvexHull(ST_SimplifyPreserveTopology("
            "geom::geometry, 0.05))) FROM regions WHERE id = :id"
        ),
        {"id": region.id},
    ).scalar()


def catalogue_scenes(days: int, limit: int) -> list[Product]:
    settings = get_settings()
    start, end = recent_window(days)
    with session_scope() as session:
        wkt = aoi_wkt(session)
    with CDSEClient(settings.cdse_username or "", settings.cdse_password or "") as client:
        products = client.search(start=start, end=end, wkt=wkt, limit=limit)
    log.info("catalogue returned %d products for the last %d days", len(products), days)
    return products


def load_ais_tracks(session, when: datetime, window_s: int) -> dict[int, list[Fix]]:
    """AIS fixes bracketing the acquisition, grouped by MMSI."""
    rows = session.execute(
        text(
            """
            SELECT mmsi, ts, ST_X(geom::geometry) AS lon, ST_Y(geom::geometry) AS lat,
                   sog_kn, cog_deg
            FROM ais_positions
            WHERE ts BETWEEN :lo AND :hi
            ORDER BY mmsi, ts
            """
        ),
        {
            "lo": when - timedelta(seconds=window_s),
            "hi": when + timedelta(seconds=window_s),
        },
    ).mappings()

    tracks: dict[int, list[Fix]] = defaultdict(list)
    for row in rows:
        tracks[row["mmsi"]].append(
            Fix(row["ts"], row["lon"], row["lat"], row["sog_kn"], row["cog_deg"])
        )
    return dict(tracks)


def fuse_scene(session, scene: SARScene) -> dict[str, int]:
    """Match a scene's detections against interpolated AIS and store verdicts."""
    settings = get_settings()
    target = scene.acquisition_midpoint

    tracks = load_ais_tracks(session, target, settings.match_max_gap_s)
    positions = interpolate_fleet(
        tracks, target, max_extrapolation_s=settings.match_max_gap_s
    )
    log.info(
        "scene %s: %d AIS tracks -> %d interpolated positions at %s",
        scene.product_name, len(tracks), len(positions), target.isoformat(),
    )

    rows = session.execute(
        text(
            """
            SELECT id, ST_X(geom::geometry) AS lon, ST_Y(geom::geometry) AS lat, length_m
            FROM detections WHERE scene_id = :scene_id
            """
        ),
        {"scene_id": scene.id},
    ).mappings()
    detections = [
        DetectionInput(r["id"], r["lon"], r["lat"], r["length_m"]) for r in rows
    ]
    if not detections:
        return {"matched": 0, "dark": 0, "indeterminate": 0}

    results = match_detections(
        detections, positions, base_radius_m=settings.match_radius_m
    )
    for result in results:
        session.query(Detection).filter(Detection.id == result.detection_id).update(
            {
                "is_dark": result.is_dark,
                "matched_mmsi": result.matched_mmsi,
                "match_distance_m": result.distance_m,
                "match_notes": {**result.notes, "gate_m": result.gate_m,
                                "status": result.status},
            },
            synchronize_session=False,
        )
    summary = summarise(results)
    log.info("scene %s fusion: %s", scene.product_name, summary)
    return summary


def process_local_scene(path: Path, *, product: Product | None = None) -> None:
    """Detect in a downloaded product and fuse the results."""
    settings = get_settings()
    params = CFARParams(
        pfa=settings.cfar_pfa, guard=settings.cfar_guard, train=settings.cfar_train
    )

    with session_scope() as session:
        if product is not None:
            scene = upsert_scene(
                session,
                product_id=product.id,
                product_name=product.name,
                platform=product.platform,
                product_type=product.product_type,
                acquired_start=product.acquired_start,
                acquired_end=product.acquired_end,
                footprint_wkt=product.footprint_wkt,
                local_path=str(path),
                status="processing",
                detector="ca-cfar",
                detector_params={
                    "pfa": params.pfa, "guard": params.guard,
                    "train": params.train, "model": params.model,
                },
            )
        else:
            scene = session.execute(
                text("SELECT * FROM sar_scenes WHERE local_path = :p"), {"p": str(path)}
            ).first()
            if scene is None:
                raise SystemExit(
                    f"{path} is not registered; run the catalogue step first so the "
                    "scene's acquisition time and footprint are known"
                )
            scene = session.get(SARScene, scene.id)
        scene_id = scene.id

    log.info("detecting in %s (CFAR %s, pfa=%.0e)", path.name, params.model, params.pfa)
    try:
        detections = detect_in_scene(path, params=params)
    except Exception as exc:  # noqa: BLE001
        with session_scope() as session:
            session.query(SARScene).filter(SARScene.id == scene_id).update(
                {"status": "failed", "error": str(exc)[:1000]}, synchronize_session=False
            )
        raise

    with session_scope() as session:
        scene = session.get(SARScene, scene_id)
        written = insert_detections(session, scene, [d.to_dict() for d in detections])
        log.info("stored %d/%d detections", written, len(detections))
        summary = fuse_scene(session, scene)
        session.query(SARScene).filter(SARScene.id == scene_id).update(
            {"status": "complete", "error": None}, synchronize_session=False
        )
    log.info("scene %s done: %s", path.name, summary)


def main() -> None:
    parser = argparse.ArgumentParser(description="BoB Sentinel SAR worker")
    parser.add_argument("--days", type=int, default=7, help="Catalogue look-back")
    parser.add_argument("--limit", type=int, default=1, help="Max scenes to process")
    parser.add_argument("--scene", type=Path, help="Process an already-downloaded product")
    parser.add_argument("--catalogue-only", action="store_true",
                        help="Search and register scenes without downloading")
    args = parser.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)

    if args.scene:
        process_local_scene(args.scene)
        return

    if not (settings.cdse_username and settings.cdse_password):
        raise SystemExit("CDSE_USERNAME / CDSE_PASSWORD are not set — see .env.example")

    products = catalogue_scenes(args.days, args.limit)
    if not products:
        log.warning(
            "no Sentinel-1 products found. Note the constellation changed in 2026: "
            "S1A retired 2026-06-30, so only S1C/S1D acquire now."
        )
        return

    with session_scope() as session:
        for product in products:
            upsert_scene(
                session,
                product_id=product.id,
                product_name=product.name,
                platform=product.platform,
                product_type=product.product_type,
                acquired_start=product.acquired_start,
                acquired_end=product.acquired_end,
                footprint_wkt=product.footprint_wkt,
                status="catalogued",
            )
    if args.catalogue_only:
        log.info("catalogued %d scenes", len(products))
        return

    scenes_dir = Path(settings.data_dir) / "scenes"
    with CDSEClient(settings.cdse_username, settings.cdse_password) as client:
        for product in products[: args.limit]:
            path = client.download(product, scenes_dir)
            process_local_scene(path, product=product)


if __name__ == "__main__":
    main()
