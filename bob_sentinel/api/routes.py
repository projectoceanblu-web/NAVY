"""API routes.

Endpoints are declared ``def`` (not ``async def``) so Starlette runs them in a
threadpool — the DB driver is sync, and blocking the event loop would stall the
AIS ingester sharing the process.

Geometry is returned as GeoJSON FeatureCollections so the Leaflet frontend can
drop a response straight into a layer.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query, Response
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from bob_sentinel import __version__
from bob_sentinel.config import Settings, get_settings
from bob_sentinel.db import get_db
from bob_sentinel.models import AISPosition, Anomaly, Detection, SARScene, Vessel
from bob_sentinel.schemas import (
    AnomalyOut,
    GeoJSONFeature,
    GeoJSONFeatureCollection,
    HealthComponent,
    HealthResponse,
    SceneOut,
    StatsOut,
)

log = logging.getLogger(__name__)
router = APIRouter()


def _point_feature(lon: float, lat: float, properties: dict) -> GeoJSONFeature:
    return GeoJSONFeature(
        geometry={"type": "Point", "coordinates": [lon, lat]}, properties=properties
    )


def _detection_status(is_dark: bool | None) -> str:
    if is_dark is True:
        return "dark"
    if is_dark is False:
        return "matched"
    return "indeterminate"


# --- health & stats --------------------------------------------------------


@router.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Liveness plus a per-credential readiness view.

    The service starts without credentials on purpose — a missing GFW token
    should not stop AIS ingestion — so this reports which subsystems are armed
    rather than failing outright.
    """
    components: dict[str, HealthComponent] = {}

    try:
        from bob_sentinel.db import get_engine

        with get_engine().connect() as connection:
            postgis = connection.execute(text("SELECT PostGIS_Version()")).scalar()
        components["database"] = HealthComponent(ok=True, detail=f"PostGIS {postgis}")
    except Exception as exc:  # noqa: BLE001
        components["database"] = HealthComponent(ok=False, detail=str(exc)[:200])

    for name, configured, hint in (
        ("cdse", bool(settings.cdse_username and settings.cdse_password),
         "set CDSE_USERNAME / CDSE_PASSWORD"),
        ("aisstream", bool(settings.aisstream_api_key), "set AISSTREAM_API_KEY"),
        ("gfw", bool(settings.gfw_api_token), "set GFW_API_TOKEN"),
    ):
        components[name] = HealthComponent(
            ok=configured, detail=None if configured else f"not configured — {hint}"
        )

    status = "ok" if components["database"].ok else "degraded"
    return HealthResponse(status=status, version=__version__, components=components)


@router.get("/api/stats", response_model=StatsOut, tags=["ops"])
def stats(db: Session = Depends(get_db)) -> StatsOut:
    """Row counts and data freshness — the fastest way to see if ingest is alive."""
    return StatsOut(
        vessels=db.scalar(select(func.count()).select_from(Vessel)) or 0,
        positions=db.scalar(select(func.count()).select_from(AISPosition)) or 0,
        scenes=db.scalar(select(func.count()).select_from(SARScene)) or 0,
        detections=db.scalar(select(func.count()).select_from(Detection)) or 0,
        dark_detections=db.scalar(
            select(func.count()).select_from(Detection).where(Detection.is_dark.is_(True))
        ) or 0,
        indeterminate_detections=db.scalar(
            select(func.count()).select_from(Detection).where(Detection.is_dark.is_(None))
        ) or 0,
        anomalies=db.scalar(select(func.count()).select_from(Anomaly)) or 0,
        latest_ais_ts=db.scalar(select(func.max(AISPosition.ts))),
        latest_scene_ts=db.scalar(select(func.max(SARScene.acquired_start))),
    )


# --- AIS -------------------------------------------------------------------


@router.get("/api/ais/latest", response_model=GeoJSONFeatureCollection, tags=["ais"])
def latest_ais(
    db: Session = Depends(get_db),
    minutes: int = Query(60, ge=1, le=10080, description="Look-back window"),
    limit: int = Query(5000, ge=1, le=50000),
) -> GeoJSONFeatureCollection:
    """The most recent position per vessel within the look-back window."""
    cutoff = datetime.now(UTC) - timedelta(minutes=minutes)
    # DISTINCT ON is the idiomatic PostgreSQL "latest row per group".
    rows = db.execute(
        text(
            """
            SELECT DISTINCT ON (p.mmsi)
                   p.mmsi, p.ts, ST_X(p.geom::geometry) AS lon,
                   ST_Y(p.geom::geometry) AS lat, p.sog_kn, p.cog_deg,
                   v.name, v.ship_type
            FROM ais_positions p
            LEFT JOIN vessels v ON v.mmsi = p.mmsi
            WHERE p.ts >= :cutoff
            ORDER BY p.mmsi, p.ts DESC
            LIMIT :limit
            """
        ),
        {"cutoff": cutoff, "limit": limit},
    ).mappings()

    features = [
        _point_feature(
            row["lon"],
            row["lat"],
            {
                "mmsi": row["mmsi"],
                "name": row["name"],
                "ts": row["ts"].isoformat(),
                "sog_kn": row["sog_kn"],
                "cog_deg": row["cog_deg"],
                "ship_type": row["ship_type"],
            },
        )
        for row in rows
    ]
    return GeoJSONFeatureCollection(features=features)


@router.get("/api/ais/track/{mmsi}", response_model=GeoJSONFeatureCollection, tags=["ais"])
def vessel_track(
    mmsi: int,
    db: Session = Depends(get_db),
    hours: int = Query(24, ge=1, le=720),
) -> GeoJSONFeatureCollection:
    """One vessel's track as a LineString (plus its endpoint as a Point)."""
    cutoff = datetime.now(UTC) - timedelta(hours=hours)
    rows = list(
        db.execute(
            text(
                """
                SELECT ts, ST_X(geom::geometry) AS lon, ST_Y(geom::geometry) AS lat,
                       sog_kn, cog_deg
                FROM ais_positions
                WHERE mmsi = :mmsi AND ts >= :cutoff
                ORDER BY ts
                """
            ),
            {"mmsi": mmsi, "cutoff": cutoff},
        ).mappings()
    )
    if not rows:
        raise HTTPException(404, f"no positions for MMSI {mmsi} in the last {hours}h")

    features: list[GeoJSONFeature] = []
    if len(rows) > 1:
        features.append(
            GeoJSONFeature(
                geometry={
                    "type": "LineString",
                    "coordinates": [[r["lon"], r["lat"]] for r in rows],
                },
                properties={"mmsi": mmsi, "positions": len(rows)},
            )
        )
    last = rows[-1]
    features.append(
        _point_feature(
            last["lon"], last["lat"],
            {"mmsi": mmsi, "ts": last["ts"].isoformat(), "sog_kn": last["sog_kn"]},
        )
    )
    return GeoJSONFeatureCollection(features=features)


# --- detections ------------------------------------------------------------


@router.get("/api/detections", response_model=GeoJSONFeatureCollection, tags=["sar"])
def detections(
    db: Session = Depends(get_db),
    status: str | None = Query(
        None, pattern="^(dark|matched|indeterminate)$",
        description="Filter by fusion verdict",
    ),
    scene_id: int | None = None,
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(5000, ge=1, le=50000),
) -> GeoJSONFeatureCollection:
    """SAR detections with their AIS-fusion verdict."""
    cutoff = datetime.now(UTC) - timedelta(days=days)
    stmt = (
        select(
            Detection.id, Detection.scene_id, Detection.ts,
            func.ST_X(Detection.geom).label("lon"),
            func.ST_Y(Detection.geom).label("lat"),
            Detection.snr_db, Detection.length_m, Detection.confidence,
            Detection.is_dark, Detection.matched_mmsi, Detection.match_distance_m,
        )
        .where(Detection.ts >= cutoff)
        .order_by(Detection.ts.desc())
        .limit(limit)
    )
    if scene_id is not None:
        stmt = stmt.where(Detection.scene_id == scene_id)
    if status == "dark":
        stmt = stmt.where(Detection.is_dark.is_(True))
    elif status == "matched":
        stmt = stmt.where(Detection.is_dark.is_(False))
    elif status == "indeterminate":
        stmt = stmt.where(Detection.is_dark.is_(None))

    features = [
        _point_feature(
            row.lon, row.lat,
            {
                "id": row.id,
                "scene_id": row.scene_id,
                "ts": row.ts.isoformat(),
                "snr_db": row.snr_db,
                "length_m": row.length_m,
                "confidence": row.confidence,
                "status": _detection_status(row.is_dark),
                "matched_mmsi": row.matched_mmsi,
                "match_distance_m": row.match_distance_m,
            },
        )
        for row in db.execute(stmt)
    ]
    return GeoJSONFeatureCollection(features=features)


@router.get("/api/scenes", response_model=list[SceneOut], tags=["sar"])
def scenes(db: Session = Depends(get_db), limit: int = Query(50, ge=1, le=500)) -> list[SceneOut]:
    counts = (
        select(Detection.scene_id, func.count().label("n"))
        .group_by(Detection.scene_id)
        .subquery()
    )
    rows = db.execute(
        select(SARScene, func.coalesce(counts.c.n, 0))
        .outerjoin(counts, counts.c.scene_id == SARScene.id)
        .order_by(SARScene.acquired_start.desc())
        .limit(limit)
    ).all()
    return [
        SceneOut(
            id=scene.id, product_id=scene.product_id, product_name=scene.product_name,
            platform=scene.platform, acquired_start=scene.acquired_start,
            acquired_end=scene.acquired_end, status=scene.status, detection_count=count,
        )
        for scene, count in rows
    ]


# --- anomalies & regions ---------------------------------------------------


@router.get("/api/anomalies", response_model=list[AnomalyOut], tags=["fusion"])
def anomalies(
    db: Session = Depends(get_db),
    kind: str | None = None,
    days: int = Query(30, ge=1, le=365),
    limit: int = Query(500, ge=1, le=5000),
) -> list[AnomalyOut]:
    cutoff = datetime.now(UTC) - timedelta(days=days)
    stmt = (
        select(
            Anomaly.id, Anomaly.kind, Anomaly.mmsi, Anomaly.counterpart_mmsi,
            Anomaly.start_ts, Anomaly.end_ts, Anomaly.score, Anomaly.details,
            func.ST_X(Anomaly.geom).label("lon"),
            func.ST_Y(Anomaly.geom).label("lat"),
        )
        .where(Anomaly.start_ts >= cutoff)
        .order_by(Anomaly.start_ts.desc())
        .limit(limit)
    )
    if kind:
        stmt = stmt.where(Anomaly.kind == kind)
    return [
        AnomalyOut(
            id=row.id, kind=row.kind, mmsi=row.mmsi,
            counterpart_mmsi=row.counterpart_mmsi, start_ts=row.start_ts,
            end_ts=row.end_ts, lon=row.lon, lat=row.lat, score=row.score,
            details=row.details,
        )
        for row in db.execute(stmt)
    ]


@router.get("/api/regions", response_model=GeoJSONFeatureCollection, tags=["geo"])
def regions(db: Session = Depends(get_db)) -> GeoJSONFeatureCollection:
    """Boundary polygons (EEZ, MPAs) as GeoJSON."""
    rows = db.execute(
        text(
            """
            SELECT id, name, kind, mrgid, source,
                   ST_AsGeoJSON(ST_SimplifyPreserveTopology(geom::geometry, 0.001)) AS gj
            FROM regions
            """
        )
    ).mappings()
    return GeoJSONFeatureCollection(
        features=[
            GeoJSONFeature(
                geometry=json.loads(row["gj"]),
                properties={
                    "id": row["id"], "name": row["name"], "kind": row["kind"],
                    "mrgid": row["mrgid"], "source": row["source"],
                },
            )
            for row in rows
        ]
    )


# --- GFW proxy -------------------------------------------------------------


@router.get("/api/gfw/tile/{z}/{x}/{y}", tags=["gfw"])
def gfw_tile(
    z: int, x: int, y: int,
    settings: Settings = Depends(get_settings),
    date_range: str | None = Query(None, alias="date-range"),
) -> Response:
    """Proxy GFW heatmap tiles.

    The token stays server-side: a token embedded in a browser tile URL is a
    published token, and GFW rate-limits per user across every token they hold.
    """
    if not settings.gfw_api_token:
        raise HTTPException(503, "GFW_API_TOKEN is not configured")
    from bob_sentinel.clients.gfw import GFWClient, GFWError

    try:
        with GFWClient(settings.gfw_api_token) as client:
            payload = client.get_tile(z, x, y, date_range=date_range)
    except GFWError as exc:
        raise HTTPException(502, str(exc)) from exc
    return Response(
        content=payload,
        media_type="application/x-protobuf",
        headers={"Cache-Control": "public, max-age=3600"},
    )
