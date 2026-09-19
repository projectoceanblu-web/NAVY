"""Persistence helpers shared by the API, the ingester and the SAR worker."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Sequence
from datetime import datetime, timedelta

from sqlalchemy import func, select, text
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.orm import Session

from bob_sentinel.ingest.aisstream import AISRecord
from bob_sentinel.models import AISPosition, Anomaly, Detection, Region, SARScene, Vessel

log = logging.getLogger(__name__)


def wkt_point(lon: float, lat: float) -> str:
    return f"SRID=4326;POINT({lon} {lat})"


def upsert_ais_records(session: Session, records: Sequence[AISRecord]) -> tuple[int, int]:
    """Persist a batch of AIS records.

    Returns (positions_written, vessels_touched).  Vessels are upserted first
    because ``ais_positions.mmsi`` is a foreign key onto them; positions use
    ``ON CONFLICT DO NOTHING`` so a replayed frame is a no-op rather than an
    integrity error.
    """
    if not records:
        return (0, 0)

    # --- vessels: one row per MMSI, newest static data wins ---
    vessel_rows: dict[int, dict] = {}
    for record in records:
        row = vessel_rows.setdefault(
            record.mmsi,
            {"mmsi": record.mmsi, "first_seen": record.ts, "last_seen": record.ts},
        )
        row["first_seen"] = min(row["first_seen"], record.ts)
        row["last_seen"] = max(row["last_seen"], record.ts)
        for field in (
            "name", "imo", "call_sign", "ship_type", "length_m", "width_m", "destination",
        ):
            value = getattr(record, field)
            if value is not None:
                row[field] = value

    vessel_stmt = pg_insert(Vessel).values(list(vessel_rows.values()))
    update_cols = {
        "last_seen": func.greatest(
            Vessel.last_seen, vessel_stmt.excluded.last_seen
        ),
        "first_seen": func.least(Vessel.first_seen, vessel_stmt.excluded.first_seen),
    }
    for field in (
        "name", "imo", "call_sign", "ship_type", "length_m", "width_m", "destination",
    ):
        # COALESCE(new, old): never blank out identity we already learned.
        update_cols[field] = func.coalesce(
            getattr(vessel_stmt.excluded, field), getattr(Vessel, field)
        )
    session.execute(
        vessel_stmt.on_conflict_do_update(index_elements=[Vessel.mmsi], set_=update_cols)
    )

    # --- positions ---
    position_rows = [
        {
            "mmsi": r.mmsi,
            "ts": r.ts,
            "geom": wkt_point(r.lon, r.lat),
            "sog_kn": r.sog_kn,
            "cog_deg": r.cog_deg,
            "heading_deg": r.heading_deg,
            "rot": r.rot,
            "nav_status": r.nav_status,
            "source": "aisstream",
        }
        for r in records
        if r.has_position
    ]
    written = 0
    if position_rows:
        # De-duplicate within the batch: the unique index cannot resolve two
        # conflicting rows in a single INSERT ... ON CONFLICT statement.
        unique: dict[tuple[int, datetime], dict] = {
            (row["mmsi"], row["ts"]): row for row in position_rows
        }
        stmt = (
            pg_insert(AISPosition)
            .values(list(unique.values()))
            .on_conflict_do_nothing(index_elements=[AISPosition.mmsi, AISPosition.ts])
            # RETURNING yields exactly the rows that were actually inserted.
            # `rowcount` is -1 ("unknown") for a multi-row INSERT ... ON CONFLICT
            # under psycopg3, so it cannot be used to count real writes.
            .returning(AISPosition.id)
        )
        written = len(session.execute(stmt).all())
    return (written, len(vessel_rows))


def get_region(
    session: Session, *, name: str | None = None, mrgid: int | None = None
) -> Region | None:
    stmt = select(Region)
    if mrgid is not None:
        stmt = stmt.where(Region.mrgid == mrgid)
    if name is not None:
        stmt = stmt.where(Region.name == name)
    return session.execute(stmt.limit(1)).scalar_one_or_none()


def upsert_region(
    session: Session, *, name: str, kind: str, geom_wkt: str, mrgid: int | None = None,
    source: str | None = None,
) -> Region:
    """Insert or replace a boundary polygon, coercing POLYGON -> MULTIPOLYGON."""
    geom = func.ST_Multi(func.ST_GeomFromText(geom_wkt, 4326))
    stmt = (
        pg_insert(Region)
        .values(name=name, kind=kind, mrgid=mrgid, source=source, geom=geom)
        .on_conflict_do_update(
            constraint="uq_region_name_kind",
            set_={"geom": geom, "mrgid": mrgid, "source": source},
        )
        .returning(Region.id)
    )
    region_id = session.execute(stmt).scalar_one()
    session.flush()
    return session.get(Region, region_id)


def upsert_scene(session: Session, **fields) -> SARScene:
    """Register a SAR product, keyed on its CDSE product id."""
    footprint = fields.pop("footprint_wkt", None)
    values = dict(fields)
    if footprint:
        values["footprint"] = func.ST_Multi(func.ST_GeomFromText(footprint, 4326))
    stmt = (
        pg_insert(SARScene)
        .values(**values)
        .on_conflict_do_update(
            index_elements=[SARScene.product_id],
            set_={
                k: v
                for k, v in values.items()
                if k not in {"product_id", "created_at"}
            },
        )
        .returning(SARScene.id)
    )
    scene_id = session.execute(stmt).scalar_one()
    session.flush()
    return session.get(SARScene, scene_id)


def insert_detections(
    session: Session, scene: SARScene, detections: Iterable[dict]
) -> int:
    """Bulk-insert detections for a scene, ignoring pixel-level duplicates."""
    rows = [
        {
            "scene_id": scene.id,
            "ts": scene.acquisition_midpoint,
            "geom": wkt_point(d["lon"], d["lat"]),
            "row": d.get("row"),
            "col": d.get("col"),
            "peak_db": d.get("peak_db"),
            "snr_db": d.get("snr_db"),
            "area_px": d.get("area_px"),
            "length_m": d.get("length_m"),
            "confidence": d.get("confidence"),
        }
        for d in detections
    ]
    if not rows:
        return 0
    stmt = (
        pg_insert(Detection)
        .values(rows)
        .on_conflict_do_nothing(constraint="uq_detection_scene_pixel")
        .returning(Detection.id)  # see upsert_ais_records: rowcount is unreliable
    )
    return len(session.execute(stmt).all())


def insert_anomalies(session: Session, anomalies: Iterable[dict]) -> int:
    rows = []
    for anomaly in anomalies:
        row = dict(anomaly)
        lon, lat = row.pop("lon", None), row.pop("lat", None)
        if lon is not None and lat is not None:
            row["geom"] = wkt_point(lon, lat)
        rows.append(row)
    if not rows:
        return 0
    stmt = (
        pg_insert(Anomaly)
        .values(rows)
        .on_conflict_do_nothing(constraint="uq_anomaly_natural")
        .returning(Anomaly.id)  # see upsert_ais_records: rowcount is unreliable
    )
    return len(session.execute(stmt).all())


def positions_near_time(
    session: Session, when: datetime, *, window_s: int = 1800
) -> list[AISPosition]:
    """Every AIS position within +/- ``window_s`` of an instant.

    This is the candidate pool for SAR matching: interpolation needs the fixes
    bracketing the acquisition, not the whole history.
    """
    stmt = text(
        """
        SELECT id, mmsi, ts, ST_X(geom::geometry) AS lon, ST_Y(geom::geometry) AS lat,
               sog_kn, cog_deg, heading_deg
        FROM ais_positions
        WHERE ts BETWEEN :lo AND :hi
        ORDER BY mmsi, ts
        """
    )
    return list(
        session.execute(
            stmt,
            {
                "lo": when - timedelta(seconds=window_s),
                "hi": when + timedelta(seconds=window_s),
            },
        ).mappings()
    )
