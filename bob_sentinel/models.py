"""PostGIS-backed ORM models.

Geometry columns are all EPSG:4326 (lon/lat).  Distance work is done with
``geography`` casts or ``ST_DWithin(..., use_spheroid)`` so metre thresholds
mean metres, not degrees.
"""

from __future__ import annotations

from datetime import datetime

from geoalchemy2 import Geometry
from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Region(Base):
    """Maritime boundary polygon — the Bangladesh EEZ, MPAs, territorial sea."""

    __tablename__ = "regions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(160), nullable=False)
    kind: Mapped[str] = mapped_column(String(40), nullable=False, default="eez")
    mrgid: Mapped[int | None] = mapped_column(Integer, index=True)
    source: Mapped[str | None] = mapped_column(String(200))
    geom: Mapped[object] = mapped_column(
        Geometry("MULTIPOLYGON", srid=4326, spatial_index=True), nullable=False
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (UniqueConstraint("name", "kind", name="uq_region_name_kind"),)


class Vessel(Base):
    """Identity rolled up per MMSI from AIS static messages."""

    __tablename__ = "vessels"

    mmsi: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    imo: Mapped[int | None] = mapped_column(BigInteger, index=True)
    call_sign: Mapped[str | None] = mapped_column(String(16))
    name: Mapped[str | None] = mapped_column(String(128), index=True)
    ship_type: Mapped[int | None] = mapped_column(Integer)
    flag: Mapped[str | None] = mapped_column(String(8))
    length_m: Mapped[float | None] = mapped_column(Float)
    width_m: Mapped[float | None] = mapped_column(Float)
    destination: Mapped[str | None] = mapped_column(String(64))
    first_seen: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_seen: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), index=True
    )

    positions: Mapped[list[AISPosition]] = relationship(back_populates="vessel")


class AISPosition(Base):
    """One decoded AIS position report."""

    __tablename__ = "ais_positions"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    mmsi: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("vessels.mmsi", ondelete="CASCADE"), nullable=False
    )
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    geom: Mapped[object] = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=True), nullable=False
    )
    sog_kn: Mapped[float | None] = mapped_column(Float)
    cog_deg: Mapped[float | None] = mapped_column(Float)
    heading_deg: Mapped[float | None] = mapped_column(Float)
    rot: Mapped[float | None] = mapped_column(Float)
    nav_status: Mapped[int | None] = mapped_column(Integer)
    source: Mapped[str] = mapped_column(String(32), default="aisstream")

    vessel: Mapped[Vessel] = relationship(back_populates="positions")

    __table_args__ = (
        UniqueConstraint("mmsi", "ts", name="uq_ais_mmsi_ts"),
        Index("ix_ais_ts_mmsi", "ts", "mmsi"),
    )


class SARScene(Base):
    """A Sentinel-1 GRD product that has been (or is being) processed."""

    __tablename__ = "sar_scenes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    product_id: Mapped[str] = mapped_column(String(80), unique=True, nullable=False)
    product_name: Mapped[str] = mapped_column(String(200), nullable=False)
    platform: Mapped[str | None] = mapped_column(String(24))  # S1C / S1D
    mode: Mapped[str | None] = mapped_column(String(8))  # IW / EW
    product_type: Mapped[str | None] = mapped_column(String(16))  # IW_GRDH_1S
    polarisation: Mapped[str | None] = mapped_column(String(16))
    orbit_direction: Mapped[str | None] = mapped_column(String(16))
    acquired_start: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    acquired_end: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False
    )
    footprint: Mapped[object] = mapped_column(
        Geometry("MULTIPOLYGON", srid=4326, spatial_index=True), nullable=False
    )
    local_path: Mapped[str | None] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(24), default="catalogued", index=True)
    detector: Mapped[str | None] = mapped_column(String(40))
    detector_params: Mapped[dict | None] = mapped_column(JSONB)
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    detections: Mapped[list[Detection]] = relationship(
        back_populates="scene", cascade="all, delete-orphan"
    )

    @property
    def acquisition_midpoint(self) -> datetime:
        """The instant AIS tracks are interpolated to for matching."""
        return self.acquired_start + (self.acquired_end - self.acquired_start) / 2


class Detection(Base):
    """A candidate vessel returned by the SAR detector, plus its match verdict."""

    __tablename__ = "detections"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    scene_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("sar_scenes.id", ondelete="CASCADE"), nullable=False
    )
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    geom: Mapped[object] = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=True), nullable=False
    )
    row: Mapped[int | None] = mapped_column(Integer)
    col: Mapped[int | None] = mapped_column(Integer)
    peak_db: Mapped[float | None] = mapped_column(Float)
    snr_db: Mapped[float | None] = mapped_column(Float)
    area_px: Mapped[int | None] = mapped_column(Integer)
    length_m: Mapped[float | None] = mapped_column(Float)
    confidence: Mapped[float | None] = mapped_column(Float)

    # fusion verdict
    is_dark: Mapped[bool | None] = mapped_column(Boolean, index=True)
    matched_mmsi: Mapped[int | None] = mapped_column(BigInteger, index=True)
    match_distance_m: Mapped[float | None] = mapped_column(Float)
    match_notes: Mapped[dict | None] = mapped_column(JSONB)

    scene: Mapped[SARScene] = relationship(back_populates="detections")

    __table_args__ = (
        UniqueConstraint("scene_id", "row", "col", name="uq_detection_scene_pixel"),
    )


class Anomaly(Base):
    """An AIS-behaviour flag: gap, speed jump, loitering, encounter, dark SAR."""

    __tablename__ = "anomalies"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    mmsi: Mapped[int | None] = mapped_column(BigInteger, index=True)
    counterpart_mmsi: Mapped[int | None] = mapped_column(BigInteger)
    detection_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("detections.id", ondelete="CASCADE")
    )
    start_ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    end_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    geom: Mapped[object | None] = mapped_column(
        Geometry("POINT", srid=4326, spatial_index=True)
    )
    score: Mapped[float | None] = mapped_column(Float)
    details: Mapped[dict | None] = mapped_column(JSONB)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        # NULLS NOT DISTINCT is essential here, not cosmetic: `counterpart_mmsi`
        # is NULL for every gap, speed-jump and loitering event, and under
        # default SQL semantics NULL != NULL — so a plain unique constraint
        # would let every re-run of the anomaly scan insert a fresh copy of
        # every finding it had already recorded.  Requires PostgreSQL 15+.
        UniqueConstraint(
            "kind",
            "mmsi",
            "start_ts",
            "counterpart_mmsi",
            name="uq_anomaly_natural",
            postgresql_nulls_not_distinct=True,
        ),
    )
