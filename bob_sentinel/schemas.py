"""Pydantic response models for the API."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class HealthComponent(BaseModel):
    ok: bool
    detail: str | None = None


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    components: dict[str, HealthComponent]


class VesselOut(BaseModel):
    mmsi: int
    name: str | None = None
    imo: int | None = None
    call_sign: str | None = None
    ship_type: int | None = None
    length_m: float | None = None
    last_seen: datetime | None = None


class PositionOut(BaseModel):
    mmsi: int
    ts: datetime
    lon: float
    lat: float
    sog_kn: float | None = None
    cog_deg: float | None = None


class DetectionOut(BaseModel):
    id: int
    scene_id: int
    ts: datetime
    lon: float
    lat: float
    snr_db: float | None = None
    length_m: float | None = None
    confidence: float | None = None
    is_dark: bool | None = None
    matched_mmsi: int | None = None
    match_distance_m: float | None = None
    status: str = Field(
        description="matched | dark | indeterminate — indeterminate means AIS "
        "coverage was too thin to interpret the absence of a match."
    )


class SceneOut(BaseModel):
    id: int
    product_id: str
    product_name: str
    platform: str | None = None
    acquired_start: datetime
    acquired_end: datetime
    status: str
    detection_count: int = 0


class AnomalyOut(BaseModel):
    id: int
    kind: str
    mmsi: int | None = None
    counterpart_mmsi: int | None = None
    start_ts: datetime
    end_ts: datetime | None = None
    lon: float | None = None
    lat: float | None = None
    score: float | None = None
    details: dict[str, Any] | None = None


class GeoJSONFeature(BaseModel):
    type: Literal["Feature"] = "Feature"
    geometry: dict[str, Any]
    properties: dict[str, Any]


class GeoJSONFeatureCollection(BaseModel):
    type: Literal["FeatureCollection"] = "FeatureCollection"
    features: list[GeoJSONFeature]


class StatsOut(BaseModel):
    vessels: int
    positions: int
    scenes: int
    detections: int
    dark_detections: int
    indeterminate_detections: int
    anomalies: int
    latest_ais_ts: datetime | None = None
    latest_scene_ts: datetime | None = None
