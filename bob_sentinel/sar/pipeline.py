"""Scene -> geolocated detections.

Tiles a GRD band, runs CA-CFAR per tile, converts surviving candidates to
lon/lat and de-duplicates across tile overlaps.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from bob_sentinel.sar.cfar import CFARParams, cfar_detect, cluster_detections
from bob_sentinel.sar.raster import (
    dedupe_candidates,
    iter_windows,
    open_scene,
    to_intensity,
    valid_data_mask,
)

log = logging.getLogger(__name__)

#: Tile overlap must exceed the CFAR outer half-width so edge vessels keep a
#: complete training ring.  4x is comfortable for the default guard+train=32.
_MIN_OVERLAP_FACTOR = 4


@dataclass
class SceneDetection:
    """A detection in geographic coordinates, ready to persist."""

    lon: float
    lat: float
    row: int
    col: int
    area_px: int
    peak_db: float
    snr_db: float
    length_m: float
    confidence: float

    def to_dict(self) -> dict:
        return asdict(self)


def _confidence(snr_db: float, area_px: int) -> float:
    """Heuristic 0-1 score.

    Deliberately simple and monotone in SNR and size: this is a screening
    prior, not a calibrated probability.  A learned detector (xView3 reference
    or AI2 sar_vessel_detect) replaces this with a real objectness score.
    """
    snr_term = np.clip((snr_db - 6.0) / 24.0, 0.0, 1.0)
    area_term = np.clip(np.log10(max(area_px, 1)) / 2.5, 0.0, 1.0)
    return float(np.clip(0.65 * snr_term + 0.35 * area_term, 0.0, 1.0))


def detect_in_scene(
    product: str | Path,
    *,
    polarisation: str = "VV",
    params: CFARParams | None = None,
    tile: int = 4096,
    overlap: int | None = None,
    max_detections: int = 20000,
    aoi_mask_fn=None,
) -> list[SceneDetection]:
    """Run the full detector over a product and return geolocated detections.

    ``aoi_mask_fn`` optionally takes (lon_array, lat_array) and returns a
    boolean keep-mask — used to drop detections on land or outside the EEZ.
    """
    params = params or CFARParams()
    required = params.outer * _MIN_OVERLAP_FACTOR
    if overlap is None:
        overlap = min(required, tile // 4)
    elif overlap < params.outer:
        raise ValueError(
            f"overlap={overlap} px is smaller than the CFAR outer window "
            f"({params.outer} px); edge vessels would lose their training ring"
        )

    # A metadata-only open to learn the raster's dimensions.
    probe = open_scene(product, polarisation=polarisation, window=(0, 0, 1, 1))
    import rasterio

    with rasterio.open(probe.path) as src:
        height, width = src.height, src.width
    log.info("scene %s is %d x %d px", Path(str(product)).name, height, width)

    records: list[dict] = []
    for row_off, col_off, win_h, win_w in iter_windows(height, width, tile, overlap):
        scene = open_scene(
            product,
            polarisation=polarisation,
            window=(row_off, col_off, win_h, win_w),
            max_pixels=None,
        )
        intensity = to_intensity(scene.data)
        mask = valid_data_mask(scene.data)
        if mask.sum() < (params.outer * 2 + 1) ** 2:
            continue  # tile is essentially all no-data padding

        hits = cfar_detect(intensity, params, mask=mask)
        candidates = cluster_detections(intensity, hits, params)
        if not candidates:
            continue

        rows = np.array([c.row for c in candidates])
        cols = np.array([c.col for c in candidates])
        lons, lats = scene.pixel_to_lonlat(rows, cols)

        keep = np.ones(len(candidates), dtype=bool)
        if aoi_mask_fn is not None:
            keep = np.asarray(aoi_mask_fn(lons, lats), dtype=bool)

        for i, candidate in enumerate(candidates):
            if not keep[i]:
                continue
            records.append(
                {
                    "lon": float(lons[i]),
                    "lat": float(lats[i]),
                    "row": int(candidate.row + row_off),
                    "col": int(candidate.col + col_off),
                    "area_px": candidate.area_px,
                    "peak_db": float(10.0 * np.log10(max(candidate.peak, 1e-12))),
                    "snr_db": candidate.snr_db,
                    "length_m": candidate.length_m(scene.pixel_spacing_m),
                    "confidence": _confidence(candidate.snr_db, candidate.area_px),
                }
            )
        if len(records) > max_detections * 4:
            log.warning(
                "scene produced >%d raw candidates; the CFAR pfa is probably "
                "too loose for this sea state",
                len(records),
            )
            break

    deduped = dedupe_candidates(records, min_separation_m=100.0)
    deduped.sort(key=lambda r: r["snr_db"], reverse=True)
    return [SceneDetection(**r) for r in deduped[:max_detections]]


def scene_timestamp(start: datetime, end: datetime) -> datetime:
    """The instant AIS is interpolated to: the acquisition midpoint."""
    return start + (end - start) / 2
