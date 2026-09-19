"""Reading Sentinel-1 GRD rasters, masking, and pixel -> lon/lat conversion.

A GRD IW product is a zip containing per-polarisation GeoTIFFs under
``measurement/``.  Those are *ungeocoded* detected amplitude with GCP-based
georeferencing, so ``rasterio`` exposes a GCP list rather than a clean affine
transform.  We build an approximate transform from the GCPs
(``rasterio.transform.from_gcps``), which is good to a few tens of metres over
open water — adequate for a 500 m AIS match gate, and far cheaper than a full
SNAP terrain-correction pass.  For metre-accurate geolocation, pre-process with
SNAP/pyroSAR and feed the terrain-corrected GeoTIFF in instead; this module
reads either.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

try:  # pragma: no cover - exercised implicitly by the raster tests
    import rasterio
    from rasterio.transform import from_gcps
    from rasterio.warp import transform as warp_transform

    HAS_RASTERIO = True
except ImportError:  # pragma: no cover
    HAS_RASTERIO = False


class RasterError(RuntimeError):
    pass


@dataclass
class SceneRaster:
    """An opened SAR band plus everything needed to geolocate its pixels."""

    data: np.ndarray
    transform: object  # affine.Affine
    crs: object
    pixel_spacing_m: float
    path: str
    polarisation: str | None = None
    row_offset: int = 0
    col_offset: int = 0

    def pixel_to_lonlat(self, rows, cols) -> tuple[np.ndarray, np.ndarray]:
        """Convert pixel indices (window-relative) to EPSG:4326 lon/lat."""
        if not HAS_RASTERIO:
            raise RasterError("rasterio is required for coordinate conversion")
        rows = np.atleast_1d(np.asarray(rows, dtype=np.float64)) + self.row_offset
        cols = np.atleast_1d(np.asarray(cols, dtype=np.float64)) + self.col_offset
        # +0.5 puts us at the pixel centre rather than its upper-left corner.
        xs, ys = rasterio.transform.xy(self.transform, rows, cols, offset="center")
        xs = np.atleast_1d(np.asarray(xs, dtype=np.float64))
        ys = np.atleast_1d(np.asarray(ys, dtype=np.float64))
        crs_str = str(self.crs) if self.crs else "EPSG:4326"
        if crs_str.upper().replace(" ", "") in {"EPSG:4326", "OGC:CRS84"}:
            return xs, ys
        lons, lats = warp_transform(self.crs, "EPSG:4326", xs.tolist(), ys.tolist())
        return np.asarray(lons), np.asarray(lats)


def find_measurement_tiffs(product: str | Path) -> list[str]:
    """List the ``measurement/*.tiff`` members of a .SAFE dir or .zip product."""
    product = Path(product)
    if product.is_dir():
        return sorted(str(p) for p in product.glob("measurement/*.tif*"))
    if product.suffix.lower() == ".zip":
        with zipfile.ZipFile(product) as archive:
            return sorted(
                name
                for name in archive.namelist()
                if "/measurement/" in name and name.lower().endswith((".tiff", ".tif"))
            )
    raise RasterError(f"not a Sentinel-1 product: {product}")


def polarisation_of(tiff_name: str) -> str | None:
    """Sentinel-1 measurement filenames embed the polarisation, e.g. ...-vv-..."""
    stem = Path(tiff_name).name.lower()
    for pol in ("vv", "vh", "hh", "hv"):
        if f"-{pol}-" in stem:
            return pol.upper()
    return None


def _rasterio_uri(product: Path, member: str) -> str:
    if product.is_dir():
        return member
    return f"zip://{product}!/{member}"


def open_scene(
    product: str | Path,
    *,
    polarisation: str = "VV",
    window: tuple[int, int, int, int] | None = None,
    max_pixels: int | None = 80_000_000,
) -> SceneRaster:
    """Open one polarisation of a GRD product.

    ``window`` is (row_off, col_off, height, width); without it the whole band
    is read, which for a full IW GRD scene is ~25k x 17k px (~3.5 GB as float64).
    ``max_pixels`` guards against that by refusing an oversized read — tile the
    scene with :func:`iter_windows` instead.
    """
    if not HAS_RASTERIO:
        raise RasterError("rasterio is not installed")

    product = Path(product)
    if product.suffix.lower() in {".tif", ".tiff"}:
        members, uri_base = [str(product)], None
    else:
        members = find_measurement_tiffs(product)
        uri_base = product
    if not members:
        raise RasterError(f"no measurement rasters found in {product}")

    chosen = next(
        (m for m in members if polarisation_of(m) == polarisation.upper()), members[0]
    )
    uri = chosen if uri_base is None else _rasterio_uri(uri_base, chosen)

    with rasterio.open(uri) as src:
        if window is None:
            if max_pixels is not None and src.width * src.height > max_pixels:
                raise RasterError(
                    f"scene is {src.width}x{src.height} px "
                    f"(> max_pixels={max_pixels}); read it in windows instead"
                )
            read_window = None
            row_off = col_off = 0
        else:
            row_off, col_off, height, width = window
            read_window = rasterio.windows.Window(col_off, row_off, width, height)

        data = src.read(1, window=read_window).astype(np.float32)
        transform, crs = _resolve_georeferencing(src)
        spacing = _pixel_spacing_m(src, transform, crs)

    return SceneRaster(
        data=data,
        transform=transform,
        crs=crs,
        pixel_spacing_m=spacing,
        path=str(uri),
        polarisation=polarisation_of(chosen),
        row_offset=row_off,
        col_offset=col_off,
    )


def _resolve_georeferencing(src) -> tuple[object, object]:
    """Prefer a real affine transform; fall back to fitting one to the GCPs."""
    gcps, gcp_crs = src.gcps
    if gcps:
        return from_gcps(gcps), (gcp_crs or src.crs or "EPSG:4326")
    if src.transform and not src.transform.is_identity:
        return src.transform, (src.crs or "EPSG:4326")
    raise RasterError(f"{src.name} has neither GCPs nor an affine transform")


def _pixel_spacing_m(src, transform, crs) -> float:
    """Ground pixel spacing in metres (IW GRDH is nominally 10 m)."""
    try:
        crs_obj = rasterio.crs.CRS.from_user_input(crs)
        if crs_obj.is_projected:
            return float(abs(transform.a))
        # Geographic: degrees -> metres at the scene's own latitude.
        centre_lat = transform.f + transform.e * (src.height / 2.0)
        metres_per_deg_lon = 111_320.0 * max(np.cos(np.radians(centre_lat)), 1e-6)
        return float(abs(transform.a) * metres_per_deg_lon)
    except Exception:  # pragma: no cover - defensive
        log.warning("could not derive pixel spacing; assuming 10 m GRDH")
        return 10.0


def iter_windows(
    height: int, width: int, tile: int = 4096, overlap: int = 128
):
    """Yield overlapping (row_off, col_off, h, w) tiles covering the scene.

    The overlap must exceed the CFAR outer half-width so a vessel near a tile
    edge still has a complete training ring; duplicates in the overlap are
    removed downstream by :func:`dedupe_candidates`.
    """
    if overlap >= tile:
        raise ValueError("overlap must be smaller than tile")
    step = tile - overlap

    def offsets(extent: int) -> list[int]:
        # Snap the final offset back from the edge so every tile is full-size.
        # A sliver tile narrower than the CFAR outer window has no complete
        # training ring, and would either crash or emit garbage thresholds.
        if extent <= tile:
            return [0]
        values = list(range(0, extent - tile + 1, step))
        if values[-1] != extent - tile:
            values.append(extent - tile)
        return values

    for row_off in offsets(height):
        for col_off in offsets(width):
            yield (row_off, col_off, min(tile, height - row_off), min(tile, width - col_off))


def to_intensity(amplitude: np.ndarray) -> np.ndarray:
    """GRD pixels are detected amplitude (DN); CFAR clutter models want intensity."""
    arr = np.asarray(amplitude, dtype=np.float64)
    return arr * arr


def to_db(intensity: np.ndarray, floor: float = 1e-12) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(intensity, dtype=np.float64), floor))


def valid_data_mask(data: np.ndarray) -> np.ndarray:
    """True where the pixel carries signal.

    GRD scenes are padded with exact zeros outside the swath; those borders
    would otherwise drag ring statistics down and manufacture detections at
    the swath edge.
    """
    arr = np.asarray(data)
    return np.isfinite(arr) & (arr > 0)


def dedupe_candidates(records: list[dict], min_separation_m: float = 100.0) -> list[dict]:
    """Drop near-duplicate detections produced by overlapping tiles.

    Records need ``lon``, ``lat`` and ``snr_db``; the strongest of each cluster
    is kept.  O(n^2) on purpose — a single scene yields hundreds of candidates,
    not millions, and this keeps the dependency surface at zero.
    """
    kept: list[dict] = []
    deg = min_separation_m / 111_320.0
    for record in sorted(records, key=lambda r: r.get("snr_db", 0.0), reverse=True):
        lon, lat = record["lon"], record["lat"]
        scale = max(np.cos(np.radians(lat)), 1e-6)
        if any(
            abs(lat - k["lat"]) < deg and abs(lon - k["lon"]) * scale < deg
            for k in kept
        ):
            continue
        kept.append(record)
    return kept
