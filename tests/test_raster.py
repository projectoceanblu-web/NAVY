"""Raster tiling, geolocation and the detection pipeline, on a real GeoTIFF."""

from __future__ import annotations

import numpy as np
import pytest

rasterio = pytest.importorskip("rasterio")
from rasterio.transform import from_origin  # noqa: E402

from bob_sentinel.sar.cfar import CFARParams  # noqa: E402
from bob_sentinel.sar.pipeline import _confidence, detect_in_scene  # noqa: E402
from bob_sentinel.sar.raster import (  # noqa: E402
    RasterError,
    dedupe_candidates,
    iter_windows,
    open_scene,
    polarisation_of,
    to_db,
    to_intensity,
    valid_data_mask,
)

TARGETS = [(100, 120), (250, 300), (320, 80)]


@pytest.fixture
def grd_tiff(tmp_path):
    """A small geocoded scene: Rayleigh sea clutter plus three bright targets."""
    rng = np.random.default_rng(42)
    size = 400
    data = rng.rayleigh(scale=1.0, size=(size, size)).astype(np.float32)
    for row, col in TARGETS:
        data[row : row + 3, col : col + 2] += 40.0
    data[:20, :] = 0.0  # no-data padding outside the swath

    path = tmp_path / "S1D_IW_GRDH_1SDV_20260103T114217-vv-001.tiff"
    # ~10 m pixels expressed in degrees near 21 N.
    transform = from_origin(90.0, 21.5, 0.0001, 0.0001)
    with rasterio.open(
        path, "w", driver="GTiff", height=size, width=size, count=1,
        dtype="float32", crs="EPSG:4326", transform=transform,
    ) as dst:
        dst.write(data, 1)
    return path


# --- tiling ----------------------------------------------------------------


def test_every_tile_is_full_size() -> None:
    """A sliver tile has no complete CFAR training ring.

    Regression: naive striding left a 64 px edge tile on a 8000 px scene.
    """
    tiles = list(iter_windows(10_000, 8_000, tile=4096, overlap=128))
    assert {(h, w) for _, _, h, w in tiles} == {(4096, 4096)}


def test_tiles_cover_the_whole_scene() -> None:
    height, width, tile = 5000, 3000, 2048
    covered = np.zeros((height, width), dtype=bool)
    for row, col, h, w in iter_windows(height, width, tile=tile, overlap=128):
        covered[row : row + h, col : col + w] = True
    assert covered.all(), "tiling left a gap in the scene"


def test_a_scene_smaller_than_one_tile_yields_one_tile() -> None:
    assert list(iter_windows(500, 500, tile=4096, overlap=128)) == [(0, 0, 500, 500)]


def test_overlap_must_be_smaller_than_the_tile() -> None:
    with pytest.raises(ValueError, match="overlap"):
        list(iter_windows(1000, 1000, tile=256, overlap=256))


# --- helpers ---------------------------------------------------------------


def test_polarisation_is_read_from_the_filename() -> None:
    assert polarisation_of("s1d-iw-grd-vv-20260103.tiff") == "VV"
    assert polarisation_of("s1d-iw-grd-vh-20260103.tiff") == "VH"
    assert polarisation_of("mystery.tiff") is None


def test_amplitude_converts_to_intensity_and_db() -> None:
    assert to_intensity([2.0, 3.0]).tolist() == [4.0, 9.0]
    assert to_db(np.array([1.0])).tolist() == [0.0]
    assert np.isfinite(to_db(np.array([0.0]))).all(), "log of zero must be floored"


def test_no_data_padding_is_excluded_from_the_valid_mask() -> None:
    data = np.array([[0.0, 1.0], [np.nan, 2.0]])
    assert valid_data_mask(data).tolist() == [[False, True], [False, True]]


def test_dedupe_keeps_the_strongest_of_a_cluster() -> None:
    records = [
        {"lon": 90.0, "lat": 21.0, "snr_db": 12.0},
        {"lon": 90.0000005, "lat": 21.0000005, "snr_db": 9.0},  # same vessel
        {"lon": 90.5, "lat": 21.0, "snr_db": 8.0},              # genuinely separate
    ]
    kept = dedupe_candidates(records, min_separation_m=100.0)
    assert [r["snr_db"] for r in kept] == [12.0, 8.0]


# --- opening & geolocation -------------------------------------------------


def test_open_scene_reads_a_window(grd_tiff) -> None:
    scene = open_scene(grd_tiff, window=(50, 50, 100, 100))
    assert scene.data.shape == (100, 100)
    assert scene.row_offset == 50 and scene.col_offset == 50
    assert scene.polarisation == "VV"


def test_pixel_spacing_is_converted_to_metres(grd_tiff) -> None:
    """0.0001 degrees of longitude at 21 N is ~10 m — GRDH's nominal spacing."""
    scene = open_scene(grd_tiff, window=(0, 0, 10, 10))
    assert scene.pixel_spacing_m == pytest.approx(10.4, abs=1.0)


def test_window_relative_pixels_geolocate_absolutely(grd_tiff) -> None:
    """A windowed read must still report true coordinates, not tile-local ones."""
    whole = open_scene(grd_tiff)
    windowed = open_scene(grd_tiff, window=(100, 100, 50, 50))
    lon_a, lat_a = whole.pixel_to_lonlat([120], [130])
    lon_b, lat_b = windowed.pixel_to_lonlat([20], [30])
    assert lon_a[0] == pytest.approx(lon_b[0])
    assert lat_a[0] == pytest.approx(lat_b[0])


def test_oversized_read_is_refused(grd_tiff) -> None:
    with pytest.raises(RasterError, match="max_pixels"):
        open_scene(grd_tiff, max_pixels=100)


def test_missing_product_is_reported(tmp_path) -> None:
    with pytest.raises(RasterError):
        open_scene(tmp_path / "nope.SAFE")


# --- end-to-end pipeline ---------------------------------------------------


def test_pipeline_finds_targets_and_geolocates_them(grd_tiff) -> None:
    detections = detect_in_scene(
        grd_tiff, params=CFARParams(pfa=1e-9, guard=4, train=12), tile=256, overlap=64
    )
    assert len(detections) == len(TARGETS)
    for detection in detections:
        assert 89.9 < detection.lon < 90.1
        assert 21.4 < detection.lat < 21.6
        assert detection.snr_db > 6
        assert 0.0 <= detection.confidence <= 1.0
        assert detection.length_m > 0


def test_pipeline_does_not_detect_in_no_data_padding(grd_tiff) -> None:
    """Zero-fill outside the swath must not manufacture edge detections."""
    detections = detect_in_scene(
        grd_tiff, params=CFARParams(pfa=1e-9, guard=4, train=12), tile=256, overlap=64
    )
    assert all(d.row >= 20 for d in detections)


def test_pipeline_rejects_an_overlap_smaller_than_the_cfar_window(grd_tiff) -> None:
    params = CFARParams(pfa=1e-9, guard=8, train=24)  # outer = 32
    with pytest.raises(ValueError, match="training ring"):
        detect_in_scene(grd_tiff, params=params, tile=256, overlap=16)


def test_aoi_mask_filters_detections(grd_tiff) -> None:
    detections = detect_in_scene(
        grd_tiff, params=CFARParams(pfa=1e-9, guard=4, train=12),
        tile=256, overlap=64, aoi_mask_fn=lambda lons, lats: np.zeros(len(lons), bool),
    )
    assert detections == []


def test_confidence_is_monotone_and_bounded() -> None:
    assert _confidence(6, 1) == 0.0
    assert _confidence(40, 500) == 1.0
    assert _confidence(12, 10) < _confidence(30, 10)
    assert _confidence(20, 5) < _confidence(20, 100)
