"""CFAR detector behaviour.

The detector's value rests on two claims — it finds vessels, and it does not
invent them — so both are measured rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pytest

from bob_sentinel.sar.cfar import (
    CFARParams,
    cfar_detect,
    cluster_detections,
    detect,
)

TARGETS = [(100, 120, 40.0), (250, 300, 25.0), (320, 80, 60.0)]


def synthetic_scene(seed: int = 42, size: int = 400) -> np.ndarray:
    """Rayleigh sea clutter with three bright vessel-like targets."""
    rng = np.random.default_rng(seed)
    image = rng.rayleigh(scale=1.0, size=(size, size))
    for row, col, amplitude in TARGETS:
        image[row : row + 3, col : col + 2] += amplitude
    return image


@pytest.mark.parametrize("model", ["gaussian", "gamma"])
def test_finds_every_planted_target(model: str) -> None:
    candidates = detect(
        synthetic_scene(), CFARParams(pfa=1e-9, guard=4, train=12, model=model)
    )
    assert len(candidates) == len(TARGETS)
    found = {(c.row, c.col) for c in candidates}
    for row, col, _ in TARGETS:
        assert any(
            abs(r - row) <= 2 and abs(c - col) <= 2 for r, c in found
        ), f"target at ({row}, {col}) was missed by the {model} detector"


def test_small_targets_survive_morphology() -> None:
    """A 3x2 px vessel must not be erased before it is ever clustered.

    Regression: a default 3x3 binary opening annihilated every target smaller
    than the structuring element, which at 10 m GRD spacing is most of the
    Bay of Bengal fishing fleet.
    """
    params = CFARParams(pfa=1e-9, guard=4, train=12, model="gamma")
    assert params.open_radius == 0
    assert detect(synthetic_scene(), params)


def test_pure_clutter_raises_no_alarms() -> None:
    """The lognormal and gamma models must hold their design false-alarm rate."""
    rng = np.random.default_rng(7)
    clutter = rng.rayleigh(scale=1.0, size=(600, 600))
    for model in ("lognormal", "gamma"):
        hits = cfar_detect(
            clutter, CFARParams(pfa=1e-9, guard=4, train=12, model=model, close_radius=0)
        )
        rate = hits.sum() / clutter.size
        assert rate <= 1e-5, f"{model} false-alarm rate {rate:.2e} far exceeds design"


def test_gaussian_model_overshoots_on_rayleigh_clutter() -> None:
    """Documents *why* gamma is the default rather than the Gaussian model.

    Rayleigh clutter is heavy-tailed relative to a normal, so a Gaussian CFAR
    systematically under-estimates the tail and over-calls.
    """
    rng = np.random.default_rng(7)
    clutter = rng.rayleigh(scale=1.0, size=(600, 600))
    gaussian = cfar_detect(
        clutter, CFARParams(pfa=1e-6, guard=4, train=12, model="gaussian", close_radius=0)
    ).sum()
    gamma = cfar_detect(
        clutter, CFARParams(pfa=1e-6, guard=4, train=12, model="gamma", close_radius=0)
    ).sum()
    assert gaussian > gamma


def test_mask_suppresses_land() -> None:
    """Bright land must neither be detected nor poison nearby sea thresholds."""
    image = synthetic_scene()
    image[:60, :] += 500.0  # a very bright "coastline" strip
    mask = np.ones(image.shape, dtype=bool)
    mask[:60, :] = False

    candidates = detect(image, CFARParams(pfa=1e-9, guard=4, train=12), mask=mask)
    assert all(c.row >= 60 for c in candidates), "a detection was returned on masked land"
    assert len(candidates) == len(TARGETS)


def test_rejects_non_2d_input() -> None:
    with pytest.raises(ValueError, match="2-D"):
        cfar_detect(np.zeros((4, 4, 3)))


def test_rejects_mismatched_mask() -> None:
    with pytest.raises(ValueError, match="mask shape"):
        cfar_detect(np.zeros((10, 10)), mask=np.ones((5, 5), dtype=bool))


@pytest.mark.parametrize(
    "kwargs",
    [{"pfa": 0.0}, {"pfa": 1.5}, {"train": 0}, {"model": "weibull"}],
)
def test_invalid_params_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        CFARParams(**kwargs)


def test_area_filter_bounds_candidates() -> None:
    image = synthetic_scene()
    params = CFARParams(pfa=1e-9, guard=4, train=12, min_area_px=50)
    hits = cfar_detect(image, params)
    assert cluster_detections(image, hits, params) == []


def test_length_estimate_uses_pixel_spacing() -> None:
    candidates = detect(synthetic_scene(), CFARParams(pfa=1e-9, guard=4, train=12))
    # Targets are 3 px tall at 10 m spacing -> ~30 m.
    assert candidates[0].length_m(10.0) == pytest.approx(30.0, abs=10.0)


def test_empty_hitmap_clusters_to_nothing() -> None:
    image = np.zeros((50, 50))
    assert cluster_detections(image, np.zeros((50, 50), dtype=bool)) == []
