from __future__ import annotations

import numpy as np
import pytest

from docscanner.data.degrade import (
    DegradationConfig,
    DegradationPipeline,
    auto_exposure,
    blur,
    illumination_field,
    jpeg,
    photometric,
    random_document_quad,
    sensor_noise,
    soft_shadow,
    to_float,
    to_uint8,
    warp_onto_background,
)
from docscanner.utils.geometry import quad_is_convex

OPS = [photometric, illumination_field, soft_shadow, blur, sensor_noise, jpeg,
       auto_exposure]


@pytest.fixture
def image() -> np.ndarray:
    rng = np.random.default_rng(0)
    img = np.full((96, 128, 3), 0.85, np.float32)
    img[30:60, 20:100] = 0.15
    return np.clip(img + rng.normal(0, 0.01, img.shape).astype(np.float32), 0, 1)


@pytest.mark.parametrize("op", OPS)
def test_every_operator_stays_in_range_and_shape(op, image):
    rng = np.random.default_rng(1)
    out = op(image, rng, DegradationConfig())
    assert out.shape == image.shape
    assert out.dtype == np.float32
    assert out.min() >= -1e-5 and out.max() <= 1.0 + 1e-5


@pytest.mark.parametrize("op", OPS)
def test_every_operator_is_reproducible(op, image):
    a = op(image, np.random.default_rng(7), DegradationConfig())
    b = op(image, np.random.default_rng(7), DegradationConfig())
    assert np.array_equal(a, b)


def test_pipeline_is_seed_reproducible(image):
    pipe = DegradationPipeline(DegradationConfig())
    a, ta = pipe(image, np.random.default_rng(3))
    b, tb = pipe(image, np.random.default_rng(3))
    assert np.array_equal(a, b)
    assert ta.names == tb.names


def test_pipeline_trace_records_parameters(image):
    pipe = DegradationPipeline(DegradationConfig())
    _, trace = pipe(image, np.random.default_rng(5), keep_stages=True)
    assert trace.names, "no operators recorded"
    assert all(isinstance(n, str) for n in trace.names)
    assert trace.summary()
    assert len(trace.stages) >= 1


def test_uint8_float_round_trip():
    arr = np.array([[[0, 128, 255]]], np.uint8)
    assert np.array_equal(to_uint8(to_float(arr)), arr)


def test_jpeg_actually_degrades(image):
    cfg = DegradationConfig(jpeg_quality=(20, 20))
    out = jpeg(image, np.random.default_rng(0), cfg)
    assert np.abs(out - image).mean() > 1e-4


def test_auto_exposure_pulls_towards_target():
    dark = np.full((64, 64, 3), 0.08, np.float32)
    cfg = DegradationConfig(ae_target_mean=(0.6, 0.6), ae_strength=(1.0, 1.0))
    out = auto_exposure(dark, np.random.default_rng(0), cfg)
    assert out.mean() > dark.mean() * 2


@pytest.mark.parametrize("seed", range(12))
def test_random_quad_is_convex_and_inside_reasonable_bounds(seed):
    cfg = DegradationConfig()
    rng = np.random.default_rng(seed)
    quad = random_document_quad(640, 480, 0.72, rng, cfg)
    assert quad.shape == (4, 2)
    assert quad_is_convex(quad)
    assert quad[:, 0].min() > -640 * 0.3
    assert quad[:, 0].max() < 640 * 1.3


def test_warp_onto_background_places_the_document(image):
    doc = (np.ones((80, 60, 3), np.float32) * 0.95)
    bg = np.zeros((200, 240, 3), np.float32)
    quad = np.array([[40, 30], [190, 45], [180, 170], [30, 150]], np.float32)
    cfg = DegradationConfig(p_drop_shadow=0.0, p_paper_texture=0.0, p_page_curl=0.0)
    out, h = warp_onto_background(doc, bg, quad, np.random.default_rng(0), cfg)
    assert out.shape == bg.shape
    assert out[100, 110].mean() > 0.5
    assert out[5, 5].mean() < 0.2
    assert h.shape == (3, 3)


def test_hard_config_is_actually_harsher(image):
    mild = DegradationPipeline(DegradationConfig.mild())
    hard = DegradationPipeline(DegradationConfig.hard())
    d_mild, d_hard = [], []
    for s in range(6):
        a, _ = mild(image, np.random.default_rng(s))
        b, _ = hard(image, np.random.default_rng(s))
        d_mild.append(np.abs(a - image).mean())
        d_hard.append(np.abs(b - image).mean())
    assert np.mean(d_hard) > np.mean(d_mild)
