"""Unit test core pipeline. Tidak butuh model ONNX: pakai predictor dummy."""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from app.core.compose import BackgroundSpec, BgMode, compose, decontaminate_colors
from app.core.engine import postprocess, preprocess
from app.core.models import DEFAULT_MODEL, MODELS, get_model
from app.core.pipeline import (
    BatchOptions,
    ProcessResult,
    collect_images,
    export_folder,
    export_zip,
    process_image,
    run_batch,
)
from app.core.refine import BrushMode, BrushStroke, RefineParams, apply_strokes, refine_alpha


class CirclePredictor:
    """Alpha = lingkaran di tengah gambar."""

    def predict_alpha(self, img: Image.Image) -> np.ndarray:
        w, h = img.size
        yy, xx = np.mgrid[0:h, 0:w]
        r = min(w, h) / 3
        return (((xx - w / 2) ** 2 + (yy - h / 2) ** 2) <= r * r).astype(np.float32)


class FailingPredictor:
    def predict_alpha(self, img):
        raise RuntimeError("boom")


@pytest.fixture
def rgb_img() -> Image.Image:
    rng = np.random.default_rng(0)
    return Image.fromarray(rng.integers(0, 255, (64, 80, 3), dtype=np.uint8), "RGB")


@pytest.fixture
def image_dir(tmp_path: Path, rgb_img: Image.Image) -> Path:
    d = tmp_path / "in"
    (d / "sub").mkdir(parents=True)
    rgb_img.save(d / "a.jpg")
    rgb_img.save(d / "sub" / "a.png")  # stem sama -> uji nama unik
    (d / "notes.txt").write_text("bukan gambar")
    return d


# ---------------------------------------------------------------- models/engine
def test_default_model_registered():
    assert DEFAULT_MODEL in MODELS
    with pytest.raises(ValueError):
        get_model("tidak-ada")


def test_preprocess_shape_and_dtype(rgb_img):
    spec = get_model("u2netp")
    x = preprocess(rgb_img, spec)
    assert x.shape == (1, 3, spec.input_size, spec.input_size)
    assert x.dtype == np.float32


@pytest.mark.parametrize("key", ["u2net", "birefnet-lite"])
def test_postprocess_range_and_size(key):
    spec = get_model(key)
    raw = np.random.default_rng(1).normal(size=(1, 1, 32, 32)).astype(np.float32)
    a = postprocess(raw, spec, (80, 64))
    assert a.shape == (64, 80)
    assert a.min() >= 0.0 and a.max() <= 1.0


# ---------------------------------------------------------------- refine
def test_refine_identity(rgb_img):
    alpha = CirclePredictor().predict_alpha(rgb_img)
    out = refine_alpha(np.asarray(rgb_img), alpha, RefineParams())
    np.testing.assert_allclose(out, alpha)


def test_edge_shift_grows_and_shrinks(rgb_img):
    rgb = np.asarray(rgb_img)
    alpha = CirclePredictor().predict_alpha(rgb_img)
    grown = refine_alpha(rgb, alpha, RefineParams(edge_shift=3)).sum()
    shrunk = refine_alpha(rgb, alpha, RefineParams(edge_shift=-3)).sum()
    assert shrunk < alpha.sum() < grown


def test_levels_and_guided_stay_in_range(rgb_img):
    rgb = np.asarray(rgb_img)
    alpha = np.linspace(0, 1, 64 * 80, dtype=np.float32).reshape(64, 80)
    out = refine_alpha(rgb, alpha, RefineParams(guided_radius=4, level_low=0.2,
                                                level_high=0.8, feather=1.5))
    assert out.dtype == np.float32
    assert 0.0 <= out.min() and out.max() <= 1.0


def test_brush_keep_and_erase():
    alpha = np.full((50, 50), 0.5, np.float32)
    keep = BrushStroke(BrushMode.KEEP, [(10, 10)], radius=5, hardness=1.0)
    erase = BrushStroke(BrushMode.ERASE, [(40, 40), (45, 45)], radius=3, hardness=1.0)
    out = apply_strokes(alpha, [keep, erase])
    assert out[10, 10] == pytest.approx(1.0)
    assert out[42, 42] == pytest.approx(0.0)
    assert out[25, 25] == pytest.approx(0.5)  # di luar stroke tidak berubah


# ---------------------------------------------------------------- compose
def test_compose_modes(rgb_img):
    rgb = np.asarray(rgb_img)
    alpha = CirclePredictor().predict_alpha(rgb_img)

    t = compose(rgb, alpha, BackgroundSpec(mode=BgMode.TRANSPARENT))
    assert t.mode == "RGBA" and t.size == rgb_img.size
    assert t.getpixel((0, 0))[3] == 0

    c = compose(rgb, alpha, BackgroundSpec(mode=BgMode.COLOR, color=(0, 255, 0)))
    assert c.mode == "RGB" and c.getpixel((0, 0)) == (0, 255, 0)

    bg = Image.new("RGB", (10, 30), (0, 0, 255))
    i = compose(rgb, alpha, BackgroundSpec(mode=BgMode.IMAGE, image=bg))
    assert i.size == rgb_img.size and i.getpixel((0, 0)) == (0, 0, 255)

    b = compose(rgb, alpha, BackgroundSpec(mode=BgMode.BLUR))
    assert b.mode == "RGB"

    with pytest.raises(ValueError):
        compose(rgb, alpha, BackgroundSpec(mode=BgMode.IMAGE, image=None))


def test_decontaminate_keeps_core(rgb_img):
    rgb = np.asarray(rgb_img)
    alpha = CirclePredictor().predict_alpha(rgb_img)
    fg = decontaminate_colors(rgb, alpha)
    cy, cx = 32, 40
    np.testing.assert_allclose(fg[cy, cx], rgb[cy, cx] / 255.0, atol=1e-6)


# ---------------------------------------------------------------- pipeline
def test_collect_images_filters_and_recurses(image_dir):
    found = collect_images([image_dir, image_dir / "a.jpg"])
    assert sorted(p.name for p in found) == ["a.jpg", "a.png"]  # unik, tanpa .txt


def test_process_image_reuses_alpha(rgb_img):
    alpha = CirclePredictor().predict_alpha(rgb_img)
    out, refined = process_image(rgb_img, FailingPredictor(), BatchOptions(), alpha=alpha)
    assert out.mode == "RGBA"
    np.testing.assert_allclose(refined, alpha)


def test_run_batch_continues_after_error_and_cancel(image_dir):
    files = collect_images([image_dir])
    res = run_batch(files, FailingPredictor(), BatchOptions())
    assert len(res) == 2 and not any(r.ok for r in res)

    seen = []
    res = run_batch(files, CirclePredictor(), BatchOptions(),
                    progress=lambda i, n, r: seen.append((i, n)))
    assert all(r.ok for r in res) and seen == [(1, 2), (2, 2)]


def test_export_zip_streaming_unique_names_and_errors(image_dir, tmp_path):
    files = collect_images([image_dir])
    ok = run_batch(files, CirclePredictor(), BatchOptions())

    def stream():
        yield from ok
        yield ProcessResult(source=Path("rusak.jpg"), error="file korup")

    zp = tmp_path / "out.zip"
    count = export_zip(stream(), zp, BatchOptions())
    assert count == 2
    with zipfile.ZipFile(zp) as zf:
        names = sorted(zf.namelist())
        assert names == ["a_nobg.png", "a_nobg_1.png", "errors.txt"]
        assert "rusak.jpg" in zf.read("errors.txt").decode()


def test_export_folder_jpeg_for_solid_bg(image_dir, tmp_path):
    opts = BatchOptions(background=BackgroundSpec(mode=BgMode.COLOR))
    res = run_batch(collect_images([image_dir]), CirclePredictor(), opts)
    written = export_folder(iter(res), tmp_path / "out", opts)
    assert len(written) == 2 and all(p.suffix == ".jpg" for p in written)