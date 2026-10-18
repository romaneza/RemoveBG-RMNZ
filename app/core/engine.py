"""Engine inferensi lokal (ONNX Runtime).

Formula pipeline:
    1. Preprocess : x = (resize(I, S×S) / 255 - mean) / std   -> tensor NCHW float32
    2. Inference  : m = model(x)[0][0, 0]                       -> peta saliency S×S
    3. Normalize  : BiRefNet -> m = sigmoid(m)
                    lainnya  -> m = (m - min) / (max - min + eps)
    4. Upscale    : α = resize_bilinear(m, W×H), clip ke [0, 1]
Refinement & komposit ada di refine.py dan compose.py.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

import numpy as np
import requests
from PIL import Image
from platformdirs import user_data_dir

from app.core.models import DEFAULT_MODEL, ModelSpec, get_model

log = logging.getLogger(__name__)

APP_NAME = "AIRemoveBG"
MODEL_DIR = Path(user_data_dir(APP_NAME, appauthor=False)) / "models"

# Urutan preferensi akselerator. Yang tidak tersedia otomatis dilewati.
_PROVIDER_PRIORITY = (
    "CUDAExecutionProvider",      # Windows/Linux + NVIDIA (paket onnxruntime-gpu)
    "DmlExecutionProvider",       # Windows DirectML (paket onnxruntime-directml)
    "CoreMLExecutionProvider",    # macOS
    "CPUExecutionProvider",
)

ProgressCb = Callable[[int, int], None]  # (bytes_done, bytes_total)


def model_path(spec: ModelSpec) -> Path:
    return MODEL_DIR / spec.filename


def is_downloaded(spec: ModelSpec) -> bool:
    p = model_path(spec)
    return p.exists() and p.stat().st_size > 0


def download_model(spec: ModelSpec, progress: ProgressCb | None = None) -> Path:
    """Unduh model ke folder data user. Ditulis ke .part dulu lalu rename (atomik)."""
    dest = model_path(spec)
    if is_downloaded(spec):
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    with requests.get(spec.url, stream=True, timeout=30) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if progress:
                    progress(done, total)

    if total and tmp.stat().st_size != total:
        tmp.unlink(missing_ok=True)
        raise IOError(f"Download {spec.filename} tidak lengkap")
    tmp.replace(dest)
    return dest


def select_providers() -> list[str]:
    import onnxruntime as ort

    available = set(ort.get_available_providers())
    return [p for p in _PROVIDER_PRIORITY if p in available] or ["CPUExecutionProvider"]


def preprocess(img: Image.Image, spec: ModelSpec) -> np.ndarray:
    s = spec.input_size
    rgb = img.convert("RGB").resize((s, s), Image.BILINEAR)
    x = np.asarray(rgb, dtype=np.float32) / 255.0
    x = (x - np.array(spec.mean, np.float32)) / np.array(spec.std, np.float32)
    return x.transpose(2, 0, 1)[None].astype(np.float32)  # 1×3×S×S


def postprocess(raw: np.ndarray, spec: ModelSpec, size: tuple[int, int]) -> np.ndarray:
    """raw: output model (1×1×S×S atau mirip). size: (W, H). Return α float32 H×W di [0,1]."""
    m = np.squeeze(raw).astype(np.float32)
    if m.ndim == 3:  # beberapa model punya channel ekstra
        m = m[0]
    if spec.apply_sigmoid:
        m = 1.0 / (1.0 + np.exp(-m))
    else:
        lo, hi = float(m.min()), float(m.max())
        m = (m - lo) / (hi - lo + 1e-8)
    mask = Image.fromarray(m).resize(size, Image.BILINEAR)
    return np.clip(np.asarray(mask, dtype=np.float32), 0.0, 1.0)


class LocalEngine:
    """Satu sesi ONNX per model, di-cache. Aman dipakai dari worker thread."""

    def __init__(self, model_key: str = DEFAULT_MODEL):
        self._lock = threading.Lock()
        self._sessions: dict[str, object] = {}
        self.model_key = model_key

    @property
    def spec(self) -> ModelSpec:
        return get_model(self.model_key)

    def _session(self, spec: ModelSpec):
        with self._lock:
            sess = self._sessions.get(spec.key)
            if sess is None:
                import onnxruntime as ort

                path = download_model(spec)
                opts = ort.SessionOptions()
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                providers = select_providers()
                log.info("Load %s dengan providers %s", spec.key, providers)
                sess = ort.InferenceSession(str(path), sess_options=opts, providers=providers)
                self._sessions[spec.key] = sess
            return sess

    def predict_alpha(self, img: Image.Image) -> np.ndarray:
        spec = self.spec
        sess = self._session(spec)
        x = preprocess(img, spec)
        input_name = sess.get_inputs()[0].name
        raw = sess.run(None, {input_name: x})[0]
        return postprocess(raw, spec, img.size)