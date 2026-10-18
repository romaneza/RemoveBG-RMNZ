"""Komposit & background replacement.

Formula:
    1. Color decontamination (hapus "halo" warna background di tepi):
           F̂ = blur(I·α, σ) / (blur(α, σ) + ε)
           F  = α_core·I + (1 - α_core)·F̂,   α_core = 1 jika α ≥ 0.98
       -> piksel semi-transparan mengambil warna dari bagian objek terdekat.
    2. Alpha compositing (straight alpha):
           C = α·F + (1 - α)·B
       B ∈ {transparan, warna solid, gambar (cover-fit), blur gambar asli}.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np
from PIL import Image


class BgMode(str, Enum):
    TRANSPARENT = "transparent"
    COLOR = "color"
    IMAGE = "image"
    BLUR = "blur"


@dataclass
class BackgroundSpec:
    mode: BgMode = BgMode.TRANSPARENT
    color: tuple[int, int, int] = (255, 255, 255)
    image: Image.Image | None = None
    blur_sigma: float = 25.0
    decontaminate: bool = True
    decon_sigma: float = 6.0


def decontaminate_colors(rgb: np.ndarray, alpha: np.ndarray, sigma: float = 6.0) -> np.ndarray:
    """rgb uint8 H×W×3, alpha float32 H×W. Return float32 H×W×3 [0,1]."""
    img = rgb.astype(np.float32) / 255.0
    a = alpha[..., None]
    num = cv2.GaussianBlur(img * a, (0, 0), sigma)
    den = cv2.GaussianBlur(alpha, (0, 0), sigma)[..., None]
    f_est = num / (den + 1e-6)
    core = (alpha >= 0.98).astype(np.float32)[..., None]
    return np.clip(core * img + (1.0 - core) * f_est, 0.0, 1.0)


def _cover_fit(bg: Image.Image, size: tuple[int, int]) -> np.ndarray:
    """Resize + crop tengah agar menutupi seluruh kanvas (seperti CSS object-fit: cover)."""
    w, h = size
    bw, bh = bg.size
    scale = max(w / bw, h / bh)
    nw, nh = max(w, round(bw * scale)), max(h, round(bh * scale))
    resized = bg.convert("RGB").resize((nw, nh), Image.LANCZOS)
    left, top = (nw - w) // 2, (nh - h) // 2
    cropped = resized.crop((left, top, left + w, top + h))
    return np.asarray(cropped, np.float32) / 255.0


def compose(rgb: np.ndarray, alpha: np.ndarray, bg: BackgroundSpec) -> Image.Image:
    """Gabungkan foreground dengan background. Return PIL RGBA (transparan) atau RGB."""
    h, w = alpha.shape
    fg = (
        decontaminate_colors(rgb, alpha, bg.decon_sigma)
        if bg.decontaminate
        else rgb.astype(np.float32) / 255.0
    )

    if bg.mode == BgMode.TRANSPARENT:
        out = np.dstack([fg, alpha])
        return Image.fromarray((out * 255 + 0.5).astype(np.uint8), "RGBA")

    if bg.mode == BgMode.COLOR:
        back = np.empty((h, w, 3), np.float32)
        back[:] = np.asarray(bg.color, np.float32) / 255.0
    elif bg.mode == BgMode.IMAGE:
        if bg.image is None:
            raise ValueError("Mode IMAGE butuh gambar background")
        back = _cover_fit(bg.image, (w, h))
    elif bg.mode == BgMode.BLUR:
        back = cv2.GaussianBlur(rgb.astype(np.float32) / 255.0, (0, 0), bg.blur_sigma)
    else:
        raise ValueError(f"Mode background tidak dikenal: {bg.mode}")

    a = alpha[..., None]
    out = a * fg + (1.0 - a) * back
    return Image.fromarray((np.clip(out, 0, 1) * 255 + 0.5).astype(np.uint8), "RGB")