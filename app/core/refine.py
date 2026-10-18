"""Refinement alpha matte.

Formula (dijalankan berurutan, semua parameter opsional):
    1. Guided filter : α₁ = a·I + b, dengan
                       a = cov(I, α) / (var(I) + ε),  b = mean(α) - a·mean(I)
                       (I = luminance gambar asli, radius r) -> tepi menempel ke detail asli
    2. Level         : α₂ = clip((α₁ - lo) / (hi - lo), 0, 1)  -> buang semi-transparan noise
    3. Shift edge    : α₃ = dilate(α₂, k) jika k > 0, erode(α₂, |k|) jika k < 0
    4. Feather       : α₄ = GaussianBlur(α₃, σ)
    5. Brush manual  : α = α₄·(1 - w) + target·w, w = soft brush mask, target ∈ {0, 1}
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import cv2
import numpy as np


class BrushMode(str, Enum):
    KEEP = "keep"    # paksa α -> 1 (kembalikan objek)
    ERASE = "erase"  # paksa α -> 0 (hapus sisa background)


@dataclass
class BrushStroke:
    mode: BrushMode
    points: list[tuple[float, float]]  # koordinat piksel gambar asli
    radius: float = 20.0
    hardness: float = 0.7              # 0 = sangat lembut, 1 = keras
    strength: float = 1.0              # opasitas stroke 0..1


@dataclass
class RefineParams:
    guided_radius: int = 0             # 0 = nonaktif. Saran 4–12
    guided_eps: float = 1e-3
    level_low: float = 0.0
    level_high: float = 1.0
    edge_shift: int = 0                # px, + melebarkan, - menyusutkan
    feather: float = 0.0               # sigma Gaussian (px)
    strokes: list[BrushStroke] = field(default_factory=list)


def guided_filter(guide: np.ndarray, src: np.ndarray, r: int, eps: float) -> np.ndarray:
    """Guided filter (He et al.) versi box filter, O(N) terhadap radius."""
    ksize = (2 * r + 1, 2 * r + 1)

    def box(x: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(x, cv2.CV_32F, ksize, borderType=cv2.BORDER_REFLECT)

    mean_i = box(guide)
    mean_p = box(src)
    cov_ip = box(guide * src) - mean_i * mean_p
    var_i = box(guide * guide) - mean_i * mean_i
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def _luminance(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0


def render_stroke_mask(shape: tuple[int, int], stroke: BrushStroke) -> np.ndarray:
    """Rasterisasi stroke jadi bobot w ∈ [0,1] dengan tepi lembut."""
    h, w = shape
    mask = np.zeros((h, w), np.float32)
    pts = np.round(np.asarray(stroke.points, np.float32)).astype(np.int32)
    r = max(1, int(round(stroke.radius)))
    if len(pts) == 1:
        cv2.circle(mask, tuple(int(v) for v in pts[0]), r, 1.0, -1, cv2.LINE_AA)
    elif len(pts) > 1:
        cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 1.0, 2 * r, cv2.LINE_AA)
    softness = (1.0 - float(np.clip(stroke.hardness, 0, 1))) * r
    if softness >= 0.5:
        mask = cv2.GaussianBlur(mask, (0, 0), softness)
    return np.clip(mask * stroke.strength, 0.0, 1.0)


def apply_strokes(alpha: np.ndarray, strokes: list[BrushStroke]) -> np.ndarray:
    out = alpha.copy()
    for s in strokes:
        w = render_stroke_mask(out.shape, s)
        target = 1.0 if s.mode == BrushMode.KEEP else 0.0
        out = out * (1.0 - w) + target * w
    return out


def refine_alpha(rgb: np.ndarray, alpha: np.ndarray, p: RefineParams) -> np.ndarray:
    """rgb: uint8 H×W×3, alpha: float32 H×W [0,1]. Return alpha baru."""
    a = alpha.astype(np.float32, copy=True)

    if p.guided_radius > 0:
        a = guided_filter(_luminance(rgb), a, p.guided_radius, p.guided_eps)

    lo, hi = float(p.level_low), float(p.level_high)
    if (lo, hi) != (0.0, 1.0):
        a = (a - lo) / max(hi - lo, 1e-6)
    a = np.clip(a, 0.0, 1.0)

    if p.edge_shift:
        k = abs(int(p.edge_shift))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1))
        a = cv2.dilate(a, kernel) if p.edge_shift > 0 else cv2.erode(a, kernel)

    if p.feather > 0:
        a = cv2.GaussianBlur(a, (0, 0), float(p.feather))

    if p.strokes:
        a = apply_strokes(a, p.strokes)

    return np.clip(a, 0.0, 1.0).astype(np.float32)