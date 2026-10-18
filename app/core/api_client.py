"""Opsi pemrosesan via API (remove.bg).

Gambar DIKIRIM ke server pihak ketiga. Hanya dipakai bila user memilih mode API
secara eksplisit. API key dibaca dari env var REMOVEBG_API_KEY atau settings,
tidak pernah di-hardcode atau di-log.

Hasil API (PNG RGBA) diambil channel alpha-nya saja, lalu masuk ke pipeline
refine/compose yang sama dengan engine lokal.
"""
from __future__ import annotations

import io
import os

import numpy as np
import requests
from PIL import Image

REMOVEBG_URL = "https://api.remove.bg/v1.0/removebg"
MAX_UPLOAD_SIDE = 4000  # batas aman agar upload tidak terlalu besar


class ApiError(RuntimeError):
    pass


class RemoveBgClient:
    def __init__(self, api_key: str | None = None, timeout: float = 60.0):
        self.api_key = api_key or os.environ.get("REMOVEBG_API_KEY", "")
        self.timeout = timeout

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def predict_alpha(self, img: Image.Image) -> np.ndarray:
        if not self.configured:
            raise ApiError("API key remove.bg belum diatur (Settings atau REMOVEBG_API_KEY)")

        orig_size = img.size
        upload = img.convert("RGB")
        upload.thumbnail((MAX_UPLOAD_SIDE, MAX_UPLOAD_SIDE), Image.LANCZOS)
        buf = io.BytesIO()
        upload.save(buf, format="JPEG", quality=95)
        buf.seek(0)

        try:
            r = requests.post(
                REMOVEBG_URL,
                files={"image_file": ("image.jpg", buf, "image/jpeg")},
                data={"size": "auto", "format": "png"},
                headers={"X-Api-Key": self.api_key},
                timeout=self.timeout,
            )
        except requests.RequestException as exc:
            raise ApiError(f"Gagal menghubungi API: {exc}") from exc

        if r.status_code != 200:
            # Jangan sertakan header/key di pesan error
            detail = r.text[:200] if r.headers.get("content-type", "").startswith("application/json") else ""
            raise ApiError(f"API error {r.status_code}. {detail}".strip())

        result = Image.open(io.BytesIO(r.content)).convert("RGBA")
        alpha = result.getchannel("A").resize(orig_size, Image.BILINEAR)
        return np.asarray(alpha, np.float32) / 255.0