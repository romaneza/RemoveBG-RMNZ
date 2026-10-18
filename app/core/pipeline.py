"""Pipeline lengkap + batch processing + ekspor ZIP.

Alur per gambar:
    load (EXIF-aware) -> predict_alpha (lokal/API) -> refine_alpha -> compose -> simpan
Batch berjalan sekuensial per gambar (sesi ONNX sudah memakai multi-thread/GPU),
dengan callback progres dan dukungan cancel.
"""
from __future__ import annotations

import io
import logging
import threading
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Protocol

import numpy as np
from PIL import Image, ImageOps

from app.core.compose import BackgroundSpec, BgMode, compose
from app.core.refine import RefineParams, refine_alpha

log = logging.getLogger(__name__)

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


class AlphaPredictor(Protocol):
    def predict_alpha(self, img: Image.Image) -> np.ndarray: ...


@dataclass
class ProcessResult:
    source: Path
    image: Image.Image | None = None
    alpha: np.ndarray | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


@dataclass
class BatchOptions:
    refine: RefineParams = field(default_factory=RefineParams)
    background: BackgroundSpec = field(default_factory=BackgroundSpec)
    jpeg_quality: int = 95


def load_image(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)  # hormati orientasi kamera
    return img.convert("RGB")


def collect_images(paths: list[Path]) -> list[Path]:
    """Terima file dan/atau folder, kembalikan daftar gambar yang didukung (urut, unik)."""
    out: list[Path] = []
    for p in paths:
        if p.is_dir():
            out.extend(sorted(f for f in p.rglob("*") if f.suffix.lower() in SUPPORTED_EXT))
        elif p.suffix.lower() in SUPPORTED_EXT:
            out.append(p)
    seen: set[Path] = set()
    return [p for p in out if not (p in seen or seen.add(p))]


def process_image(
    img: Image.Image,
    predictor: AlphaPredictor,
    opts: BatchOptions,
    alpha: np.ndarray | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Proses satu gambar. `alpha` bisa diberikan untuk melewati inferensi (re-render setelah edit)."""
    rgb = np.asarray(img.convert("RGB"))
    raw = alpha if alpha is not None else predictor.predict_alpha(img)
    refined = refine_alpha(rgb, raw, opts.refine)
    return compose(rgb, refined, opts.background), refined


def output_name(src: Path, bg: BackgroundSpec) -> str:
    ext = ".png" if bg.mode == BgMode.TRANSPARENT else ".jpg"
    return f"{src.stem}_nobg{ext}"


def encode_image(img: Image.Image, jpeg_quality: int = 95) -> tuple[bytes, str]:
    buf = io.BytesIO()
    if img.mode == "RGBA":
        img.save(buf, format="PNG", optimize=True)
        return buf.getvalue(), ".png"
    img.save(buf, format="JPEG", quality=jpeg_quality, subsampling=0)
    return buf.getvalue(), ".jpg"


def _unique(name: str, used: set[str]) -> str:
    stem, dot, ext = name.rpartition(".")
    candidate, i = name, 1
    while candidate in used:
        candidate = f"{stem}_{i}{dot}{ext}"
        i += 1
    used.add(candidate)
    return candidate


ProgressCb = Callable[[int, int, ProcessResult], None]


def run_batch(
    files: list[Path],
    predictor: AlphaPredictor,
    opts: BatchOptions,
    progress: ProgressCb | None = None,
    cancel: threading.Event | None = None,
) -> list[ProcessResult]:
    results: list[ProcessResult] = []
    total = len(files)
    for i, path in enumerate(files, 1):
        if cancel and cancel.is_set():
            break
        res = ProcessResult(source=path)
        try:
            res.image, res.alpha = process_image(load_image(path), predictor, opts)
        except Exception as exc:  # satu gambar gagal tidak menghentikan batch
            log.exception("Gagal memproses %s", path)
            res.error = str(exc)
        results.append(res)
        if progress:
            progress(i, total, res)
    return results


def export_folder(results: Iterable[ProcessResult], out_dir: Path, opts: BatchOptions) -> list[Path]:
    """Tulis hasil sukses ke folder. `results` boleh berupa generator (streaming)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    written: list[Path] = []
    for r in results:
        if not r.ok or r.image is None:
            continue
        data, ext = encode_image(r.image, opts.jpeg_quality)
        name = _unique(f"{r.source.stem}_nobg{ext}", used)
        dest = out_dir / name
        dest.write_bytes(data)
        written.append(dest)
    return written


def export_zip(results: Iterable[ProcessResult], zip_path: Path, opts: BatchOptions) -> int:
    """Tulis semua hasil sukses ke satu ZIP, return jumlah gambar yang ditulis.

    `results` boleh berupa generator: tiap gambar di-encode lalu langsung dilepas,
    jadi RAM tetap kecil untuk batch besar. PNG/JPEG sudah terkompresi -> ZIP_STORED.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    errors: list[str] = []
    count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as zf:
        for r in results:
            if not r.ok or r.image is None:
                errors.append(f"{r.source.name}: {r.error}")
                continue
            data, ext = encode_image(r.image, opts.jpeg_quality)
            zf.writestr(_unique(f"{r.source.stem}_nobg{ext}", used), data)
            count += 1
        if errors:
            zf.writestr("errors.txt", "\n".join(errors))
    return count