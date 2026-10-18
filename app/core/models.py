"""Registry model AI untuk segmentasi/matting background.

Semua model berformat ONNX agar bisa jalan di macOS (CoreML/CPU) dan
Windows (DirectML/CPU) lewat satu runtime.
"""
from __future__ import annotations

from dataclasses import dataclass, field

_REMBG_RELEASE = "https://github.com/danielgatis/rembg/releases/download/v0.0.0"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    url: str
    filename: str
    input_size: int                      # sisi input persegi (px)
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    apply_sigmoid: bool = False          # BiRefNet output berupa logits
    approx_size_mb: int = 0
    license: str = ""
    notes: str = ""
    tags: tuple[str, ...] = field(default_factory=tuple)


MODELS: dict[str, ModelSpec] = {
    "u2netp": ModelSpec(
        key="u2netp",
        label="U2-Net Lite (cepat)",
        url=f"{_REMBG_RELEASE}/u2netp.onnx",
        filename="u2netp.onnx",
        input_size=320,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        approx_size_mb=5,
        license="Apache-2.0",
        notes="Preview cepat, kualitas tepi rendah.",
        tags=("fast", "preview"),
    ),
    "u2net": ModelSpec(
        key="u2net",
        label="U2-Net (umum)",
        url=f"{_REMBG_RELEASE}/u2net.onnx",
        filename="u2net.onnx",
        input_size=320,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        approx_size_mb=176,
        license="Apache-2.0",
        notes="Baseline umum untuk semua jenis objek.",
        tags=("general",),
    ),
    "isnet-general": ModelSpec(
        key="isnet-general",
        label="ISNet General (default)",
        url=f"{_REMBG_RELEASE}/isnet-general-use.onnx",
        filename="isnet-general-use.onnx",
        input_size=1024,
        mean=(0.5, 0.5, 0.5),
        std=(1.0, 1.0, 1.0),
        approx_size_mb=176,
        license="Apache-2.0",
        notes="Seimbang kecepatan/kualitas, bagus untuk produk & objek.",
        tags=("general", "default"),
    ),
    "birefnet-lite": ModelSpec(
        key="birefnet-lite",
        label="BiRefNet Lite (kualitas tinggi)",
        url=f"{_REMBG_RELEASE}/BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        filename="BiRefNet-general-bb_swin_v1_tiny-epoch_232.onnx",
        input_size=1024,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        apply_sigmoid=True,
        approx_size_mb=224,
        license="MIT",
        notes="Detail rambut/tepi halus lebih baik, lebih lambat dari ISNet.",
        tags=("quality",),
    ),
    "birefnet": ModelSpec(
        key="birefnet",
        label="BiRefNet General (terbaik, berat)",
        url=f"{_REMBG_RELEASE}/BiRefNet-general-epoch_244.onnx",
        filename="BiRefNet-general-epoch_244.onnx",
        input_size=1024,
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD,
        apply_sigmoid=True,
        approx_size_mb=973,
        license="MIT",
        notes="Kualitas tertinggi, butuh GPU untuk batch besar.",
        tags=("quality", "heavy"),
    ),
}

DEFAULT_MODEL = "isnet-general"


def get_model(key: str) -> ModelSpec:
    try:
        return MODELS[key]
    except KeyError as exc:
        raise ValueError(f"Model tidak dikenal: {key!r}. Pilihan: {list(MODELS)}") from exc