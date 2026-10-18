    ` ````# RemoveBG-RMNZ

Aplikasi desktop macOS & Windows untuk menghapus background semua jenis objek. Pemrosesan utama lokal (offline, ONNX Runtime) dengan opsi API remove.bg.

Fitur MVP: batch processing + ekspor ZIP/folder, manual edge refinement (brush keep/erase + slider tepi), background replacement (transparan, warna, gambar, blur).

## Menjalankan

Butuh Python 3.10–3.12 (PySide6 6.7 dan onnxruntime 1.18 belum mendukung 3.13).

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt   # Windows: .venv\Scripts\python
.venv/bin/python -m app.main
.venv/bin/python -m pytest -q
```

Model diunduh otomatis saat pertama kali dipakai ke folder data user:
- macOS: `~/Library/Application Support/AIRemoveBG/models`
- Windows: `%LOCALAPPDATA%\AIRemoveBG\models`

Akselerasi GPU dipilih otomatis: CoreML (macOS), DirectML (Windows, ganti paket ke `onnxruntime-directml`), CUDA (`onnxruntime-gpu`), fallback CPU.

## Formula Engine

```
1. Preprocess   x = (resize(I, S×S)/255 − mean) / std          → tensor 1×3×S×S
2. Inference    m = model(x)                                    → saliency S×S
3. Normalize    BiRefNet: m = sigmoid(m) | lainnya: m = (m−min)/(max−min+ε)
4. Upscale      α = resize_bilinear(m, W×H), clip [0,1]
5. Guided filter α₁ = mean(a)·I + mean(b),  a = cov(I,α)/(var(I)+ε),  b = mean(α) − a·mean(I)
6. Level        α₂ = clip((α₁ − lo)/(hi − lo), 0, 1)
7. Geser tepi   α₃ = dilate(α₂,k) jika k>0, erode(α₂,|k|) jika k<0
8. Feather      α₄ = GaussianBlur(α₃, σ)
9. Brush        α  = α₄·(1−w) + t·w,   w = soft brush mask, t ∈ {0 (erase), 1 (keep)}
10. Decontam    F̂ = blur(I·α)/(blur(α)+ε);  F = I jika α ≥ 0.98, selain itu F̂
11. Komposit    C = α·F + (1−α)·B,   B ∈ {transparan, warna, gambar cover-fit, blur(I)}
```

Alpha mentah (langkah 1–4) disimpan per gambar sebagai uint8. Langkah 5–11 dihitung ulang saat preview/ekspor, jadi mengubah slider atau brush tidak memicu inferensi ulang.

## Pilihan Model

| Key | Input | Ukuran | Lisensi | Kegunaan |
|---|---|---|---|---|
| `u2netp` | 320 | ~5 MB | Apache-2.0 | Preview cepat, tepi kasar |
| `u2net` | 320 | ~176 MB | Apache-2.0 | Baseline umum |
| `isnet-general` (default) | 1024 | ~176 MB | Apache-2.0 | Seimbang, bagus untuk produk/objek |
| `birefnet-lite` | 1024 | ~224 MB | MIT | Rambut/bulu lebih detail |
| `birefnet` | 1024 | ~973 MB | MIT | Kualitas tertinggi, perlu GPU untuk batch |
| API remove.bg | – | – | Komersial (per kredit) | Gambar dikirim ke server pihak ketiga |

Registry ada di `app/core/models.py`. Menambah model ONNX baru cukup dengan menambah `ModelSpec`.

## Struktur Folder

```
app/
  main.py              entry point (python -m app.main)
  core/                logika murni, tanpa dependensi Qt
    models.py          registry model + parameter normalisasi
    engine.py          download model, pemilihan provider, inferensi ONNX
    refine.py          guided filter, level, geser tepi, feather, brush
    compose.py         color decontamination + background replacement
    api_client.py      klien remove.bg (opsional)
    pipeline.py        load EXIF-aware, batch, ekspor folder/ZIP
  ui/
    main_window.py     layout 3 panel, settings, batch & ekspor
    canvas.py          kanvas zoom/pan + brush
    workers.py         QThread worker agar UI tidak freeze
tests/test_core.py     unit test core (predictor dummy, tanpa model)
requirements.txt       dependensi ter-pin
pyproject.toml
```

## UI

```
┌──────────────┬─────────────────────────────────────┬──────────────────┐
│ + File/Folder│ Tampilan [Hasil|Asli|Mask]           │ Engine           │
│              │ Hapus(E) Pertahankan(K) Ukuran Keras │  ○ Lokal ○ API   │
│ ✓ a.jpg      │ Undo  Reset  Fit                     │  Model ▾         │
│ ✓✎ b.png     │ ┌─────────────────────────────────┐ │ Penyempurnaan    │
│ ○ c.jpg      │ │         kanvas (checker)         │ │  guided/level/   │
│ ⚠ d.jpg      │ │   scroll=zoom, spasi=pan, [ ]    │ │  geser/feather   │
│              │ └─────────────────────────────────┘ │ Background       │
│ Hapus        │                                      │ Proses & Ekspor  │
└──────────────┴─────────────────────────────────────┴──────────────────┘
 status bar · progres batch · Batal
```

Shortcut: Ctrl/Cmd+O tambah file, Ctrl/Cmd+Enter proses terpilih, Ctrl/Cmd+Z undo stroke, Ctrl/Cmd+S simpan, Ctrl/Cmd+E ekspor ZIP, E/K mode brush, F fit. Drag & drop file/folder ke jendela.

## Privasi & Keamanan

- Mode lokal tidak mengirim data ke mana pun (selain unduh model sekali dari GitHub release rembg).
- Mode API menampilkan peringatan sebelum dipakai. API key hanya di memori sesi atau env `REMOVEBG_API_KEY`, tidak ditulis ke disk atau log.

## Roadmap

Fase 0 – MVP (repo ini)
- Engine lokal multi-model, opsi API, batch + ZIP, brush keep/erase, refine tepi, 4 mode background.

Fase 1 – Kualitas & stabilitas (2–3 minggu)
- Verifikasi checksum SHA-256 model dan dialog progres download.
- Inferensi tiled/high-res untuk gambar > 4K, preview resolusi rendah untuk render brush yang lebih cepat.
- Simpan settings (model, refine, API key di Keychain/Credential Manager).
- Undo/redo penuh untuk slider, bukan hanya stroke.

Fase 2 – Editing lanjutan (3–4 minggu)
- Trimap + matting model (ViTMatte) untuk rambut/kaca/objek transparan.
- Brush "refine edge" (matting lokal), tool lasso/polygon.
- Point/box prompt dengan SAM untuk memilih objek tertentu.
- Shadow sintetis dan posisi/scale foreground di atas background.

Fase 3 – Distribusi (2 minggu)
- Build PyInstaller per OS, code signing + notarization macOS, signing Windows (MSIX/Inno Setup).
- Auto-update, crash report opt-in.
- Batch paralel multi-sesi, preset ekspor (ukuran, format WebP, margin/crop otomatis).