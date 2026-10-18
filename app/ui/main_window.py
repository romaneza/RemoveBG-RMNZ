"""Jendela utama: daftar file, kanvas editor, panel pengaturan, batch & ekspor.

Layout:
    [Daftar gambar] | [Toolbar brush + Kanvas] | [Engine / Refine / Background / Ekspor]

Alur data:
    - Batch worker hanya menghitung alpha MENTAH per gambar (disimpan uint8 agar hemat RAM).
    - Refine, brush stroke, dan background diterapkan saat render preview & saat ekspor,
      jadi mengubah slider tidak perlu inferensi ulang.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QButtonGroup,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.api_client import RemoveBgClient
from app.core.compose import BackgroundSpec, BgMode
from app.core.engine import LocalEngine, is_downloaded
from app.core.models import DEFAULT_MODEL, MODELS
from app.core.pipeline import (
    SUPPORTED_EXT,
    BatchOptions,
    ProcessResult,
    collect_images,
    export_folder,
    export_zip,
    load_image,
    output_name,
    process_image,
)
from app.core.refine import BrushMode, RefineParams, refine_alpha
from app.ui.canvas import MaskCanvas
from app.ui.workers import BatchWorker, start_worker


@dataclass
class ImageItem:
    path: Path
    alpha_u8: np.ndarray | None = None      # alpha mentah dari model, uint8 H×W
    strokes: list = field(default_factory=list)
    error: str | None = None

    @property
    def alpha(self) -> np.ndarray | None:
        return None if self.alpha_u8 is None else self.alpha_u8.astype(np.float32) / 255.0


VIEW_RESULT, VIEW_ORIGINAL, VIEW_MASK = 0, 1, 2


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("RemoveBG-RMNZ")
        self.resize(1400, 860)
        self.setAcceptDrops(True)

        self.engine = LocalEngine(DEFAULT_MODEL)
        self.api = RemoveBgClient()
        self.items: list[ImageItem] = []
        self._path_index: dict[Path, int] = {}
        self._current_idx = -1
        self._current_path: Path | None = None
        self._current_img: Image.Image | None = None
        self._bg_color = (255, 255, 255)
        self._bg_image: Image.Image | None = None
        self._api_warned = False
        self._busy = False
        self._thread = None
        self._worker: BatchWorker | None = None

        # Debounce render saat slider digeser
        self._render_timer = QTimer(self)
        self._render_timer.setSingleShot(True)
        self._render_timer.setInterval(120)
        self._render_timer.timeout.connect(self._render_current)

        self._build_ui()
        self._build_actions()
        self._set_busy(False)

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        splitter = QSplitter(Qt.Horizontal, self)
        splitter.addWidget(self._build_file_panel())
        splitter.addWidget(self._build_canvas_panel())
        splitter.addWidget(self._build_settings_panel())
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([260, 820, 320])
        self.setCentralWidget(splitter)

        self.progress = QProgressBar()
        self.progress.setMaximumWidth(240)
        self.progress.setAccessibleName("Progres batch")
        self.btn_cancel = QPushButton("Batal")
        self.btn_cancel.clicked.connect(self._cancel_batch)
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().addPermanentWidget(self.btn_cancel)
        self.statusBar().showMessage("Tarik & lepas gambar atau folder ke jendela ini")

    def _build_file_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        row = QHBoxLayout()
        b_files = QPushButton("Tambah File…")
        b_files.clicked.connect(self._pick_files)
        b_dir = QPushButton("Tambah Folder…")
        b_dir.clicked.connect(self._pick_folder)
        row.addWidget(b_files)
        row.addWidget(b_dir)
        lay.addLayout(row)

        self.list = QListWidget()
        self.list.setAccessibleName("Daftar gambar")
        self.list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.list.currentRowChanged.connect(self._on_select)
        lay.addWidget(self.list, 1)

        self.btn_remove = QPushButton("Hapus dari Daftar")
        self.btn_remove.clicked.connect(self._remove_selected)
        lay.addWidget(self.btn_remove)
        return w

    def _build_canvas_panel(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)

        # Kanvas dibuat dulu karena toolbar terhubung ke sini
        self.canvas = MaskCanvas()
        self.canvas.setAcceptDrops(False)
        self.canvas.viewport().setAcceptDrops(False)
        self.canvas.strokeFinished.connect(self._on_stroke)
        self.canvas.brushRadiusChanged.connect(self._sync_radius_slider)

        bar = QHBoxLayout()
        self.view_combo = QComboBox()
        self.view_combo.addItems(["Hasil", "Asli", "Mask"])
        self.view_combo.setAccessibleName("Mode tampilan")
        self.view_combo.currentIndexChanged.connect(lambda _: self._render_current())
        bar.addWidget(QLabel("Tampilan:"))
        bar.addWidget(self.view_combo)
        bar.addSpacing(12)

        self.btn_erase = QToolButton()
        self.btn_erase.setText("Hapus (E)")
        self.btn_erase.setToolTip("Brush hapus: buang sisa background")
        self.btn_keep = QToolButton()
        self.btn_keep.setText("Pertahankan (K)")
        self.btn_keep.setToolTip("Brush pertahankan: kembalikan bagian objek")
        group = QButtonGroup(self)
        for b in (self.btn_erase, self.btn_keep):
            b.setCheckable(True)
            group.addButton(b)
            bar.addWidget(b)
        self.btn_erase.toggled.connect(lambda on: on and self.canvas.set_brush_mode(BrushMode.ERASE))
        self.btn_keep.toggled.connect(lambda on: on and self.canvas.set_brush_mode(BrushMode.KEEP))
        self.btn_erase.setChecked(True)

        bar.addWidget(QLabel("Ukuran:"))
        self.radius_slider = QSlider(Qt.Horizontal)
        self.radius_slider.setRange(1, 300)
        self.radius_slider.setValue(int(self.canvas.brush_radius))
        self.radius_slider.setMaximumWidth(140)
        self.radius_slider.setAccessibleName("Ukuran brush")
        self.radius_slider.setToolTip("Ukuran brush ([ / ])")
        self.radius_slider.valueChanged.connect(self.canvas.set_brush_radius)
        bar.addWidget(self.radius_slider)

        bar.addWidget(QLabel("Kekerasan:"))
        self.hard_slider = QSlider(Qt.Horizontal)
        self.hard_slider.setRange(0, 100)
        self.hard_slider.setValue(int(self.canvas.brush_hardness * 100))
        self.hard_slider.setMaximumWidth(100)
        self.hard_slider.setAccessibleName("Kekerasan brush")
        self.hard_slider.valueChanged.connect(lambda v: setattr(self.canvas, "brush_hardness", v / 100))
        bar.addWidget(self.hard_slider)

        bar.addStretch(1)
        for text, slot in (("Undo", self._undo_stroke), ("Reset Edit", self._reset_edits), ("Fit", self.canvas.fit)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            bar.addWidget(b)

        lay.addLayout(bar)
        lay.addWidget(self.canvas, 1)
        return w

    def _build_settings_panel(self) -> QWidget:
        inner = QWidget()
        lay = QVBoxLayout(inner)

        # --- Engine
        g = QGroupBox("Engine")
        f = QFormLayout(g)
        self.rb_local = QRadioButton("Lokal (offline)")
        self.rb_api = QRadioButton("API remove.bg (online)")
        self.rb_local.setChecked(True)
        self.rb_api.toggled.connect(self._on_api_toggled)
        f.addRow(self.rb_local)
        f.addRow(self.rb_api)
        self.model_combo = QComboBox()
        self.model_combo.setAccessibleName("Model AI")
        for key, spec in MODELS.items():
            self.model_combo.addItem(f"{spec.label} · ~{spec.approx_size_mb} MB", key)
            self.model_combo.setItemData(self.model_combo.count() - 1, spec.notes, Qt.ToolTipRole)
        self.model_combo.setCurrentIndex(self.model_combo.findData(DEFAULT_MODEL))
        f.addRow("Model:", self.model_combo)
        self.btn_api_key = QPushButton("Atur API Key…")
        self.btn_api_key.clicked.connect(self._ask_api_key)
        f.addRow(self.btn_api_key)
        lay.addWidget(g)

        # --- Refine
        g = QGroupBox("Penyempurnaan Tepi")
        f = QFormLayout(g)
        self.sp_guided = QSpinBox()
        self.sp_guided.setRange(0, 30)
        self.sp_guided.setToolTip("Guided filter: tepi mengikuti detail gambar asli (rambut, bulu). 0 = mati")
        self.sp_low = QDoubleSpinBox()
        self.sp_low.setRange(0.0, 0.95)
        self.sp_low.setSingleStep(0.05)
        self.sp_low.setToolTip("Alpha di bawah nilai ini jadi transparan penuh")
        self.sp_high = QDoubleSpinBox()
        self.sp_high.setRange(0.05, 1.0)
        self.sp_high.setSingleStep(0.05)
        self.sp_high.setValue(1.0)
        self.sp_high.setToolTip("Alpha di atas nilai ini jadi solid penuh")
        self.sp_shift = QSpinBox()
        self.sp_shift.setRange(-20, 20)
        self.sp_shift.setSuffix(" px")
        self.sp_shift.setToolTip("Positif melebarkan, negatif menyusutkan tepi")
        self.sp_feather = QDoubleSpinBox()
        self.sp_feather.setRange(0.0, 20.0)
        self.sp_feather.setSingleStep(0.5)
        self.sp_feather.setSuffix(" px")
        f.addRow("Guided radius:", self.sp_guided)
        f.addRow("Level bawah:", self.sp_low)
        f.addRow("Level atas:", self.sp_high)
        f.addRow("Geser tepi:", self.sp_shift)
        f.addRow("Feather:", self.sp_feather)
        for sp in (self.sp_guided, self.sp_low, self.sp_high, self.sp_shift, self.sp_feather):
            sp.valueChanged.connect(self._schedule_render)
        lay.addWidget(g)

        # --- Background
        g = QGroupBox("Background")
        f = QFormLayout(g)
        self.bg_combo = QComboBox()
        self.bg_combo.setAccessibleName("Mode background")
        for text, mode in (("Transparan", BgMode.TRANSPARENT), ("Warna", BgMode.COLOR),
                           ("Gambar", BgMode.IMAGE), ("Blur asli", BgMode.BLUR)):
            self.bg_combo.addItem(text, mode.value)
        self.bg_combo.currentIndexChanged.connect(lambda _: self._on_bg_mode())
        self.btn_color = QPushButton("Pilih Warna…")
        self.btn_color.clicked.connect(self._pick_color)
        self._update_color_swatch()
        self.btn_bg_image = QPushButton("Pilih Gambar…")
        self.btn_bg_image.clicked.connect(self._pick_bg_image)
        self.sp_blur = QDoubleSpinBox()
        self.sp_blur.setRange(1.0, 100.0)
        self.sp_blur.setValue(25.0)
        self.sp_blur.valueChanged.connect(self._schedule_render)
        self.cb_decon = QCheckBox("Bersihkan halo warna di tepi")
        self.cb_decon.setChecked(True)
        self.cb_decon.toggled.connect(self._schedule_render)
        f.addRow("Mode:", self.bg_combo)
        f.addRow(self.btn_color)
        f.addRow(self.btn_bg_image)
        f.addRow("Kekuatan blur:", self.sp_blur)
        f.addRow(self.cb_decon)
        lay.addWidget(g)

        # --- Proses & ekspor
        g = QGroupBox("Proses & Ekspor")
        v = QVBoxLayout(g)
        self.btn_process_sel = QPushButton("Proses Terpilih")
        self.btn_process_sel.clicked.connect(self._process_selected)
        self.btn_process_all = QPushButton("Proses Semua")
        self.btn_process_all.clicked.connect(self._process_all)
        self.btn_save = QPushButton("Simpan Gambar Ini…")
        self.btn_save.clicked.connect(self._save_current)
        self.btn_export_dir = QPushButton("Ekspor ke Folder…")
        self.btn_export_dir.clicked.connect(self._export_folder)
        self.btn_export_zip = QPushButton("Ekspor ZIP…")
        self.btn_export_zip.clicked.connect(self._export_zip)
        for b in (self.btn_process_sel, self.btn_process_all, self.btn_save,
                  self.btn_export_dir, self.btn_export_zip):
            v.addWidget(b)
        lay.addWidget(g)
        lay.addStretch(1)

        self._on_bg_mode()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)
        scroll.setMinimumWidth(300)
        return scroll

    def _build_actions(self) -> None:
        def act(text, shortcut, slot):
            a = QAction(text, self)
            a.setShortcut(QKeySequence(shortcut))  # Ctrl otomatis jadi Cmd di macOS
            a.triggered.connect(slot)
            self.addAction(a)

        act("Tambah file", QKeySequence.Open, self._pick_files)
        act("Undo stroke", QKeySequence.Undo, self._undo_stroke)
        act("Simpan gambar", QKeySequence.Save, self._save_current)
        act("Ekspor ZIP", "Ctrl+E", self._export_zip)
        act("Proses terpilih", "Ctrl+Return", self._process_selected)
        act("Brush hapus", "E", lambda: self.btn_erase.setChecked(True))
        act("Brush pertahankan", "K", lambda: self.btn_keep.setChecked(True))
        act("Fit", "F", self.canvas.fit)

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        for w in (self.btn_process_sel, self.btn_process_all, self.btn_export_dir,
                  self.btn_export_zip, self.btn_remove, self.rb_local, self.rb_api):
            w.setEnabled(not busy)
        self.model_combo.setEnabled(not busy and not self.rb_api.isChecked())
        self.progress.setVisible(busy)
        self.btn_cancel.setVisible(busy)

    def _status(self, msg: str) -> None:
        self.statusBar().showMessage(msg)

    # ------------------------------------------------------------- daftar file
    def _pick_files(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXT))
        files, _ = QFileDialog.getOpenFileNames(self, "Pilih gambar", "", f"Gambar ({exts})")
        if files:
            self.add_paths([Path(f) for f in files])

    def _pick_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Pilih folder")
        if d:
            self.add_paths([Path(d)])

    def add_paths(self, paths: list[Path]) -> None:
        new = [p for p in collect_images([p.resolve() for p in paths]) if p not in self._path_index]
        if not new:
            self._status("Tidak ada gambar baru yang didukung")
            return
        was_empty = not self.items
        self.items.extend(ImageItem(p) for p in new)
        self._rebuild_list(select=0 if was_empty else self._current_idx)
        self._status(f"{len(new)} gambar ditambahkan")

    def _remove_selected(self) -> None:
        if self._busy:
            return
        rows = sorted({self.list.row(i) for i in self.list.selectedItems()}, reverse=True)
        for r in rows:
            del self.items[r]
        self._current_path = None  # paksa reload item terpilih
        self._rebuild_list(select=min(rows) if rows else self._current_idx)

    def _rebuild_list(self, select: int) -> None:
        self._path_index = {it.path: i for i, it in enumerate(self.items)}
        self.list.blockSignals(True)
        self.list.clear()
        for _ in self.items:
            self.list.addItem(QListWidgetItem())
        for i in range(len(self.items)):
            self._refresh_row(i)
        self.list.blockSignals(False)

        if not self.items:
            self._current_idx, self._current_path, self._current_img = -1, None, None
            self.canvas.set_image(None)
            return
        self.list.setCurrentRow(max(0, min(select, len(self.items) - 1)))

    def _refresh_row(self, i: int) -> None:
        it = self.items[i]
        li = self.list.item(i)
        if it.error:
            prefix = "⚠ "
        elif it.alpha_u8 is not None:
            prefix = "✓ " + ("✎ " if it.strokes else "")
        else:
            prefix = "○ "
        li.setText(prefix + it.path.name)
        li.setToolTip(it.error or str(it.path))

    def _on_select(self, row: int) -> None:
        if not 0 <= row < len(self.items):
            return
        it = self.items[row]
        if row == self._current_idx and it.path == self._current_path:
            return
        self._current_idx = row
        try:
            self._current_img = load_image(it.path)
            self._current_path = it.path
        except Exception as exc:
            self._current_img, self._current_path = None, None
            self.canvas.set_image(None)
            self._status(f"Gagal membuka {it.path.name}: {exc}")
            return
        self._render_current(fit=True)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event) -> None:
        paths = [Path(u.toLocalFile()) for u in event.mimeData().urls() if u.isLocalFile()]
        if paths:
            self.add_paths(paths)

    # ------------------------------------------------------------- render
    def _collect_opts(self, strokes=()) -> BatchOptions:
        low = self.sp_low.value()
        refine = RefineParams(
            guided_radius=self.sp_guided.value(),
            level_low=low,
            level_high=max(self.sp_high.value(), low + 0.01),
            edge_shift=self.sp_shift.value(),
            feather=self.sp_feather.value(),
            strokes=list(strokes),
        )
        bg = BackgroundSpec(
            mode=BgMode(self.bg_combo.currentData()),
            color=self._bg_color,
            image=self._bg_image,
            blur_sigma=self.sp_blur.value(),
            decontaminate=self.cb_decon.isChecked(),
        )
        return BatchOptions(refine=refine, background=bg)

    def _schedule_render(self, *_) -> None:
        self._render_timer.start()

    def _render_current(self, fit: bool = False) -> None:
        img = self._current_img
        if img is None or not 0 <= self._current_idx < len(self.items):
            return
        it = self.items[self._current_idx]
        view = self.view_combo.currentIndex()
        self.canvas.brush_enabled = it.alpha_u8 is not None

        if view == VIEW_ORIGINAL or it.alpha_u8 is None:
            self.canvas.set_image(img, fit=fit)
            if it.alpha_u8 is None:
                self._status("Belum diproses. Tekan Proses Terpilih (Ctrl+Enter)")
            return

        opts = self._collect_opts(it.strokes)
        try:
            if view == VIEW_MASK:
                refined = refine_alpha(np.asarray(img), it.alpha, opts.refine)
                out = Image.fromarray((refined * 255 + 0.5).astype(np.uint8), "L")
            else:
                out, _ = process_image(img, self.engine, opts, alpha=it.alpha)
        except Exception as exc:
            self._status(f"Render gagal: {exc}")
            return
        self.canvas.set_image(out, fit=fit)

    # ------------------------------------------------------------- brush
    def _sync_radius_slider(self, r: float) -> None:
        self.radius_slider.blockSignals(True)
        self.radius_slider.setValue(int(round(r)))
        self.radius_slider.blockSignals(False)

    def _current_item(self) -> ImageItem | None:
        return self.items[self._current_idx] if 0 <= self._current_idx < len(self.items) else None

    def _on_stroke(self, stroke) -> None:
        it = self._current_item()
        if it is None or it.alpha_u8 is None:
            return
        it.strokes.append(stroke)
        self._refresh_row(self._current_idx)
        self._render_current()

    def _undo_stroke(self) -> None:
        it = self._current_item()
        if it and it.strokes:
            it.strokes.pop()
            self._refresh_row(self._current_idx)
            self._render_current()

    def _reset_edits(self) -> None:
        it = self._current_item()
        if it and it.strokes:
            it.strokes.clear()
            self._refresh_row(self._current_idx)
            self._render_current()

    # ------------------------------------------------------------- background
    def _on_bg_mode(self) -> None:
        mode = BgMode(self.bg_combo.currentData())
        self.btn_color.setEnabled(mode == BgMode.COLOR)
        self.btn_bg_image.setEnabled(mode == BgMode.IMAGE)
        self.sp_blur.setEnabled(mode == BgMode.BLUR)
        self._schedule_render()

    def _update_color_swatch(self) -> None:
        pm = QPixmap(16, 16)
        pm.fill(QColor(*self._bg_color))
        self.btn_color.setIcon(QIcon(pm))

    def _pick_color(self) -> None:
        c = QColorDialog.getColor(QColor(*self._bg_color), self, "Warna background")
        if c.isValid():
            self._bg_color = (c.red(), c.green(), c.blue())
            self._update_color_swatch()
            self._schedule_render()

    def _pick_bg_image(self) -> None:
        exts = " ".join(f"*{e}" for e in sorted(SUPPORTED_EXT))
        f, _ = QFileDialog.getOpenFileName(self, "Gambar background", "", f"Gambar ({exts})")
        if not f:
            return
        try:
            self._bg_image = load_image(Path(f))
        except Exception as exc:
            QMessageBox.warning(self, "Background", f"Gagal membuka gambar: {exc}")
            return
        self.btn_bg_image.setText(f"Gambar: {Path(f).name}")
        self._schedule_render()

    def _bg_valid(self) -> bool:
        if BgMode(self.bg_combo.currentData()) == BgMode.IMAGE and self._bg_image is None:
            QMessageBox.information(self, "Background", "Pilih gambar background dulu.")
            return False
        return True

    # ------------------------------------------------------------- engine
    def _on_api_toggled(self, on: bool) -> None:
        self.model_combo.setEnabled(not on and not self._busy)
        if not on:
            return
        if not self._api_warned:
            self._api_warned = True
            QMessageBox.information(
                self, "Mode API",
                "Di mode API, gambar dikirim ke server remove.bg (pihak ketiga) untuk diproses "
                "dan memakai kredit akun remove.bg Anda. Mode lokal tidak mengirim data ke mana pun.",
            )
        if not self.api.configured:
            self._ask_api_key()

    def _ask_api_key(self) -> None:
        key, ok = QInputDialog.getText(
            self, "API Key remove.bg",
            "API key (hanya disimpan di memori selama aplikasi terbuka):",
            QLineEdit.EchoMode.Password,
        )
        if ok and key.strip():
            self.api.api_key = key.strip()
            self._status("API key diatur untuk sesi ini")

    def _predictor(self):
        if self.rb_api.isChecked():
            if not self.api.configured:
                self._ask_api_key()
            return self.api if self.api.configured else None
        self.engine.model_key = self.model_combo.currentData()
        return self.engine

    # ------------------------------------------------------------- batch
    def _process_selected(self) -> None:
        rows = sorted({self.list.row(i) for i in self.list.selectedItems()})
        if not rows and self._current_idx >= 0:
            rows = [self._current_idx]
        self._start_batch(rows)

    def _process_all(self) -> None:
        self._start_batch(list(range(len(self.items))))

    def _start_batch(self, rows: list[int]) -> None:
        if self._busy or not rows:
            return
        predictor = self._predictor()
        if predictor is None:
            return
        files = [self.items[r].path for r in rows]
        # Worker cukup menghitung alpha mentah: RefineParams() default = identitas,
        # refine/background asli diterapkan saat preview & ekspor.
        opts = BatchOptions(refine=RefineParams(), background=BackgroundSpec(decontaminate=False))
        self._worker = BatchWorker(files, predictor, opts)
        self._worker.progress.connect(self._on_batch_progress)
        self._worker.finished.connect(self._on_batch_done)
        self._worker.failed.connect(self._on_batch_failed)

        self.progress.setRange(0, len(files))
        self.progress.setValue(0)
        self._set_busy(True)
        if predictor is self.engine and not is_downloaded(self.engine.spec):
            spec = self.engine.spec
            self._status(f"Mengunduh model {spec.label} (~{spec.approx_size_mb} MB), hanya sekali…")
        else:
            self._status(f"Memproses 0/{len(files)}…")
        self._thread = start_worker(self._worker, self)

    def _on_batch_progress(self, done: int, total: int, res: ProcessResult) -> None:
        idx = self._path_index.get(res.source)
        if idx is not None:
            it = self.items[idx]
            if res.ok and res.alpha is not None:
                it.alpha_u8 = (np.clip(res.alpha, 0, 1) * 255 + 0.5).astype(np.uint8)
                it.error = None
            else:
                it.error = res.error
            self._refresh_row(idx)
            if idx == self._current_idx:
                self._render_current()
        res.image = res.alpha = None  # lepas memori, hasil akhir dirender ulang saat ekspor
        self.progress.setValue(done)
        self._status(f"Memproses {done}/{total}…")

    def _on_batch_done(self, results: list) -> None:
        ok = sum(1 for r in results if r.ok)
        fail = len(results) - ok
        cancelled = len(results) < self.progress.maximum()
        self._finish_batch(f"Selesai: {ok} berhasil, {fail} gagal" + (" (dibatalkan)" if cancelled else ""))

    def _on_batch_failed(self, msg: str) -> None:
        self._finish_batch("Proses gagal")
        QMessageBox.critical(self, "Proses gagal", msg)

    def _finish_batch(self, msg: str) -> None:
        self._worker = None
        self._thread = None
        self._set_busy(False)
        self._status(msg)

    def _cancel_batch(self) -> None:
        if self._worker is not None:
            self._worker.cancel()
            self._status("Membatalkan setelah gambar yang sedang diproses…")

    # ------------------------------------------------------------- ekspor
    def _can_export(self) -> bool:
        if self._busy:
            return False
        if not any(it.alpha_u8 is not None for it in self.items):
            QMessageBox.information(self, "Ekspor", "Belum ada gambar yang diproses.")
            return False
        return self._bg_valid()

    def _progress_dialog(self, title: str) -> QProgressDialog:
        dlg = QProgressDialog(title, "Batal", 0, 0, self)
        dlg.setWindowTitle(title)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        dlg.setAutoClose(False)
        dlg.setAutoReset(False)
        dlg.show()
        return dlg

    def _iter_export_results(self, dlg: QProgressDialog) -> Iterator[ProcessResult]:
        """Render satu per satu (streaming) agar RAM tetap kecil untuk batch besar."""
        targets = [it for it in self.items if it.alpha_u8 is not None or it.error]
        dlg.setMaximum(len(targets))
        for i, it in enumerate(targets):
            if dlg.wasCanceled():
                return
            dlg.setValue(i)
            dlg.setLabelText(f"Merender {it.path.name} ({i + 1}/{len(targets)})…")
            QApplication.processEvents()
            if it.alpha_u8 is None:
                yield ProcessResult(source=it.path, error=it.error)
                continue
            try:
                out, _ = process_image(load_image(it.path), self.engine,
                                       self._collect_opts(it.strokes), alpha=it.alpha)
                res = ProcessResult(source=it.path, image=out)
            except Exception as exc:
                res = ProcessResult(source=it.path, error=str(exc))
            yield res
        dlg.setValue(len(targets))

    def _export_zip(self) -> None:
        if not self._can_export():
            return
        path, _ = QFileDialog.getSaveFileName(self, "Ekspor ZIP", "hasil_nobg.zip", "ZIP (*.zip)")
        if not path:
            return
        zp = Path(path) if path.lower().endswith(".zip") else Path(path + ".zip")
        dlg = self._progress_dialog("Mengekspor ZIP")
        try:
            count = export_zip(self._iter_export_results(dlg), zp, BatchOptions())
            cancelled = dlg.wasCanceled()
        except Exception as exc:
            dlg.close()
            QMessageBox.critical(self, "Ekspor ZIP", f"Gagal menulis ZIP: {exc}")
            return
        dlg.close()
        if cancelled:
            zp.unlink(missing_ok=True)
            self._status("Ekspor ZIP dibatalkan")
        else:
            self._status(f"{count} gambar diekspor ke {zp}")

    def _export_folder(self) -> None:
        if not self._can_export():
            return
        d = QFileDialog.getExistingDirectory(self, "Folder tujuan")
        if not d:
            return
        dlg = self._progress_dialog("Mengekspor ke folder")
        try:
            written = export_folder(self._iter_export_results(dlg), Path(d), BatchOptions())
            cancelled = dlg.wasCanceled()
        except Exception as exc:
            dlg.close()
            QMessageBox.critical(self, "Ekspor", f"Gagal menulis file: {exc}")
            return
        dlg.close()
        suffix = " (dibatalkan sebagian)" if cancelled else ""
        self._status(f"{len(written)} gambar diekspor ke {d}{suffix}")

    def _save_current(self) -> None:
        it = self._current_item()
        if it is None or it.alpha_u8 is None or self._current_img is None:
            self._status("Proses gambar ini dulu sebelum disimpan")
            return
        if not self._bg_valid():
            return
        opts = self._collect_opts(it.strokes)
        out, _ = process_image(self._current_img, self.engine, opts, alpha=it.alpha)
        transparent = out.mode == "RGBA"
        default = it.path.with_name(output_name(it.path, opts.background))
        filt = "PNG (*.png)" if transparent else "JPEG (*.jpg *.jpeg);;PNG (*.png)"
        path, _ = QFileDialog.getSaveFileName(self, "Simpan gambar", str(default), filt)
        if not path:
            return
        p = Path(path)
        if transparent and p.suffix.lower() != ".png":
            p = p.with_suffix(".png")  # JPEG tidak mendukung transparansi
        elif not p.suffix:
            p = p.with_suffix(".jpg")
        try:
            if p.suffix.lower() in (".jpg", ".jpeg"):
                out.save(p, quality=95, subsampling=0)
            else:
                out.save(p)
        except Exception as exc:
            QMessageBox.critical(self, "Simpan", f"Gagal menyimpan: {exc}")
            return
        self._status(f"Tersimpan: {p}")

    # ------------------------------------------------------------- lifecycle
    def closeEvent(self, event) -> None:
        if self._worker is not None:
            self._worker.cancel()
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(10_000)  # tunggu gambar yang sedang diproses selesai
        super().closeEvent(event)