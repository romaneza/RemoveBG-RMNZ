"""Worker thread agar inferensi tidak membekukan UI."""
from __future__ import annotations

import threading
from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal

from app.core.pipeline import BatchOptions, ProcessResult, run_batch


class BatchWorker(QObject):
    progress = Signal(int, int, object)   # done, total, ProcessResult
    finished = Signal(list)               # list[ProcessResult]
    failed = Signal(str)

    def __init__(self, files: list[Path], predictor, opts: BatchOptions):
        super().__init__()
        self.files = files
        self.predictor = predictor
        self.opts = opts
        self.cancel_event = threading.Event()

    def run(self) -> None:
        try:
            results = run_batch(
                self.files,
                self.predictor,
                self.opts,
                progress=lambda i, n, r: self.progress.emit(i, n, r),
                cancel=self.cancel_event,
            )
            self.finished.emit(results)
        except Exception as exc:  # error fatal (mis. model gagal diunduh)
            self.failed.emit(str(exc))

    def cancel(self) -> None:
        self.cancel_event.set()


def start_worker(worker: QObject, parent: QObject) -> QThread:
    """Pindahkan worker ke QThread baru dan jalankan `worker.run`."""
    thread = QThread(parent)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)
    thread.start()
    return thread


__all__ = ["BatchWorker", "ProcessResult", "start_worker"]