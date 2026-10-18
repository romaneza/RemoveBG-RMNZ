"""Entry point: python -m app.main"""
from __future__ import annotations

import logging
import sys

from PySide6.QtWidgets import QApplication

from app import __version__
from app.ui.main_window import MainWindow


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app = QApplication(sys.argv)
    app.setApplicationName("RemoveBG-RMNZ")
    app.setApplicationDisplayName("RemoveBG-RMNZ")
    app.setApplicationVersion(__version__)
    app.setOrganizationName("RMNZ")

    win = MainWindow()
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())