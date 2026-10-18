"""Canvas editor: preview hasil + manual edge refinement (brush keep/erase).

- Scroll         : zoom
- Spasi + drag   : pan (atau tombol tengah mouse)
- Klik kiri drag : brush sesuai mode aktif
- [ / ]          : kecilkan / besarkan brush
"""
from __future__ import annotations

from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (
    QGraphicsEllipseItem,
    QGraphicsPathItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

from app.core.refine import BrushMode, BrushStroke


def pil_to_qpixmap(img: Image.Image) -> QPixmap:
    img = img.convert("RGBA")
    data = img.tobytes("raw", "RGBA")
    qimg = QImage(data, img.width, img.height, img.width * 4, QImage.Format_RGBA8888)
    return QPixmap.fromImage(qimg.copy())  # copy agar tidak bergantung ke buffer `data`


def _checker_brush(size: int = 12) -> QBrush:
    tile = QPixmap(size * 2, size * 2)
    tile.fill(QColor(255, 255, 255))
    p = QPainter(tile)
    p.fillRect(0, 0, size, size, QColor(204, 204, 204))
    p.fillRect(size, size, size, size, QColor(204, 204, 204))
    p.end()
    return QBrush(tile)


class MaskCanvas(QGraphicsView):
    strokeFinished = Signal(object)   # BrushStroke
    brushRadiusChanged = Signal(float)

    MIN_ZOOM, MAX_ZOOM = 0.05, 32.0

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.SmoothPixmapTransform)
        self.setRenderHint(QPainter.Antialiasing)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setBackgroundBrush(_checker_brush())
        self.setMouseTracking(True)
        self.setAccessibleName("Kanvas preview dan editor tepi")

        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.SmoothTransformation)
        self.scene().addItem(self._pixmap_item)

        self._cursor = QGraphicsEllipseItem()
        self._cursor.setPen(QPen(QColor(255, 255, 255), 0))
        self._cursor.setZValue(10)
        self._cursor.setVisible(False)
        self.scene().addItem(self._cursor)

        self._live_path = QGraphicsPathItem()
        self._live_path.setZValue(5)
        self.scene().addItem(self._live_path)

        self.brush_mode = BrushMode.ERASE
        self.brush_radius = 20.0
        self.brush_hardness = 0.7
        self.brush_enabled = True

        self._points: list[tuple[float, float]] = []
        self._painting = False
        self._space_down = False
        self._has_image = False

    # ---- gambar ----
    def set_image(self, img: Image.Image | None, fit: bool = False) -> None:
        if img is None:
            self._pixmap_item.setPixmap(QPixmap())
            self._has_image = False
            return
        self._pixmap_item.setPixmap(pil_to_qpixmap(img))
        self.scene().setSceneRect(QRectF(0, 0, img.width, img.height))
        self._has_image = True
        if fit:
            self.fit()

    def fit(self) -> None:
        if self._has_image:
            self.fitInView(self._pixmap_item, Qt.KeepAspectRatio)

    # ---- brush ----
    def set_brush_mode(self, mode: BrushMode) -> None:
        self.brush_mode = mode

    def set_brush_radius(self, r: float) -> None:
        self.brush_radius = max(1.0, min(500.0, r))
        self.brushRadiusChanged.emit(self.brush_radius)
        self._update_cursor(self._cursor.rect().center())

    def _stroke_color(self) -> QColor:
        return QColor(0, 200, 80, 120) if self.brush_mode == BrushMode.KEEP else QColor(230, 40, 40, 120)

    def _update_cursor(self, center: QPointF) -> None:
        r = self.brush_radius
        self._cursor.setRect(center.x() - r, center.y() - r, 2 * r, 2 * r)

    def _can_paint(self) -> bool:
        return self._has_image and self.brush_enabled and not self._space_down

    # ---- event ----
    def wheelEvent(self, event) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        current = self.transform().m11()
        if self.MIN_ZOOM <= current * factor <= self.MAX_ZOOM:
            self.scale(factor, factor)

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_down = True
            self.setDragMode(QGraphicsView.ScrollHandDrag)
        elif event.key() == Qt.Key_BracketLeft:
            self.set_brush_radius(self.brush_radius * 0.8)
        elif event.key() == Qt.Key_BracketRight:
            self.set_brush_radius(self.brush_radius * 1.25)
        else:
            super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key_Space and not event.isAutoRepeat():
            self._space_down = False
            self.setDragMode(QGraphicsView.NoDrag)
        else:
            super().keyReleaseEvent(event)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            self.setDragMode(QGraphicsView.ScrollHandDrag)
            # Qt butuh event tombol kiri untuk memulai hand-drag
            fake = event.__class__(event.type(), event.position(), event.globalPosition(),
                                   Qt.LeftButton, Qt.LeftButton, event.modifiers())
            super().mousePressEvent(fake)
            return
        if event.button() == Qt.LeftButton and self._can_paint():
            self._painting = True
            pos = self.mapToScene(event.position().toPoint())
            self._points = [(pos.x(), pos.y())]
            self._redraw_live()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        pos = self.mapToScene(event.position().toPoint())
        self._cursor.setVisible(self._can_paint())
        self._update_cursor(pos)
        if self._painting:
            last = self._points[-1]
            # sampling: tambahkan titik bila bergeser > 1/4 radius
            if (pos.x() - last[0]) ** 2 + (pos.y() - last[1]) ** 2 > (self.brush_radius / 4) ** 2:
                self._points.append((pos.x(), pos.y()))
                self._redraw_live()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MiddleButton:
            super().mouseReleaseEvent(event)
            self.setDragMode(QGraphicsView.NoDrag)
            return
        if event.button() == Qt.LeftButton and self._painting:
            self._painting = False
            stroke = BrushStroke(
                mode=self.brush_mode,
                points=list(self._points),
                radius=self.brush_radius,
                hardness=self.brush_hardness,
            )
            self._points.clear()
            self._live_path.setPath(QPainterPath())
            self.strokeFinished.emit(stroke)
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:
        self._cursor.setVisible(False)
        super().leaveEvent(event)

    def _redraw_live(self) -> None:
        path = QPainterPath()
        x0, y0 = self._points[0]
        path.moveTo(x0, y0)
        for x, y in self._points[1:]:
            path.lineTo(x, y)
        if len(self._points) == 1:
            path.addEllipse(QPointF(x0, y0), self.brush_radius, self.brush_radius)
        pen = QPen(self._stroke_color(), 2 * self.brush_radius, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        self._live_path.setPen(pen)
        self._live_path.setBrush(self._stroke_color() if len(self._points) == 1 else Qt.NoBrush)
        self._live_path.setPath(path)