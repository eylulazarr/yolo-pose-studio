from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml
from PySide6.QtCore import QPointF, QRectF, Qt, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGraphicsEllipseItem,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressDialog,
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ui.base_page import BasePage


SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


@dataclass
class PoseKeypoint:
    x: float = 0.0
    y: float = 0.0
    visibility: int = 0


@dataclass
class PoseObject:
    class_id: int = 0
    bbox: QRectF | None = None
    keypoints: list[PoseKeypoint] = field(default_factory=list)


class AnnotationCanvas(QGraphicsView):
    """Görsel, bbox ve keypoint çizimi/düzenlemesi için etkileşimli canvas."""

    bbox_created = Signal(QRectF)
    keypoint_placed = Signal(int, QPointF)
    canvas_clicked = Signal(QPointF)

    object_selected = Signal(int)
    keypoint_selected = Signal(int, int)
    edit_started = Signal()
    bbox_edited = Signal(int, QRectF)
    keypoint_edited = Signal(int, int, QPointF)
    edit_finished = Signal()

    MODE_SELECT = "select"
    MODE_BBOX = "bbox"
    MODE_KEYPOINT = "keypoint"

    DRAG_KEYPOINT = "keypoint"
    DRAG_BBOX_MOVE = "bbox_move"
    DRAG_BBOX_RESIZE = "bbox_resize"

    HANDLE_TOP_LEFT = "top_left"
    HANDLE_TOP_RIGHT = "top_right"
    HANDLE_BOTTOM_LEFT = "bottom_left"
    HANDLE_BOTTOM_RIGHT = "bottom_right"

    def __init__(self) -> None:
        super().__init__()

        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)

        self._pixmap_item: QGraphicsPixmapItem | None = None
        self._overlay_items: list[Any] = []
        self._temporary_bbox_item: QGraphicsRectItem | None = None

        self._mode = self.MODE_SELECT
        self._bbox_start: QPointF | None = None
        self._active_keypoint_index = 0
        self._active_object_index = -1
        self._has_image = False
        self._objects_snapshot: list[PoseObject] = []
        # Görsel ilk açıldığında ve pencere boyutu değiştiğinde otomatik sığdır.
        # Kullanıcı mouse wheel ile zoom yaparsa otomatik sığdırma kapanır.
        self._auto_fit_enabled = True

        self._drag_kind: str | None = None
        self._drag_object_index = -1
        self._drag_keypoint_index = -1
        self._drag_handle: str | None = None
        self._drag_start_point: QPointF | None = None
        self._drag_start_bbox: QRectF | None = None
        self._drag_edit_started = False

        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setBackgroundBrush(QColor("#151922"))
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setTransformationAnchor(
            QGraphicsView.ViewportAnchor.AnchorUnderMouse
        )
        self.setResizeAnchor(
            QGraphicsView.ViewportAnchor.AnchorViewCenter
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.setMinimumSize(520, 420)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

    @property
    def has_image(self) -> bool:
        return self._has_image

    @property
    def image_rect(self) -> QRectF:
        if self._pixmap_item is None:
            return QRectF()

        return self._pixmap_item.boundingRect()

    def set_mode(self, mode: str) -> None:
        self._reset_drag_state()
        self._mode = mode

        if mode == self.MODE_SELECT:
            self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def set_active_keypoint_index(self, index: int) -> None:
        self._active_keypoint_index = max(0, index)

    def load_image(self, image_path: Path) -> bool:
        pixmap = QPixmap(str(image_path))

        if pixmap.isNull():
            self.clear_canvas()
            return False

        self._scene.clear()
        self._overlay_items.clear()
        self._temporary_bbox_item = None
        self._reset_drag_state()

        self._pixmap_item = self._scene.addPixmap(pixmap)
        self._pixmap_item.setZValue(0)

        rect = self._pixmap_item.boundingRect()
        self._scene.setSceneRect(rect)
        self._has_image = True

        self._auto_fit_enabled = True
        self.fit_image()

        return True

    def fit_image(self) -> None:
        """Görselin tamamını canvas içine, oranını bozmadan sığdırır."""
        if not self._has_image or self._pixmap_item is None:
            return

        rect = self._pixmap_item.boundingRect()
        if rect.isEmpty():
            return

        self._auto_fit_enabled = True
        self.resetTransform()
        self.fitInView(
            rect,
            Qt.AspectRatioMode.KeepAspectRatio,
        )

    def resizeEvent(self, event) -> None:
        """Canvas küçülüp büyüdüğünde görselin kesilmesini engeller."""
        super().resizeEvent(event)

        if (
            self._auto_fit_enabled
            and self._has_image
            and self._pixmap_item is not None
        ):
            self.fitInView(
                self._pixmap_item.boundingRect(),
                Qt.AspectRatioMode.KeepAspectRatio,
            )

    def clear_canvas(self) -> None:
        self._scene.clear()
        self._pixmap_item = None
        self._overlay_items.clear()
        self._temporary_bbox_item = None
        self._bbox_start = None
        self._objects_snapshot = []
        self._active_object_index = -1
        self._has_image = False
        self._reset_drag_state()

    def redraw(
        self,
        *,
        objects: list[PoseObject],
        active_object_index: int,
        active_keypoint_index: int,
        class_names: dict[int, str],
        skeleton: list[tuple[int, int]],
    ) -> None:
        self._objects_snapshot = objects
        self._active_object_index = active_object_index
        self._active_keypoint_index = active_keypoint_index

        self._clear_overlays()

        if not self._has_image:
            return

        for object_index, pose_object in enumerate(objects):
            is_active = object_index == active_object_index

            self._draw_object(
                pose_object=pose_object,
                object_index=object_index,
                is_active=is_active,
                active_keypoint_index=active_keypoint_index,
                class_names=class_names,
                skeleton=skeleton,
            )

    def _clear_overlays(self) -> None:
        for item in self._overlay_items:
            if item.scene() is self._scene:
                self._scene.removeItem(item)

        self._overlay_items.clear()

        if (
            self._temporary_bbox_item is not None
            and self._temporary_bbox_item.scene() is self._scene
        ):
            self._scene.removeItem(self._temporary_bbox_item)

        self._temporary_bbox_item = None

    def _draw_object(
        self,
        *,
        pose_object: PoseObject,
        object_index: int,
        is_active: bool,
        active_keypoint_index: int,
        class_names: dict[int, str],
        skeleton: list[tuple[int, int]],
    ) -> None:
        bbox_color = (
            QColor("#16d9ff") if is_active else QColor("#42d392")
        )

        if pose_object.bbox is not None:
            bbox = pose_object.bbox.normalized()
            bbox_item = QGraphicsRectItem(bbox)
            bbox_item.setPen(QPen(bbox_color, 3 if is_active else 2))
            bbox_item.setZValue(3)
            self._scene.addItem(bbox_item)
            self._overlay_items.append(bbox_item)

            class_name = class_names.get(
                pose_object.class_id,
                f"class_{pose_object.class_id}",
            )
            label_item = QGraphicsSimpleTextItem(
                f"{class_name} #{object_index + 1}"
            )
            label_item.setBrush(bbox_color)
            label_item.setPos(
                bbox.left(),
                max(0.0, bbox.top() - 22.0),
            )
            label_item.setZValue(5)
            self._scene.addItem(label_item)
            self._overlay_items.append(label_item)

            if is_active:
                self._draw_bbox_handles(bbox)

        for start_index, end_index in skeleton:
            if (
                start_index >= len(pose_object.keypoints)
                or end_index >= len(pose_object.keypoints)
            ):
                continue

            start_point = pose_object.keypoints[start_index]
            end_point = pose_object.keypoints[end_index]

            if (
                start_point.visibility == 0
                or end_point.visibility == 0
            ):
                continue

            line_item = QGraphicsLineItem(
                start_point.x,
                start_point.y,
                end_point.x,
                end_point.y,
            )
            line_color = (
                QColor("#00e5ff")
                if is_active
                else QColor("#7786ff")
            )
            line_item.setPen(QPen(line_color, 3 if is_active else 2))
            line_item.setZValue(4)
            self._scene.addItem(line_item)
            self._overlay_items.append(line_item)

        for keypoint_index, keypoint in enumerate(pose_object.keypoints):
            if keypoint.visibility == 0:
                continue

            point_color = (
                QColor("#ffd54a")
                if keypoint.visibility == 1
                else QColor("#ff4f64")
            )

            if is_active and keypoint_index == active_keypoint_index:
                radius = 7.0
                pen = QPen(QColor("#ffffff"), 2)
            else:
                radius = 5.0
                pen = QPen(QColor("#10131a"), 1)

            ellipse_item = QGraphicsEllipseItem(
                keypoint.x - radius,
                keypoint.y - radius,
                radius * 2,
                radius * 2,
            )
            ellipse_item.setBrush(point_color)
            ellipse_item.setPen(pen)
            ellipse_item.setZValue(6)
            self._scene.addItem(ellipse_item)
            self._overlay_items.append(ellipse_item)

            index_item = QGraphicsSimpleTextItem(str(keypoint_index))
            index_item.setBrush(QColor("#63b3ff"))
            index_item.setPos(keypoint.x + 7, keypoint.y - 17)
            index_item.setZValue(7)
            self._scene.addItem(index_item)
            self._overlay_items.append(index_item)

    def _draw_bbox_handles(self, bbox: QRectF) -> None:
        handle_size = 10.0
        half_size = handle_size / 2.0

        for point in self._bbox_handle_points(bbox).values():
            handle_item = QGraphicsRectItem(
                point.x() - half_size,
                point.y() - half_size,
                handle_size,
                handle_size,
            )
            handle_item.setBrush(QColor("#ffffff"))
            handle_item.setPen(QPen(QColor("#16d9ff"), 2))
            handle_item.setZValue(9)
            self._scene.addItem(handle_item)
            self._overlay_items.append(handle_item)

    @staticmethod
    def _bbox_handle_points(bbox: QRectF) -> dict[str, QPointF]:
        rect = bbox.normalized()
        return {
            AnnotationCanvas.HANDLE_TOP_LEFT: rect.topLeft(),
            AnnotationCanvas.HANDLE_TOP_RIGHT: rect.topRight(),
            AnnotationCanvas.HANDLE_BOTTOM_LEFT: rect.bottomLeft(),
            AnnotationCanvas.HANDLE_BOTTOM_RIGHT: rect.bottomRight(),
        }

    def _clip_to_image(self, point: QPointF) -> QPointF:
        rect = self.image_rect

        x = min(max(point.x(), rect.left()), rect.right())
        y = min(max(point.y(), rect.top()), rect.bottom())

        return QPointF(x, y)

    def _scene_tolerance(self, pixels: float = 11.0) -> float:
        scale = abs(self.transform().m11())
        return pixels / max(scale, 0.05)

    def _interaction_object_order(self) -> list[int]:
        indices = list(range(len(self._objects_snapshot)))

        if self._active_object_index in indices:
            indices.remove(self._active_object_index)
            indices.append(self._active_object_index)

        indices.reverse()
        return indices

    def _hit_test(
        self,
        scene_point: QPointF,
    ) -> tuple[str, int, int | str | None] | None:
        tolerance = self._scene_tolerance()
        tolerance_squared = tolerance * tolerance

        for object_index in self._interaction_object_order():
            pose_object = self._objects_snapshot[object_index]

            for keypoint_index, keypoint in enumerate(
                pose_object.keypoints
            ):
                if keypoint.visibility == 0:
                    continue

                dx = keypoint.x - scene_point.x()
                dy = keypoint.y - scene_point.y()

                if dx * dx + dy * dy <= tolerance_squared:
                    return (
                        self.DRAG_KEYPOINT,
                        object_index,
                        keypoint_index,
                    )

            if pose_object.bbox is None:
                continue

            bbox = pose_object.bbox.normalized()

            for handle_name, handle_point in self._bbox_handle_points(
                bbox
            ).items():
                dx = handle_point.x() - scene_point.x()
                dy = handle_point.y() - scene_point.y()

                if dx * dx + dy * dy <= tolerance_squared:
                    return (
                        self.DRAG_BBOX_RESIZE,
                        object_index,
                        handle_name,
                    )

            if bbox.contains(scene_point):
                return (
                    self.DRAG_BBOX_MOVE,
                    object_index,
                    None,
                )

        return None

    def _set_hover_cursor(self, scene_point: QPointF) -> None:
        if self._mode != self.MODE_SELECT or self._drag_kind is not None:
            return

        hit = self._hit_test(scene_point)

        if hit is None:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)
            return

        drag_kind, _, detail = hit

        if drag_kind == self.DRAG_KEYPOINT:
            cursor = Qt.CursorShape.PointingHandCursor
        elif drag_kind == self.DRAG_BBOX_MOVE:
            cursor = Qt.CursorShape.SizeAllCursor
        elif detail in {
            self.HANDLE_TOP_LEFT,
            self.HANDLE_BOTTOM_RIGHT,
        }:
            cursor = Qt.CursorShape.SizeFDiagCursor
        else:
            cursor = Qt.CursorShape.SizeBDiagCursor

        self.viewport().setCursor(cursor)

    def _ensure_edit_started(self) -> None:
        if self._drag_edit_started:
            return

        self._drag_edit_started = True
        self.edit_started.emit()

    def _reset_drag_state(self) -> None:
        self._drag_kind = None
        self._drag_object_index = -1
        self._drag_keypoint_index = -1
        self._drag_handle = None
        self._drag_start_point = None
        self._drag_start_bbox = None
        self._drag_edit_started = False

    def _move_bbox_within_image(
        self,
        bbox: QRectF,
        delta: QPointF,
    ) -> QRectF:
        image_rect = self.image_rect
        rect = bbox.normalized()

        if rect.width() >= image_rect.width():
            left = image_rect.left()
            width = image_rect.width()
        else:
            width = rect.width()
            left = min(
                max(rect.left() + delta.x(), image_rect.left()),
                image_rect.right() - width,
            )

        if rect.height() >= image_rect.height():
            top = image_rect.top()
            height = image_rect.height()
        else:
            height = rect.height()
            top = min(
                max(rect.top() + delta.y(), image_rect.top()),
                image_rect.bottom() - height,
            )

        return QRectF(left, top, width, height)

    def _resize_bbox(
        self,
        bbox: QRectF,
        handle: str,
        point: QPointF,
    ) -> QRectF:
        rect = bbox.normalized()
        point = self._clip_to_image(point)
        minimum_size = max(4.0, self._scene_tolerance(5.0))

        left = rect.left()
        right = rect.right()
        top = rect.top()
        bottom = rect.bottom()

        if handle == self.HANDLE_TOP_LEFT:
            left = min(point.x(), right - minimum_size)
            top = min(point.y(), bottom - minimum_size)
        elif handle == self.HANDLE_TOP_RIGHT:
            right = max(point.x(), left + minimum_size)
            top = min(point.y(), bottom - minimum_size)
        elif handle == self.HANDLE_BOTTOM_LEFT:
            left = min(point.x(), right - minimum_size)
            bottom = max(point.y(), top + minimum_size)
        elif handle == self.HANDLE_BOTTOM_RIGHT:
            right = max(point.x(), left + minimum_size)
            bottom = max(point.y(), top + minimum_size)

        resized = QRectF(
            QPointF(left, top),
            QPointF(right, bottom),
        ).normalized()

        return resized.intersected(self.image_rect)

    def mousePressEvent(self, event) -> None:
        if (
            not self._has_image
            or event.button() != Qt.MouseButton.LeftButton
        ):
            super().mousePressEvent(event)
            return

        scene_point = self.mapToScene(event.position().toPoint())

        if not self.image_rect.contains(scene_point):
            super().mousePressEvent(event)
            return

        scene_point = self._clip_to_image(scene_point)
        self.canvas_clicked.emit(scene_point)

        if self._mode == self.MODE_BBOX:
            self._bbox_start = scene_point
            self._temporary_bbox_item = QGraphicsRectItem(
                QRectF(scene_point, scene_point)
            )
            self._temporary_bbox_item.setPen(
                QPen(
                    QColor("#16d9ff"),
                    2,
                    Qt.PenStyle.DashLine,
                )
            )
            self._temporary_bbox_item.setZValue(10)
            self._scene.addItem(self._temporary_bbox_item)
            event.accept()
            return

        if self._mode == self.MODE_KEYPOINT:
            self.keypoint_placed.emit(
                self._active_keypoint_index,
                scene_point,
            )
            event.accept()
            return

        hit = self._hit_test(scene_point)

        if hit is None:
            super().mousePressEvent(event)
            return

        drag_kind, object_index, detail = hit
        self.object_selected.emit(object_index)

        self._drag_kind = drag_kind
        self._drag_object_index = object_index
        self._drag_start_point = scene_point
        self._drag_edit_started = False

        pose_object = self._objects_snapshot[object_index]

        if drag_kind == self.DRAG_KEYPOINT:
            self._drag_keypoint_index = int(detail)
            self.keypoint_selected.emit(
                object_index,
                self._drag_keypoint_index,
            )
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
        else:
            self._drag_start_bbox = (
                QRectF(pose_object.bbox)
                if pose_object.bbox is not None
                else None
            )

            if drag_kind == self.DRAG_BBOX_RESIZE:
                self._drag_handle = str(detail)
            else:
                self.viewport().setCursor(Qt.CursorShape.SizeAllCursor)

        event.accept()

    def mouseMoveEvent(self, event) -> None:
        scene_point = self.mapToScene(event.position().toPoint())

        if (
            self._mode == self.MODE_BBOX
            and self._bbox_start is not None
            and self._temporary_bbox_item is not None
        ):
            scene_point = self._clip_to_image(scene_point)
            rect = QRectF(self._bbox_start, scene_point).normalized()
            self._temporary_bbox_item.setRect(rect)
            event.accept()
            return

        if self._drag_kind is not None:
            scene_point = self._clip_to_image(scene_point)

            if self._drag_kind == self.DRAG_KEYPOINT:
                self._ensure_edit_started()
                self.keypoint_edited.emit(
                    self._drag_object_index,
                    self._drag_keypoint_index,
                    scene_point,
                )

            elif (
                self._drag_kind == self.DRAG_BBOX_MOVE
                and self._drag_start_point is not None
                and self._drag_start_bbox is not None
            ):
                self._ensure_edit_started()
                delta = scene_point - self._drag_start_point
                new_bbox = self._move_bbox_within_image(
                    self._drag_start_bbox,
                    delta,
                )
                self.bbox_edited.emit(
                    self._drag_object_index,
                    new_bbox,
                )

            elif (
                self._drag_kind == self.DRAG_BBOX_RESIZE
                and self._drag_start_bbox is not None
                and self._drag_handle is not None
            ):
                self._ensure_edit_started()
                new_bbox = self._resize_bbox(
                    self._drag_start_bbox,
                    self._drag_handle,
                    scene_point,
                )
                self.bbox_edited.emit(
                    self._drag_object_index,
                    new_bbox,
                )

            self.canvas_clicked.emit(scene_point)
            event.accept()
            return

        if self._mode == self.MODE_SELECT and self._has_image:
            if self.image_rect.contains(scene_point):
                self._set_hover_cursor(scene_point)
            else:
                self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if (
            self._mode == self.MODE_BBOX
            and self._bbox_start is not None
            and self._temporary_bbox_item is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            rect = (
                self._temporary_bbox_item.rect()
                .normalized()
                .intersected(self.image_rect)
            )

            self._scene.removeItem(self._temporary_bbox_item)
            self._temporary_bbox_item = None
            self._bbox_start = None

            if rect.width() >= 4 and rect.height() >= 4:
                self.bbox_created.emit(rect)

            event.accept()
            return

        if (
            self._drag_kind is not None
            and event.button() == Qt.MouseButton.LeftButton
        ):
            edit_started = self._drag_edit_started
            scene_point = self.mapToScene(event.position().toPoint())
            self._reset_drag_state()

            if edit_started:
                self.edit_finished.emit()

            if self._mode == self.MODE_SELECT:
                if self.image_rect.contains(scene_point):
                    self._set_hover_cursor(scene_point)
                else:
                    self.viewport().setCursor(
                        Qt.CursorShape.OpenHandCursor
                    )

            event.accept()
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:
        if not self._has_image:
            return

        # Kullanıcı manuel zoom yaptı; pencere resize olunca zoomu bozmayalım.
        self._auto_fit_enabled = False

        factor = 1.18 if event.angleDelta().y() > 0 else 1 / 1.18
        current_scale = self.transform().m11()
        next_scale = current_scale * factor

        if 0.05 <= next_scale <= 30:
            self.scale(factor, factor)

        event.accept()

    def leaveEvent(self, event) -> None:
        if self._mode == self.MODE_SELECT and self._drag_kind is None:
            self.viewport().setCursor(Qt.CursorShape.OpenHandCursor)

        super().leaveEvent(event)


class AnnotationPage(BasePage):
    """YOLO Pose için 12 gövde keypoint etiketleme ve otomatik etiketleme ekranı."""

    def __init__(self) -> None:
        super().__init__(
            title="Pose Veri Etiketleme",
            description=(
                "Kullanıcı data.yaml ve images klasörünü seçerek görseller "
                "üzerinde bounding box ve pose keypoint etiketleri oluşturabilir."
            ),
        )

        self.data_yaml_path: Path | None = None
        self.images_directory: Path | None = None
        self.labels_directory: Path | None = None

        self.image_paths: list[Path] = []
        self.current_image_index = -1
        self.current_image_path: Path | None = None

        self.class_names: dict[int, str] = {}
        self.keypoint_count = 0
        self.keypoint_dimensions = 3
        self.keypoint_names: list[str] = []
        self.skeleton: list[tuple[int, int]] = []

        self.annotations: list[PoseObject] = []
        self.active_object_index = -1
        self.active_keypoint_index = 0

        self.is_dirty = False
        self.undo_stack: list[list[PoseObject]] = []

        self.data_yaml_input = QLineEdit()
        self.images_input = QLineEdit()
        self.labels_input = QLineEdit()

        self.class_combo = QComboBox()
        self.visibility_combo = QComboBox()
        self.object_list = QListWidget()
        self.keypoint_list = QListWidget()

        self.progress_label = QLabel(
            "Dataset yüklenmedi"
        )
        self.image_name_label = QLabel(
            "Görsel seçilmedi"
        )
        self.coordinate_label = QLabel(
            "x: -, y: -"
        )

        self.auto_save_checkbox = QCheckBox(
            "Görsel değiştirirken otomatik kaydet"
        )
        self.auto_save_checkbox.setChecked(True)

        self.canvas = AnnotationCanvas()

        self.previous_button = QPushButton("← Önceki")
        self.next_button = QPushButton("Sonraki →")
        self.save_button = QPushButton("Etiketi Kaydet")
        self.undo_button = QPushButton("Geri Al")
        self.open_labels_button = QPushButton(
            "Labels Klasörünü Aç"
        )
        self.fit_image_button = QPushButton(
            "⛶ Görseli Ekrana Sığdır"
        )

        self.auto_label_button = QPushButton(
            "✨ Tüm Klasörü Otomatik Etiketle"
        )

        self._configure_inputs()
        self._build_ui()
        self._connect_signals()
        self._create_shortcuts()
        self._update_action_states()

    def _build_ui(self) -> None:
        page_layout = self.layout()

        # BasePage'in eski placeholder mesajı bu sayfada büyük boşluk bırakıyordu.
        # Annotation çalışma alanı gerçek modül olduğu için placeholder'ı gizle.
        for label in self.findChildren(QLabel):
            if "sonraki aşamalarda geliştirilecektir" in label.text().lower():
                label.hide()

        if page_layout is None:
            page_layout = QVBoxLayout(self)
            page_layout.setContentsMargins(30, 24, 30, 24)
            page_layout.setSpacing(12)
        else:
            page_layout.setSpacing(12)

        workspace = QSplitter(
            Qt.Orientation.Horizontal
        )
        workspace.setChildrenCollapsible(False)
        workspace.setHandleWidth(8)
        workspace.setMinimumHeight(580)
        workspace.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )

        control_scroll = QScrollArea()
        control_scroll.setWidgetResizable(True)
        control_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        control_scroll.setMinimumWidth(360)
        control_scroll.setMaximumWidth(460)

        control_widget = QWidget()
        control_layout = QVBoxLayout(control_widget)
        control_layout.setContentsMargins(4, 4, 8, 12)
        control_layout.setSpacing(14)

        control_layout.addWidget(
            self._create_dataset_card()
        )
        control_layout.addWidget(
            self._create_navigation_card()
        )
        control_layout.addWidget(
            self._create_object_card()
        )
        control_layout.addWidget(
            self._create_keypoint_card()
        )
        control_layout.addWidget(
            self._create_save_card()
        )
        control_layout.addStretch()

        control_scroll.setWidget(control_widget)

        canvas_card = QFrame()
        canvas_card.setObjectName("formCard")
        canvas_layout = QVBoxLayout(canvas_card)
        canvas_layout.setContentsMargins(14, 14, 14, 14)
        canvas_layout.setSpacing(10)

        canvas_header = QHBoxLayout()

        self.image_name_label.setObjectName(
            "sectionTitle"
        )
        self.image_name_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self.coordinate_label.setObjectName(
            "inputLabel"
        )

        canvas_header.addWidget(
            self.image_name_label,
            stretch=1,
        )

        self.fit_image_button.setMinimumHeight(34)
        self.fit_image_button.setToolTip(
            "Görselin tamamını çalışma alanına sığdır"
        )
        canvas_header.addWidget(
            self.fit_image_button
        )

        canvas_header.addWidget(
            self.coordinate_label
        )

        canvas_help = QLabel(
            "Fare tekerleği: zoom • Boş alanda sürükle: kaydır • "
            "Seç modunda bbox içinden taşı, beyaz köşelerden boyutlandır • "
            "Keypointi tutup sürükle: düzelt • Turkuaz çizgiler anatomik bağlantıları gösterir • "
            "BBox modunda sürükle: yeni kutu çiz"
        )
        canvas_help.setObjectName("inputLabel")
        canvas_help.setWordWrap(True)

        canvas_layout.addLayout(canvas_header)
        canvas_layout.addWidget(canvas_help)
        canvas_layout.addWidget(
            self.canvas,
            stretch=1,
        )

        workspace.addWidget(control_scroll)
        workspace.addWidget(canvas_card)
        workspace.setStretchFactor(0, 0)
        workspace.setStretchFactor(1, 1)
        workspace.setSizes([400, 980])

        self._insert_before_final_spacer(
            page_layout,
            workspace,
        )

    @staticmethod
    def _insert_before_final_spacer(
        layout,
        widget: QWidget,
    ) -> None:
        count = layout.count()

        if (
            count > 0
            and layout.itemAt(count - 1).spacerItem()
            is not None
        ):
            layout.insertWidget(count - 1, widget, 1)
        else:
            layout.addWidget(widget, 1)

    def _create_dataset_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("formCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("1. Dataset")
        title.setObjectName("sectionTitle")

        info = QLabel(
            "data.yaml, ham görseller ve label çıktı klasörünü seç."
        )
        info.setObjectName("inputLabel")
        info.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(info)

        layout.addWidget(
            self._create_path_field(
                title="data.yaml",
                line_edit=self.data_yaml_input,
                button_text="Dosya Seç",
                callback=self._select_data_yaml,
            )
        )
        layout.addWidget(
            self._create_path_field(
                title="Images klasörü",
                line_edit=self.images_input,
                button_text="Klasör Seç",
                callback=self._select_images_directory,
            )
        )
        layout.addWidget(
            self._create_path_field(
                title="Labels klasörü",
                line_edit=self.labels_input,
                button_text="Klasör Seç",
                callback=self._select_labels_directory,
            )
        )

        load_button = QPushButton("Dataseti Yükle")
        load_button.setObjectName("primaryButton")
        load_button.setMinimumHeight(42)
        load_button.clicked.connect(
            self._load_dataset
        )

        layout.addWidget(load_button)

        self.auto_label_button.setObjectName("primaryButton")
        self.auto_label_button.setMinimumHeight(42)
        self.auto_label_button.clicked.connect(
            self._auto_label_dataset
        )
        layout.addWidget(self.auto_label_button)

        return card

    def _create_navigation_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("formCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel("2. Görsel Gezinme")
        title.setObjectName("sectionTitle")

        self.progress_label.setObjectName(
            "inputLabel"
        )
        self.progress_label.setWordWrap(True)

        button_row = QHBoxLayout()
        button_row.setSpacing(8)

        self.previous_button.setMinimumHeight(38)
        self.next_button.setMinimumHeight(38)

        button_row.addWidget(self.previous_button)
        button_row.addWidget(self.next_button)

        layout.addWidget(title)
        layout.addWidget(self.progress_label)
        layout.addLayout(button_row)
        layout.addWidget(self.auto_save_checkbox)

        return card

    def _create_object_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("formCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel("3. Nesne ve Bounding Box")
        title.setObjectName("sectionTitle")

        class_label = QLabel("Sınıf")
        class_label.setObjectName("inputLabel")

        self.class_combo.setMinimumHeight(38)

        self.object_list.setMinimumHeight(100)
        self.object_list.setMaximumHeight(160)

        new_object_button = QPushButton(
            "Yeni Nesne Ekle"
        )
        new_object_button.setMinimumHeight(38)
        new_object_button.clicked.connect(
            self._add_new_object
        )

        bbox_button = QPushButton(
            "BBox Çizme Modu"
        )
        bbox_button.setObjectName("secondaryButton")
        bbox_button.setMinimumHeight(38)
        bbox_button.clicked.connect(
            self._activate_bbox_mode
        )

        select_button = QPushButton(
            "Düzenle / Kaydır Modu"
        )
        select_button.setMinimumHeight(38)
        select_button.clicked.connect(
            self._activate_select_mode
        )

        delete_object_button = QPushButton(
            "Aktif Nesneyi Sil"
        )
        delete_object_button.setObjectName(
            "dangerButton"
        )
        delete_object_button.setMinimumHeight(38)
        delete_object_button.clicked.connect(
            self._delete_active_object
        )

        mode_row = QHBoxLayout()
        mode_row.setSpacing(8)
        mode_row.addWidget(bbox_button)
        mode_row.addWidget(select_button)

        layout.addWidget(title)
        layout.addWidget(class_label)
        layout.addWidget(self.class_combo)
        layout.addWidget(self.object_list)
        layout.addWidget(new_object_button)
        layout.addLayout(mode_row)
        layout.addWidget(delete_object_button)

        return card

    def _create_keypoint_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("formCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(10)

        title = QLabel("4. Keypoint Etiketleme")
        title.setObjectName("sectionTitle")

        info = QLabel(
            "Yalnızca 12 gövde noktası kullanılır. Listeden noktayı seç, "
            "visibility değerini belirle ve görsel üzerinde konumuna tıkla. "
            "Yüz keypointleri kullanılmaz."
        )
        info.setObjectName("inputLabel")
        info.setWordWrap(True)

        self.keypoint_list.setMinimumHeight(190)

        visibility_label = QLabel("Visibility")
        visibility_label.setObjectName("inputLabel")

        self.visibility_combo.addItem(
            "0 - Görünmüyor / işaretlenmedi",
            0,
        )
        self.visibility_combo.addItem(
            "1 - Kapalı / zor görünüyor",
            1,
        )
        self.visibility_combo.addItem(
            "2 - Net görünüyor",
            2,
        )
        self.visibility_combo.setCurrentIndex(2)
        self.visibility_combo.setMinimumHeight(38)

        keypoint_mode_button = QPushButton(
            "Keypoint Yerleştirme Modu"
        )
        keypoint_mode_button.setObjectName(
            "primaryButton"
        )
        keypoint_mode_button.setMinimumHeight(40)
        keypoint_mode_button.clicked.connect(
            self._activate_keypoint_mode
        )

        previous_keypoint_button = QPushButton(
            "← Önceki Nokta"
        )
        next_keypoint_button = QPushButton(
            "Sonraki Nokta →"
        )
        previous_keypoint_button.setMinimumHeight(36)
        next_keypoint_button.setMinimumHeight(36)

        previous_keypoint_button.clicked.connect(
            self._select_previous_keypoint
        )
        next_keypoint_button.clicked.connect(
            self._select_next_keypoint
        )

        mark_invisible_button = QPushButton(
            "Noktayı Görünmez Yap"
        )
        mark_invisible_button.setMinimumHeight(38)
        mark_invisible_button.clicked.connect(
            self._mark_active_keypoint_invisible
        )

        nav_row = QHBoxLayout()
        nav_row.setSpacing(8)
        nav_row.addWidget(previous_keypoint_button)
        nav_row.addWidget(next_keypoint_button)

        layout.addWidget(title)
        layout.addWidget(info)
        layout.addWidget(self.keypoint_list)
        layout.addWidget(visibility_label)
        layout.addWidget(self.visibility_combo)
        layout.addWidget(keypoint_mode_button)
        layout.addLayout(nav_row)
        layout.addWidget(mark_invisible_button)

        return card

    def _create_save_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("formCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(9)

        title = QLabel("5. Kayıt")
        title.setObjectName("sectionTitle")

        self.save_button.setObjectName(
            "primaryButton"
        )
        self.save_button.setMinimumHeight(42)

        self.undo_button.setMinimumHeight(38)
        self.open_labels_button.setMinimumHeight(38)

        clear_button = QPushButton(
            "Bu Görselin Etiketlerini Temizle"
        )
        clear_button.setObjectName("dangerButton")
        clear_button.setMinimumHeight(38)
        clear_button.clicked.connect(
            self._clear_current_annotations
        )

        layout.addWidget(title)
        layout.addWidget(self.save_button)
        layout.addWidget(self.undo_button)
        layout.addWidget(self.open_labels_button)
        layout.addWidget(clear_button)

        return card

    @staticmethod
    def _create_path_field(
        *,
        title: str,
        line_edit: QLineEdit,
        button_text: str,
        callback,
    ) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QLabel(title)
        label.setObjectName("inputLabel")

        row = QHBoxLayout()
        row.setSpacing(8)

        line_edit.setMinimumHeight(38)
        line_edit.setClearButtonEnabled(True)
        line_edit.setReadOnly(False)

        button = QPushButton(button_text)
        button.setMinimumHeight(38)
        button.setMinimumWidth(100)
        button.clicked.connect(callback)

        row.addWidget(line_edit, 1)
        row.addWidget(button)

        layout.addWidget(label)
        layout.addLayout(row)

        return container

    def _configure_inputs(self) -> None:
        self.data_yaml_input.setPlaceholderText(
            "data.yaml dosyasını seç"
        )
        self.images_input.setPlaceholderText(
            "ham görsellerin bulunduğu klasör"
        )
        self.labels_input.setPlaceholderText(
            "label .txt dosyalarının kaydedileceği klasör"
        )

    def _connect_signals(self) -> None:
        self.previous_button.clicked.connect(
            self._show_previous_image
        )
        self.next_button.clicked.connect(
            self._show_next_image
        )
        self.save_button.clicked.connect(
            self._save_current_label
        )
        self.undo_button.clicked.connect(
            self._undo
        )
        self.open_labels_button.clicked.connect(
            self._open_labels_directory
        )
        self.fit_image_button.clicked.connect(
            self.canvas.fit_image
        )

        self.object_list.currentRowChanged.connect(
            self._on_object_selection_changed
        )
        self.class_combo.currentIndexChanged.connect(
            self._on_class_changed
        )
        self.keypoint_list.currentRowChanged.connect(
            self._on_keypoint_selection_changed
        )

        self.canvas.bbox_created.connect(
            self._on_bbox_created
        )
        self.canvas.keypoint_placed.connect(
            self._on_keypoint_placed
        )
        self.canvas.canvas_clicked.connect(
            self._update_coordinate_label
        )
        self.canvas.object_selected.connect(
            self._on_canvas_object_selected
        )
        self.canvas.keypoint_selected.connect(
            self._on_canvas_keypoint_selected
        )
        self.canvas.edit_started.connect(
            self._on_canvas_edit_started
        )
        self.canvas.bbox_edited.connect(
            self._on_canvas_bbox_edited
        )
        self.canvas.keypoint_edited.connect(
            self._on_canvas_keypoint_edited
        )
        self.canvas.edit_finished.connect(
            self._on_canvas_edit_finished
        )

    def _create_shortcuts(self) -> None:
        shortcuts = [
            ("Ctrl+S", self._save_current_label),
            ("Ctrl+Z", self._undo),
            ("A", self._show_previous_image),
            ("D", self._show_next_image),
            ("B", self._activate_bbox_mode),
            ("K", self._activate_keypoint_mode),
            ("F", self.canvas.fit_image),
            ("Escape", self._activate_select_mode),
        ]

        self._shortcuts: list[QShortcut] = []

        for sequence, callback in shortcuts:
            shortcut = QShortcut(
                QKeySequence(sequence),
                self,
            )
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)

    def _auto_label_dataset(self) -> None:
        """Hazır YOLO11 pose modeliyle tüm görselleri otomatik etiketler."""
        if (
            not self.image_paths
            or self.images_directory is None
            or self.labels_directory is None
        ):
            QMessageBox.information(
                self,
                "Dataset Yüklenmedi",
                "Önce data.yaml, Images ve Labels yollarını seçip "
                "'Dataseti Yükle' butonuna bas.",
            )
            return

        if self.keypoint_count != 12 or self.keypoint_dimensions != 3:
            QMessageBox.warning(
                self,
                "Uyumsuz Keypoint Yapısı",
                (
                    "Bu proje 12 gövde keypoint'i kullanıyor.\n"
                    "Hazır YOLO modeli arkada COCO-17 üretir; "
                    "yüz noktaları (0-4) otomatik olarak atılır.\n\n"
                    f"Mevcut YAML: {self.keypoint_count} keypoint, "
                    f"{self.keypoint_dimensions} boyut.\n\n"
                    "kpt_shape: [12, 3] olmalı."
                ),
            )
            return

        overwrite_answer = QMessageBox.question(
            self,
            "Otomatik Etiketleme",
            (
                f"{len(self.image_paths)} görsel YOLO11n-pose ile işlenecek.\n\n"
                "Daha önce oluşturulmuş .txt label dosyaları da "
                "yeniden yazılsın mı?\n\n"
                "Evet: Hepsini yeniden üret\n"
                "Hayır: Var olan label dosyalarını koru"
            ),
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
                | QMessageBox.StandardButton.Cancel
            ),
            QMessageBox.StandardButton.No,
        )

        if overwrite_answer == QMessageBox.StandardButton.Cancel:
            return

        overwrite_existing = (
            overwrite_answer == QMessageBox.StandardButton.Yes
        )

        try:
            from ultralytics import YOLO
        except ImportError:
            QMessageBox.critical(
                self,
                "Ultralytics Bulunamadı",
                (
                    "Otomatik etiketleme için ultralytics kurulu değil.\n\n"
                    "Terminal:\n"
                    "pip install ultralytics"
                ),
            )
            return

        try:
            import torch

            device = (
                "mps"
                if (
                    hasattr(torch.backends, "mps")
                    and torch.backends.mps.is_available()
                )
                else "cpu"
            )
        except Exception:
            device = "cpu"

        try:
            model = YOLO("yolo11n-pose.pt")
        except Exception as error:
            QMessageBox.critical(
                self,
                "Model Yüklenemedi",
                (
                    "yolo11n-pose.pt yüklenemedi.\n\n"
                    f"{error}"
                ),
            )
            return

        progress = QProgressDialog(
            "Otomatik etiketleme başlatılıyor...",
            "İptal",
            0,
            len(self.image_paths),
            self,
        )
        progress.setWindowTitle("YOLO Pose Otomatik Etiketleme")
        progress.setWindowModality(
            Qt.WindowModality.WindowModal
        )
        progress.setMinimumDuration(0)

        labeled_count = 0
        skipped_count = 0
        empty_count = 0
        error_count = 0

        for index, image_path in enumerate(
            self.image_paths,
            start=1,
        ):
            if progress.wasCanceled():
                break

            progress.setValue(index - 1)
            progress.setLabelText(
                f"{index}/{len(self.image_paths)} • {image_path.name}"
            )
            QApplication.processEvents()

            label_path = self._label_path_for_image(image_path)

            if label_path.exists() and not overwrite_existing:
                skipped_count += 1
                continue

            try:
                results = model.predict(
                    source=str(image_path),
                    conf=0.25,
                    imgsz=640,
                    device=device,
                    verbose=False,
                )

                if not results:
                    if label_path.exists():
                        label_path.unlink()
                    empty_count += 1
                    continue

                result = results[0]

                if (
                    result.boxes is None
                    or result.keypoints is None
                    or len(result.boxes) == 0
                ):
                    if label_path.exists():
                        label_path.unlink()
                    empty_count += 1
                    continue

                boxes_xywhn = (
                    result.boxes.xywhn
                    .detach()
                    .cpu()
                    .numpy()
                )
                # Hazır Ultralytics modeli COCO-17 üretir.
                # İlk 5 nokta yüzdür:
                # 0 nose, 1 left_eye, 2 right_eye, 3 left_ear, 4 right_ear
                # Biz yalnızca 5..16 arasındaki 12 gövde noktasını kullanıyoruz.
                all_keypoints_xyn = (
                    result.keypoints.xyn
                    .detach()
                    .cpu()
                    .numpy()
                )
                keypoints_xyn = all_keypoints_xyn[:, 5:17, :]

                keypoint_conf = None
                if result.keypoints.conf is not None:
                    all_keypoint_conf = (
                        result.keypoints.conf
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    keypoint_conf = all_keypoint_conf[:, 5:17]

                lines: list[str] = []

                detection_count = min(
                    len(boxes_xywhn),
                    len(keypoints_xyn),
                )

                for detection_index in range(detection_count):
                    x_center, y_center, width, height = (
                        boxes_xywhn[detection_index]
                    )

                    values = [
                        "0",  # Tek sınıf: person
                        f"{float(x_center):.6f}",
                        f"{float(y_center):.6f}",
                        f"{float(width):.6f}",
                        f"{float(height):.6f}",
                    ]

                    points = keypoints_xyn[detection_index]

                    if len(points) != self.keypoint_count:
                        continue

                    for keypoint_index, point in enumerate(points):
                        x_value = float(point[0])
                        y_value = float(point[1])

                        confidence = 1.0
                        if keypoint_conf is not None:
                            confidence = float(
                                keypoint_conf[
                                    detection_index,
                                    keypoint_index,
                                ]
                            )

                        # YOLO confidence -> YOLO Pose visibility
                        # 2: net, 1: düşük güven/kısmen kapalı, 0: yok
                        if confidence >= 0.50:
                            visibility = 2
                        elif confidence >= 0.20:
                            visibility = 1
                        else:
                            visibility = 0

                        if visibility == 0:
                            x_value = 0.0
                            y_value = 0.0

                        values.extend(
                            [
                                f"{self._clamp01(x_value):.6f}",
                                f"{self._clamp01(y_value):.6f}",
                                str(visibility),
                            ]
                        )

                    lines.append(" ".join(values))

                label_path.parent.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                if lines:
                    label_path.write_text(
                        "\n".join(lines) + "\n",
                        encoding="utf-8",
                    )
                    labeled_count += 1
                else:
                    if label_path.exists():
                        label_path.unlink()
                    empty_count += 1

            except Exception as error:
                error_count += 1
                print(
                    "[AUTO LABEL ERROR]",
                    image_path,
                    repr(error),
                )

        progress.setValue(len(self.image_paths))

        # Açık olan görselin yeni labelını tekrar yükle.
        if self.image_paths:
            self._load_current_image()

        QMessageBox.information(
            self,
            "Otomatik Etiketleme Tamamlandı",
            (
                f"Model: yolo11n-pose.pt\n"
                f"Çıktı: 12 gövde keypoint (COCO 5-16)\n"
                f"Cihaz: {device}\n\n"
                f"Etiketlenen: {labeled_count}\n"
                f"Atlanan (zaten vardı): {skipped_count}\n"
                f"Pose bulunamayan: {empty_count}\n"
                f"Hata: {error_count}\n\n"
                "Şimdi görselleri tek tek kontrol edip yalnızca "
                "hatalı bbox/keypointleri düzelt."
            ),
        )

    def _select_data_yaml(self) -> None:
        current = self.data_yaml_input.text().strip()
        start = (
            str(Path(current).expanduser().parent)
            if current
            else str(Path.home())
        )

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "data.yaml Seç",
            start,
            "YAML Dosyaları (*.yaml *.yml);;Tüm Dosyalar (*)",
        )

        if file_path:
            self.data_yaml_input.setText(file_path)

    def _select_images_directory(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Images Klasörünü Seç",
            self._start_directory(
                self.images_input.text()
            ),
        )

        if folder:
            self.images_input.setText(folder)

    def _select_labels_directory(self) -> None:
        folder = QFileDialog.getExistingDirectory(
            self,
            "Labels Klasörünü Seç",
            self._start_directory(
                self.labels_input.text()
            ),
        )

        if folder:
            self.labels_input.setText(folder)

    @staticmethod
    def _start_directory(text: str) -> str:
        path = Path(text.strip()).expanduser()

        if path.is_dir():
            return str(path)

        return str(Path.home())

    def _load_dataset(self) -> None:
        try:
            yaml_path = Path(
                self.data_yaml_input.text().strip()
            ).expanduser().resolve()
            images_directory = Path(
                self.images_input.text().strip()
            ).expanduser().resolve()
            labels_directory = Path(
                self.labels_input.text().strip()
            ).expanduser().resolve()

            self._validate_dataset_paths(
                yaml_path=yaml_path,
                images_directory=images_directory,
                labels_directory=labels_directory,
            )

            yaml_data = self._read_yaml(yaml_path)

            self.class_names = (
                self._extract_class_names(yaml_data)
            )
            (
                self.keypoint_count,
                self.keypoint_dimensions,
            ) = self._extract_kpt_shape(yaml_data)

            self.keypoint_names = (
                self._extract_keypoint_names(
                    yaml_data,
                    self.keypoint_count,
                )
            )
            self.skeleton = self._extract_skeleton(
                yaml_data,
                self.keypoint_count,
            )

            image_paths = sorted(
                path
                for path in images_directory.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in SUPPORTED_IMAGE_EXTENSIONS
                )
            )

            if not image_paths:
                raise ValueError(
                    "Images klasöründe desteklenen görsel bulunamadı."
                )

            labels_directory.mkdir(
                parents=True,
                exist_ok=True,
            )

        except (
            FileNotFoundError,
            PermissionError,
            ValueError,
            OSError,
            yaml.YAMLError,
        ) as error:
            QMessageBox.critical(
                self,
                "Dataset Yüklenemedi",
                str(error),
            )
            return

        self.data_yaml_path = yaml_path
        self.images_directory = images_directory
        self.labels_directory = labels_directory
        self.image_paths = image_paths

        self._populate_class_combo()
        self._populate_keypoint_list()

        self.current_image_index = 0
        self._load_current_image()

        QMessageBox.information(
            self,
            "Dataset Yüklendi",
            (
                f"{len(self.image_paths)} görsel bulundu.\n"
                f"Sınıf sayısı: {len(self.class_names)}\n"
                f"Keypoint sayısı: {self.keypoint_count}"
            ),
        )

    @staticmethod
    def _validate_dataset_paths(
        *,
        yaml_path: Path,
        images_directory: Path,
        labels_directory: Path,
    ) -> None:
        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"data.yaml bulunamadı:\n{yaml_path}"
            )

        if yaml_path.suffix.lower() not in {
            ".yaml",
            ".yml",
        }:
            raise ValueError(
                "Dataset dosyası .yaml veya .yml olmalıdır."
            )

        if not images_directory.is_dir():
            raise FileNotFoundError(
                f"Images klasörü bulunamadı:\n{images_directory}"
            )

        if (
            labels_directory.exists()
            and not labels_directory.is_dir()
        ):
            raise ValueError(
                "Labels yolu bir klasör olmalıdır."
            )

    @staticmethod
    def _read_yaml(path: Path) -> dict[str, Any]:
        with path.open(
            "r",
            encoding="utf-8",
        ) as yaml_file:
            data = yaml.safe_load(yaml_file)

        if not isinstance(data, dict):
            raise ValueError(
                "data.yaml geçerli bir YAML sözlüğü değil."
            )

        return data

    @staticmethod
    def _extract_class_names(
        yaml_data: dict[str, Any],
    ) -> dict[int, str]:
        names = yaml_data.get("names")

        if isinstance(names, list):
            return {
                index: str(name)
                for index, name in enumerate(names)
            }

        if isinstance(names, dict):
            return {
                int(class_id): str(class_name)
                for class_id, class_name
                in names.items()
            }

        raise ValueError(
            "data.yaml içinde geçerli 'names' alanı yok."
        )

    @staticmethod
    def _extract_kpt_shape(
        yaml_data: dict[str, Any],
    ) -> tuple[int, int]:
        shape = yaml_data.get("kpt_shape")

        if (
            not isinstance(shape, (list, tuple))
            or len(shape) != 2
        ):
            raise ValueError(
                "data.yaml içinde kpt_shape: [sayı, boyut] bulunmalı."
            )

        count = int(shape[0])
        dimensions = int(shape[1])

        if count <= 0:
            raise ValueError(
                "Keypoint sayısı sıfırdan büyük olmalıdır."
            )

        if dimensions not in {2, 3}:
            raise ValueError(
                "Keypoint boyutu 2 veya 3 olmalıdır."
            )

        return count, dimensions

    @staticmethod
    def _extract_keypoint_names(
        yaml_data: dict[str, Any],
        count: int,
    ) -> list[str]:
        raw_names = (
            yaml_data.get("keypoint_names")
            or yaml_data.get("kpt_names")
        )

        if (
            isinstance(raw_names, list)
            and len(raw_names) == count
        ):
            return [
                str(name)
                for name in raw_names
            ]

        return [
            f"Keypoint {index}"
            for index in range(count)
        ]

    @staticmethod
    def _extract_skeleton(
        yaml_data: dict[str, Any],
        keypoint_count: int,
    ) -> list[tuple[int, int]]:
        """
        data.yaml içindeki skeleton bağlantılarını okur.

        Bu projede 12 gövde keypoint'i kullanılır:
        0 left_shoulder
        1 right_shoulder
        2 left_elbow
        3 right_elbow
        4 left_wrist
        5 right_wrist
        6 left_hip
        7 right_hip
        8 left_knee
        9 right_knee
        10 left_ankle
        11 right_ankle
        """

        raw_skeleton = yaml_data.get("skeleton")

        if isinstance(raw_skeleton, list):
            pairs: list[tuple[int, int]] = []

            for raw_pair in raw_skeleton:
                if (
                    not isinstance(raw_pair, (list, tuple))
                    or len(raw_pair) != 2
                ):
                    continue

                first = int(raw_pair[0])
                second = int(raw_pair[1])

                if (
                    0 <= first < keypoint_count
                    and 0 <= second < keypoint_count
                ):
                    pairs.append((first, second))

            if pairs:
                return pairs

        if keypoint_count == 12:
            return [
                # Omuz hattı
                (0, 1),

                # Sol kol
                (0, 2),
                (2, 4),

                # Sağ kol
                (1, 3),
                (3, 5),

                # Gövde
                (0, 6),
                (1, 7),
                (6, 7),

                # Sol bacak
                (6, 8),
                (8, 10),

                # Sağ bacak
                (7, 9),
                (9, 11),
            ]

        return []

    def _populate_class_combo(self) -> None:
        self.class_combo.blockSignals(True)
        self.class_combo.clear()

        for class_id in sorted(self.class_names):
            self.class_combo.addItem(
                f"{class_id} - {self.class_names[class_id]}",
                class_id,
            )

        self.class_combo.blockSignals(False)

    def _populate_keypoint_list(self) -> None:
        self.keypoint_list.blockSignals(True)
        self.keypoint_list.clear()

        for index, name in enumerate(
            self.keypoint_names
        ):
            item = QListWidgetItem(
                f"{index:02d} • {name}"
            )
            self.keypoint_list.addItem(item)

        self.keypoint_list.blockSignals(False)

        if self.keypoint_count > 0:
            self.active_keypoint_index = 0
            self.keypoint_list.setCurrentRow(0)

    def _load_current_image(self) -> None:
        if not (
            0 <= self.current_image_index
            < len(self.image_paths)
        ):
            return

        image_path = self.image_paths[
            self.current_image_index
        ]

        if not self.canvas.load_image(image_path):
            QMessageBox.critical(
                self,
                "Görsel Açılamadı",
                f"Görsel okunamadı:\n{image_path}",
            )
            return

        self.current_image_path = image_path

        try:
            self.annotations = (
                self._read_label_for_current_image()
            )
        except (ValueError, OSError) as error:
            self.annotations = []
            QMessageBox.warning(
                self,
                "Label Okuma Uyarısı",
                str(error),
            )

        self.undo_stack.clear()
        self.is_dirty = False

        if self.annotations:
            self.active_object_index = 0
        else:
            self.active_object_index = -1

        self._refresh_all_views()
        self._update_progress()
        self._update_action_states()
        self.canvas.set_mode(
            AnnotationCanvas.MODE_SELECT
        )

    def _read_label_for_current_image(
        self,
    ) -> list[PoseObject]:
        label_path = self._current_label_path()

        if (
            label_path is None
            or not label_path.is_file()
        ):
            return []

        image_rect = self.canvas.image_rect
        image_width = image_rect.width()
        image_height = image_rect.height()

        expected_count = (
            5
            + self.keypoint_count
            * self.keypoint_dimensions
        )

        objects: list[PoseObject] = []

        for line_number, raw_line in enumerate(
            label_path.read_text(
                encoding="utf-8"
            ).splitlines(),
            start=1,
        ):
            line = raw_line.strip()

            if not line:
                continue

            raw_values = line.split()

            if len(raw_values) != expected_count:
                raise ValueError(
                    f"{label_path.name} satır {line_number}: "
                    f"{expected_count} değer bekleniyor, "
                    f"{len(raw_values)} bulundu."
                )

            values = [float(value) for value in raw_values]

            class_id = int(values[0])
            x_center, y_center, width, height = (
                values[1:5]
            )

            bbox = QRectF(
                (x_center - width / 2) * image_width,
                (y_center - height / 2) * image_height,
                width * image_width,
                height * image_height,
            )

            keypoints: list[PoseKeypoint] = []
            raw_keypoints = values[5:]

            for index in range(self.keypoint_count):
                start = (
                    index
                    * self.keypoint_dimensions
                )
                x_value = raw_keypoints[start]
                y_value = raw_keypoints[start + 1]

                visibility = (
                    int(raw_keypoints[start + 2])
                    if self.keypoint_dimensions == 3
                    else 2
                )

                keypoints.append(
                    PoseKeypoint(
                        x=x_value * image_width,
                        y=y_value * image_height,
                        visibility=visibility,
                    )
                )

            objects.append(
                PoseObject(
                    class_id=class_id,
                    bbox=bbox,
                    keypoints=keypoints,
                )
            )

        return objects

    def _current_label_path(self) -> Path | None:
        if (
            self.current_image_path is None
            or self.images_directory is None
            or self.labels_directory is None
        ):
            return None

        relative_path = (
            self.current_image_path.relative_to(
                self.images_directory
            )
        )

        return (
            self.labels_directory
            / relative_path.with_suffix(".txt")
        )

    def _show_previous_image(self) -> None:
        self._navigate_to(
            self.current_image_index - 1
        )

    def _show_next_image(self) -> None:
        self._navigate_to(
            self.current_image_index + 1
        )

    def _navigate_to(self, target_index: int) -> None:
        if not (
            0 <= target_index < len(self.image_paths)
        ):
            return

        if (
            self.is_dirty
            and self.auto_save_checkbox.isChecked()
        ):
            if not self._save_current_label(
                show_message=False
            ):
                return
        elif self.is_dirty:
            choice = QMessageBox.question(
                self,
                "Kaydedilmemiş Değişiklik",
                (
                    "Bu görselde kaydedilmemiş değişiklikler var.\n"
                    "Kaydedip devam edilsin mi?"
                ),
                (
                    QMessageBox.StandardButton.Save
                    | QMessageBox.StandardButton.Discard
                    | QMessageBox.StandardButton.Cancel
                ),
                QMessageBox.StandardButton.Save,
            )

            if choice == QMessageBox.StandardButton.Cancel:
                return

            if choice == QMessageBox.StandardButton.Save:
                if not self._save_current_label(
                    show_message=False
                ):
                    return

        self.current_image_index = target_index
        self._load_current_image()

    def _add_new_object(self) -> None:
        if not self.canvas.has_image:
            return

        self._push_undo()

        class_id = int(
            self.class_combo.currentData()
            if self.class_combo.currentData()
            is not None
            else 0
        )

        pose_object = PoseObject(
            class_id=class_id,
            bbox=None,
            keypoints=[
                PoseKeypoint()
                for _ in range(
                    self.keypoint_count
                )
            ],
        )

        self.annotations.append(pose_object)
        self.active_object_index = (
            len(self.annotations) - 1
        )
        self.active_keypoint_index = 0
        self.is_dirty = True

        self._refresh_all_views()
        self._activate_bbox_mode()

    def _delete_active_object(self) -> None:
        if not self._has_active_object():
            return

        self._push_undo()
        del self.annotations[
            self.active_object_index
        ]

        if self.annotations:
            self.active_object_index = min(
                self.active_object_index,
                len(self.annotations) - 1,
            )
        else:
            self.active_object_index = -1

        self.is_dirty = True
        self._refresh_all_views()

    def _clear_current_annotations(self) -> None:
        if not self.annotations:
            return

        answer = QMessageBox.question(
            self,
            "Etiketleri Temizle",
            "Bu görseldeki bütün nesneler silinsin mi?",
            (
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No
            ),
            QMessageBox.StandardButton.No,
        )

        if answer != QMessageBox.StandardButton.Yes:
            return

        self._push_undo()
        self.annotations = []
        self.active_object_index = -1
        self.is_dirty = True
        self._refresh_all_views()

    def _activate_bbox_mode(self) -> None:
        if not self._has_active_object():
            QMessageBox.information(
                self,
                "Nesne Seçilmedi",
                "Önce 'Yeni Nesne Ekle' butonuna bas.",
            )
            return

        self.canvas.set_mode(
            AnnotationCanvas.MODE_BBOX
        )

    def _activate_keypoint_mode(self) -> None:
        if not self._has_active_object():
            QMessageBox.information(
                self,
                "Nesne Seçilmedi",
                "Önce bir nesne ekle veya seç.",
            )
            return

        if (
            self.annotations[
                self.active_object_index
            ].bbox
            is None
        ):
            QMessageBox.information(
                self,
                "BBox Eksik",
                "Keypoint yerleştirmeden önce bounding box çiz.",
            )
            return

        self.canvas.set_active_keypoint_index(
            self.active_keypoint_index
        )
        self.canvas.set_mode(
            AnnotationCanvas.MODE_KEYPOINT
        )

    def _activate_select_mode(self) -> None:
        self.canvas.set_mode(
            AnnotationCanvas.MODE_SELECT
        )

    def _on_canvas_object_selected(
        self,
        object_index: int,
    ) -> None:
        if not (
            0 <= object_index < len(self.annotations)
        ):
            return

        self.active_object_index = object_index

        if self.object_list.currentRow() != object_index:
            self.object_list.setCurrentRow(object_index)
        else:
            self._on_object_selection_changed(object_index)

    def _on_canvas_keypoint_selected(
        self,
        object_index: int,
        keypoint_index: int,
    ) -> None:
        if not (
            0 <= object_index < len(self.annotations)
            and 0 <= keypoint_index < self.keypoint_count
        ):
            return

        self.active_object_index = object_index
        self.active_keypoint_index = keypoint_index

        if self.object_list.currentRow() != object_index:
            self.object_list.setCurrentRow(object_index)

        if self.keypoint_list.currentRow() != keypoint_index:
            self.keypoint_list.setCurrentRow(keypoint_index)
        else:
            self._on_keypoint_selection_changed(keypoint_index)

    def _on_canvas_edit_started(self) -> None:
        self._push_undo()

    def _on_canvas_bbox_edited(
        self,
        object_index: int,
        rect: QRectF,
    ) -> None:
        if not (
            0 <= object_index < len(self.annotations)
        ):
            return

        self.active_object_index = object_index
        self.annotations[object_index].bbox = (
            rect.normalized().intersected(
                self.canvas.image_rect
            )
        )
        self.is_dirty = True
        self._refresh_canvas()
        self._update_progress()

    def _on_canvas_keypoint_edited(
        self,
        object_index: int,
        keypoint_index: int,
        point: QPointF,
    ) -> None:
        if not (
            0 <= object_index < len(self.annotations)
            and 0 <= keypoint_index < self.keypoint_count
        ):
            return

        pose_object = self.annotations[object_index]

        if keypoint_index >= len(pose_object.keypoints):
            return

        keypoint = pose_object.keypoints[keypoint_index]
        keypoint.x = point.x()
        keypoint.y = point.y()

        self.active_object_index = object_index
        self.active_keypoint_index = keypoint_index
        self.is_dirty = True
        self._refresh_canvas()
        self._update_progress()

    def _on_canvas_edit_finished(self) -> None:
        self._refresh_all_views()

    def _on_bbox_created(self, rect: QRectF) -> None:
        if not self._has_active_object():
            return

        self._push_undo()
        self.annotations[
            self.active_object_index
        ].bbox = rect
        self.is_dirty = True

        self._refresh_all_views()
        self._activate_keypoint_mode()

    def _on_keypoint_placed(
        self,
        keypoint_index: int,
        point: QPointF,
    ) -> None:
        if not self._has_active_object():
            return

        pose_object = self.annotations[
            self.active_object_index
        ]

        if not (
            0 <= keypoint_index
            < len(pose_object.keypoints)
        ):
            return

        self._push_undo()

        visibility = (
            int(self.visibility_combo.currentData())
            if self.keypoint_dimensions == 3
            else 2
        )

        pose_object.keypoints[
            keypoint_index
        ] = PoseKeypoint(
            x=point.x(),
            y=point.y(),
            visibility=visibility,
        )

        self.is_dirty = True
        self._refresh_all_views()

        if (
            keypoint_index
            < self.keypoint_count - 1
        ):
            self.active_keypoint_index += 1
            self.keypoint_list.setCurrentRow(
                self.active_keypoint_index
            )
            self.canvas.set_active_keypoint_index(
                self.active_keypoint_index
            )
        else:
            self._activate_select_mode()

    def _mark_active_keypoint_invisible(
        self,
    ) -> None:
        if not self._has_active_object():
            return

        pose_object = self.annotations[
            self.active_object_index
        ]

        if not (
            0 <= self.active_keypoint_index
            < len(pose_object.keypoints)
        ):
            return

        self._push_undo()
        pose_object.keypoints[
            self.active_keypoint_index
        ] = PoseKeypoint(
            x=0.0,
            y=0.0,
            visibility=0,
        )
        self.is_dirty = True
        self._refresh_all_views()
        self._select_next_keypoint()

    def _select_previous_keypoint(self) -> None:
        if self.keypoint_count <= 0:
            return

        self.active_keypoint_index = max(
            0,
            self.active_keypoint_index - 1,
        )
        self.keypoint_list.setCurrentRow(
            self.active_keypoint_index
        )

    def _select_next_keypoint(self) -> None:
        if self.keypoint_count <= 0:
            return

        self.active_keypoint_index = min(
            self.keypoint_count - 1,
            self.active_keypoint_index + 1,
        )
        self.keypoint_list.setCurrentRow(
            self.active_keypoint_index
        )

    def _on_object_selection_changed(
        self,
        row: int,
    ) -> None:
        if not (
            0 <= row < len(self.annotations)
        ):
            return

        self.active_object_index = row

        class_id = self.annotations[row].class_id
        combo_index = self.class_combo.findData(
            class_id
        )

        if combo_index >= 0:
            self.class_combo.blockSignals(True)
            self.class_combo.setCurrentIndex(
                combo_index
            )
            self.class_combo.blockSignals(False)

        self._refresh_canvas()
        self._refresh_keypoint_list_status()
        self._update_action_states()

    def _on_class_changed(self) -> None:
        if not self._has_active_object():
            return

        class_id = self.class_combo.currentData()

        if class_id is None:
            return

        if (
            self.annotations[
                self.active_object_index
            ].class_id
            == int(class_id)
        ):
            return

        self._push_undo()
        self.annotations[
            self.active_object_index
        ].class_id = int(class_id)
        self.is_dirty = True
        self._refresh_object_list()
        self._refresh_canvas()

    def _on_keypoint_selection_changed(
        self,
        row: int,
    ) -> None:
        if not (
            0 <= row < self.keypoint_count
        ):
            return

        self.active_keypoint_index = row
        self.canvas.set_active_keypoint_index(row)
        self._refresh_canvas()

        if self._has_active_object():
            keypoint = self.annotations[
                self.active_object_index
            ].keypoints[row]

            visibility_index = (
                self.visibility_combo.findData(
                    keypoint.visibility
                )
            )

            if visibility_index >= 0:
                self.visibility_combo.setCurrentIndex(
                    visibility_index
                )

    def _save_current_label(
        self,
        *,
        show_message: bool = True,
    ) -> bool:
        if self.current_image_path is None:
            return False

        label_path = self._current_label_path()

        if label_path is None:
            return False

        incomplete_indices = [
            index + 1
            for index, pose_object
            in enumerate(self.annotations)
            if pose_object.bbox is None
        ]

        if incomplete_indices:
            QMessageBox.warning(
                self,
                "Eksik Bounding Box",
                (
                    "Şu nesnelerde bbox yok: "
                    + ", ".join(
                        map(str, incomplete_indices)
                    )
                ),
            )
            return False

        image_rect = self.canvas.image_rect
        image_width = image_rect.width()
        image_height = image_rect.height()

        if image_width <= 0 or image_height <= 0:
            return False

        lines: list[str] = []

        for pose_object in self.annotations:
            assert pose_object.bbox is not None

            bbox = (
                pose_object.bbox
                .normalized()
                .intersected(image_rect)
            )

            x_center = (
                bbox.center().x() / image_width
            )
            y_center = (
                bbox.center().y() / image_height
            )
            width = bbox.width() / image_width
            height = bbox.height() / image_height

            values = [
                str(pose_object.class_id),
                f"{self._clamp01(x_center):.6f}",
                f"{self._clamp01(y_center):.6f}",
                f"{self._clamp01(width):.6f}",
                f"{self._clamp01(height):.6f}",
            ]

            for keypoint in pose_object.keypoints:
                if keypoint.visibility == 0:
                    x_value = 0.0
                    y_value = 0.0
                else:
                    x_value = self._clamp01(
                        keypoint.x / image_width
                    )
                    y_value = self._clamp01(
                        keypoint.y / image_height
                    )

                values.extend(
                    [
                        f"{x_value:.6f}",
                        f"{y_value:.6f}",
                    ]
                )

                if self.keypoint_dimensions == 3:
                    values.append(
                        str(
                            int(
                                keypoint.visibility
                            )
                        )
                    )

            lines.append(" ".join(values))

        try:
            label_path.parent.mkdir(
                parents=True,
                exist_ok=True,
            )

            if lines:
                label_path.write_text(
                    "\n".join(lines) + "\n",
                    encoding="utf-8",
                )
            elif label_path.exists():
                label_path.unlink()

        except OSError as error:
            QMessageBox.critical(
                self,
                "Kayıt Hatası",
                str(error),
            )
            return False

        self.is_dirty = False
        self._update_progress()
        self._update_action_states()

        if show_message:
            QMessageBox.information(
                self,
                "Etiket Kaydedildi",
                f"Label dosyası:\n{label_path}",
            )

        return True

    @staticmethod
    def _clamp01(value: float) -> float:
        return min(max(value, 0.0), 1.0)

    def _open_labels_directory(self) -> None:
        if self.labels_directory is None:
            return

        self.labels_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        QDesktopServices.openUrl(
            QUrl.fromLocalFile(
                str(self.labels_directory)
            )
        )

    def _push_undo(self) -> None:
        self.undo_stack.append(
            copy.deepcopy(self.annotations)
        )

        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)

    def _undo(self) -> None:
        if not self.undo_stack:
            return

        self.annotations = self.undo_stack.pop()

        if self.annotations:
            self.active_object_index = min(
                max(self.active_object_index, 0),
                len(self.annotations) - 1,
            )
        else:
            self.active_object_index = -1

        self.is_dirty = True
        self._refresh_all_views()

    def _refresh_all_views(self) -> None:
        self._refresh_object_list()
        self._refresh_keypoint_list_status()
        self._refresh_canvas()
        self._update_action_states()

    def _refresh_object_list(self) -> None:
        selected_row = self.active_object_index

        self.object_list.blockSignals(True)
        self.object_list.clear()

        for index, pose_object in enumerate(
            self.annotations
        ):
            class_name = self.class_names.get(
                pose_object.class_id,
                f"class_{pose_object.class_id}",
            )
            bbox_state = (
                "bbox ✓"
                if pose_object.bbox is not None
                else "bbox eksik"
            )

            self.object_list.addItem(
                f"#{index + 1} • {class_name} • {bbox_state}"
            )

        self.object_list.blockSignals(False)

        if (
            0 <= selected_row
            < self.object_list.count()
        ):
            self.object_list.setCurrentRow(
                selected_row
            )

    def _refresh_keypoint_list_status(self) -> None:
        for index in range(
            self.keypoint_list.count()
        ):
            item = self.keypoint_list.item(index)
            name = self.keypoint_names[index]

            visibility = 0

            if self._has_active_object():
                visibility = (
                    self.annotations[
                        self.active_object_index
                    ].keypoints[index].visibility
                )

            symbol = {
                0: "○",
                1: "◐",
                2: "●",
            }.get(visibility, "?")

            item.setText(
                f"{index:02d} {symbol} {name}"
            )

    def _refresh_canvas(self) -> None:
        self.canvas.redraw(
            objects=self.annotations,
            active_object_index=(
                self.active_object_index
            ),
            active_keypoint_index=(
                self.active_keypoint_index
            ),
            class_names=self.class_names,
            skeleton=self.skeleton,
        )

    def _update_progress(self) -> None:
        if not self.image_paths:
            self.progress_label.setText(
                "Dataset yüklenmedi"
            )
            self.image_name_label.setText(
                "Görsel seçilmedi"
            )
            return

        labeled_count = sum(
            1
            for image_path in self.image_paths
            if self._label_path_for_image(
                image_path
            ).is_file()
        )

        current_number = (
            self.current_image_index + 1
        )

        dirty_text = (
            " • kaydedilmedi"
            if self.is_dirty
            else ""
        )

        self.progress_label.setText(
            (
                f"Görsel: {current_number}/{len(self.image_paths)}\n"
                f"Etiketli: {labeled_count}/{len(self.image_paths)}"
                f"{dirty_text}"
            )
        )

        if self.current_image_path is not None:
            self.image_name_label.setText(
                self.current_image_path.name
                + dirty_text
            )

    def _label_path_for_image(
        self,
        image_path: Path,
    ) -> Path:
        assert self.images_directory is not None
        assert self.labels_directory is not None

        relative_path = image_path.relative_to(
            self.images_directory
        )

        return (
            self.labels_directory
            / relative_path.with_suffix(".txt")
        )

    def _update_coordinate_label(
        self,
        point: QPointF,
    ) -> None:
        self.coordinate_label.setText(
            f"x: {point.x():.1f}, y: {point.y():.1f}"
        )

    def _has_active_object(self) -> bool:
        return (
            0 <= self.active_object_index
            < len(self.annotations)
        )

    def _update_action_states(self) -> None:
        has_dataset = bool(self.image_paths)
        has_active_object = (
            self._has_active_object()
        )

        self.previous_button.setEnabled(
            has_dataset
            and self.current_image_index > 0
        )
        self.next_button.setEnabled(
            has_dataset
            and self.current_image_index
            < len(self.image_paths) - 1
        )

        self.save_button.setEnabled(has_dataset)
        self.auto_label_button.setEnabled(has_dataset)
        self.fit_image_button.setEnabled(
            has_dataset and self.canvas.has_image
        )
        self.undo_button.setEnabled(
            bool(self.undo_stack)
        )
        self.open_labels_button.setEnabled(
            self.labels_directory is not None
        )

        self.class_combo.setEnabled(
            has_active_object
        )
        self.keypoint_list.setEnabled(
            has_active_object
        )
        self.visibility_combo.setEnabled(
            has_active_object
            and self.keypoint_dimensions == 3
        )

        self._update_progress()