from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.pose_inference_service import (
    InferenceProgress,
    InferenceRunResult,
    InferenceSettings,
    PoseInferenceService,
)
from services.pose_preview_service import PosePreviewService
from ui.base_page import BasePage


BUILD_ID = "2026-07-30-model-test-v9-macos-camera"


class InferenceWorker(QObject):
    """YOLO Pose inference işlemini arka planda çalıştırır."""

    log_message = Signal(str)
    progress_changed = Signal(object)
    inference_finished = Signal(object)
    inference_failed = Signal(str)
    worker_finished = Signal()

    def __init__(
        self,
        *,
        service: PoseInferenceService,
        mode: str,
        settings: InferenceSettings,
        source_path: str,
        camera_index: int,
    ) -> None:
        super().__init__()
        self.service = service
        self.mode = mode
        self.settings = settings
        self.source_path = source_path
        self.camera_index = camera_index

    @Slot()
    def run(self) -> None:
        try:
            result = self.service.run(
                mode=self.mode,
                settings=self.settings,
                source_path=self.source_path,
                camera_index=self.camera_index,
                log_callback=self.log_message.emit,
                progress_callback=self.progress_changed.emit,
            )
        except Exception as error:
            self.inference_failed.emit(str(error))
        else:
            self.inference_finished.emit(result)
        finally:
            self.worker_finished.emit()


class TestingPage(BasePage):
    """Model inference testi ve YOLO Pose label önizleme sayfası."""

    def __init__(self) -> None:
        super().__init__(
            title="Model Testi",
            description=(
                "Eğitilmiş YOLO Pose modeli fotoğraf, görsel klasörü, video ve "
                "canlı webcam üzerinde test edilir. Label Preview sekmesinde "
                "dataset etiketleri ayrıca görsel olarak kontrol edilebilir."
            ),
        )

        self.inference_service = PoseInferenceService()
        self.preview_service = PosePreviewService()

        self.inference_thread: QThread | None = None
        self.inference_worker: InferenceWorker | None = None
        self.last_inference_result: InferenceRunResult | None = None
        self.last_inference_output_directory: Path | None = None
        self._last_preview_pixmap: QPixmap | None = None
        self._inference_active = False
        self._stop_request_sent = False

        self._create_inference_controls()
        self._create_preview_controls()
        self._configure_inference_controls()
        self._configure_preview_controls()
        self._build_ui()
        self._connect_inference_signals()
        self._set_inference_running(False)

        self.append_inference_log(f"TestingPage build: {BUILD_ID}")

    # ==================================================================
    # Kontrol nesneleri
    # ==================================================================

    def _create_inference_controls(self) -> None:
        self.model_input = QLineEdit()
        self.select_model_button = QPushButton("Model Seç")

        self.source_type_combo = QComboBox()
        self.source_input = QLineEdit()
        self.select_source_button = QPushButton("Kaynak Seç")
        self.webcam_index_combo = QComboBox()
        self.camera_help_label = QLabel(
            "Kamera 0 çoğunlukla MacBook'un dahili kamerasıdır. "
            "Harici bir webcam kullanıyorsanız Kamera 1-3 seçeneklerini deneyin."
        )

        self.inference_output_input = QLineEdit("test_output")
        self.select_inference_output_button = QPushButton("Klasör Seç")

        self.confidence_spin = QDoubleSpinBox()
        self.iou_spin = QDoubleSpinBox()
        self.image_size_spin = QSpinBox()
        self.device_combo = QComboBox()
        self.keypoint_confidence_spin = QDoubleSpinBox()
        self.line_width_spin = QSpinBox()
        self.point_radius_spin = QSpinBox()

        self.show_boxes_checkbox = QCheckBox("Bounding box çiz")
        self.show_labels_checkbox = QCheckBox("Sınıf adını yaz")
        self.show_confidence_checkbox = QCheckBox("Confidence değerini yaz")
        self.show_keypoints_checkbox = QCheckBox("Keypoint noktalarını çiz")
        self.show_skeleton_checkbox = QCheckBox("İskelet bağlantılarını çiz")
        self.save_output_checkbox = QCheckBox("İşlenmiş sonucu kaydet")

        self.validate_inference_button = QPushButton("Ayarları Doğrula")
        self.start_inference_button = QPushButton("Testi Başlat")
        self.stop_inference_button = QPushButton("Testi Durdur")
        self.open_inference_output_button = QPushButton("Çıktı Klasörünü Aç")
        self.clear_inference_button = QPushButton("Formu / Logu Temizle")

        self.inference_progress_bar = QProgressBar()
        self.inference_status_label = QLabel("Henüz model testi başlatılmadı.")
        self.inference_metric_label = QLabel(
            "Tespit sayısı, inference süresi ve FPS burada gösterilir."
        )
        self.inference_preview_label = QLabel("Test önizlemesi burada gösterilir.")
        self.inference_log_output = QTextEdit()

    def _create_preview_controls(self) -> None:
        # Mevcut testing_page.py içindeki Pose Preview alanları korunmuştur.
        self.data_yaml_input = QLineEdit()
        self.images_input = QLineEdit()
        self.labels_input = QLineEdit()
        self.output_input = QLineEdit()

        self.max_images_spin = QSpinBox()
        self.log_output = QTextEdit()

        self.generate_preview_button = QPushButton("Pose Preview Oluştur")
        self.open_preview_button = QPushButton("Preview Klasörünü Aç")
        self.last_preview_directory: Path | None = None

    # ==================================================================
    # Başlangıç ayarları
    # ==================================================================

    def _configure_inference_controls(self) -> None:
        for line_edit in (
            self.model_input,
            self.source_input,
            self.inference_output_input,
        ):
            line_edit.setMinimumHeight(42)
            line_edit.setClearButtonEnabled(True)
            line_edit.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )

        self.model_input.setPlaceholderText(
            "Eğitim çıktısı best.pt/last.pt dosyasını seçin"
        )
        self.source_input.setPlaceholderText(
            "Fotoğraf, görsel klasörü veya video kaynağını seçin"
        )
        self.inference_output_input.setPlaceholderText(
            "İşlenmiş sonuçların kaydedileceği klasör"
        )

        self.source_type_combo.addItem("Tek Fotoğraf", "image")
        self.source_type_combo.addItem("Görsel Klasörü", "directory")
        self.source_type_combo.addItem("Video", "video")
        self.source_type_combo.addItem("Canlı Webcam", "webcam")

        self.device_combo.addItem("Otomatik", "auto")
        self.device_combo.addItem("Apple MPS", "mps")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("CUDA GPU 0", "0")

        for combo in (self.source_type_combo, self.device_combo):
            combo.setMinimumHeight(42)

        # Otomatik kamera taraması macOS'ta gereksiz beklemeye yol açabildiği için
        # kaldırıldı. Kullanıcı yalnızca 0-3 arasındaki indekslerden seçim yapar.
        self.webcam_index_combo.addItem("Kamera 0 — Mac dahili / varsayılan", 0)
        self.webcam_index_combo.addItem("Kamera 1 — Harici kamera", 1)
        self.webcam_index_combo.addItem("Kamera 2 — İkinci / sanal kamera", 2)
        self.webcam_index_combo.addItem("Kamera 3 — Diğer kamera", 3)
        self.webcam_index_combo.setCurrentIndex(0)
        self.webcam_index_combo.setMinimumHeight(42)
        self.webcam_index_combo.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.webcam_index_combo.setToolTip(
            "Çoğu MacBook'ta Kamera 0 kullanılır. Kamera otomatik taranmaz."
        )
        self.camera_help_label.setObjectName("fieldDescription")
        self.camera_help_label.setWordWrap(True)

        self.confidence_spin.setRange(0.0, 1.0)
        self.confidence_spin.setDecimals(2)
        self.confidence_spin.setSingleStep(0.05)
        self.confidence_spin.setValue(0.25)

        self.iou_spin.setRange(0.0, 1.0)
        self.iou_spin.setDecimals(2)
        self.iou_spin.setSingleStep(0.05)
        self.iou_spin.setValue(0.70)

        self.keypoint_confidence_spin.setRange(0.0, 1.0)
        self.keypoint_confidence_spin.setDecimals(2)
        self.keypoint_confidence_spin.setSingleStep(0.05)
        self.keypoint_confidence_spin.setValue(0.25)

        self.image_size_spin.setRange(64, 4096)
        self.image_size_spin.setSingleStep(32)
        self.image_size_spin.setValue(640)

        self.line_width_spin.setRange(1, 20)
        self.line_width_spin.setValue(2)

        self.point_radius_spin.setRange(1, 30)
        self.point_radius_spin.setValue(4)

        for spin_box in (
            self.confidence_spin,
            self.iou_spin,
            self.keypoint_confidence_spin,
            self.image_size_spin,
            self.line_width_spin,
            self.point_radius_spin,
        ):
            spin_box.setMinimumHeight(42)

        self.show_boxes_checkbox.setChecked(True)
        self.show_labels_checkbox.setChecked(True)
        self.show_confidence_checkbox.setChecked(True)
        self.show_keypoints_checkbox.setChecked(True)
        self.show_skeleton_checkbox.setChecked(True)
        self.save_output_checkbox.setChecked(True)

        for button in (
            self.select_model_button,
            self.select_source_button,
            self.select_inference_output_button,
            self.validate_inference_button,
            self.start_inference_button,
            self.stop_inference_button,
            self.open_inference_output_button,
            self.clear_inference_button,
        ):
            self._configure_action_button(button, 150)

        self.validate_inference_button.setObjectName("secondaryButton")
        self.start_inference_button.setObjectName("primaryButton")
        self.stop_inference_button.setObjectName("dangerButton")
        self.open_inference_output_button.setObjectName("secondaryButton")
        self.select_model_button.setObjectName("secondaryButton")
        self.select_source_button.setObjectName("secondaryButton")
        self.select_inference_output_button.setObjectName("secondaryButton")

        self.inference_progress_bar.setRange(0, 1000)
        self.inference_progress_bar.setValue(0)
        self.inference_progress_bar.setFormat("%p%")
        self.inference_progress_bar.setMinimumHeight(25)

        self.inference_status_label.setWordWrap(True)
        self.inference_metric_label.setWordWrap(True)

        self.inference_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.inference_preview_label.setMinimumHeight(430)
        self.inference_preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.inference_preview_label.setObjectName("settingFrame")

        self.inference_log_output.setReadOnly(True)
        self.inference_log_output.setMinimumHeight(240)
        self.inference_log_output.setPlaceholderText(
            "Model yükleme, kaynak işleme ve çıktı bilgileri burada gösterilir."
        )

        self._update_source_controls()

    def _configure_preview_controls(self) -> None:
        path_inputs = (
            self.data_yaml_input,
            self.images_input,
            self.labels_input,
            self.output_input,
        )

        for line_edit in path_inputs:
            line_edit.setReadOnly(False)
            line_edit.setClearButtonEnabled(True)
            line_edit.setMinimumHeight(42)
            line_edit.setMinimumWidth(320)
            line_edit.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )

        self.data_yaml_input.setPlaceholderText(
            "data.yaml seçin veya tam yolunu yazın"
        )
        self.images_input.setPlaceholderText(
            "images klasörünü seçin veya tam yolunu yazın"
        )
        self.labels_input.setPlaceholderText(
            "labels klasörünü seçin veya tam yolunu yazın"
        )
        self.output_input.setPlaceholderText(
            "çıktı klasörünü seçin veya tam yolunu yazın"
        )

        self.max_images_spin.setRange(1, 100000)
        self.max_images_spin.setValue(10)
        self.max_images_spin.setMinimumWidth(170)
        self.max_images_spin.setMinimumHeight(42)

        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(220)
        self.log_output.setMaximumHeight(340)
        self.log_output.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        self.log_output.setPlaceholderText(
            "Pose preview doğrulama ve üretim sonuçları burada gösterilir."
        )

        self._configure_action_button(self.generate_preview_button, 170)
        self._configure_action_button(self.open_preview_button, 170)
        self.generate_preview_button.setObjectName("primaryButton")
        self.open_preview_button.setObjectName("secondaryButton")
        self.open_preview_button.setEnabled(True)

    # ==================================================================
    # Ana UI
    # ==================================================================

    def _build_ui(self) -> None:
        page_layout = self.layout()
        if page_layout is None:
            page_layout = QVBoxLayout(self)
            page_layout.setContentsMargins(32, 28, 32, 28)
            page_layout.setSpacing(16)

        self._remove_base_placeholder(page_layout)

        tabs = QTabWidget()
        tabs.setDocumentMode(True)
        tabs.addTab(self._build_inference_tab(), "Eğitilmiş Model Testi")
        tabs.addTab(self._build_preview_tab(), "Pose Label Preview")

        insert_index = page_layout.count()
        if insert_index > 0:
            last_item = page_layout.itemAt(insert_index - 1)
            if last_item is not None and last_item.spacerItem() is not None:
                insert_index -= 1

        page_layout.insertWidget(insert_index, tabs, 1)

    @staticmethod
    def _remove_base_placeholder(page_layout: QVBoxLayout) -> None:
        placeholder_text = "Bu modül sonraki aşamalarda geliştirilecektir."
        for index in range(page_layout.count() - 1, -1, -1):
            item = page_layout.itemAt(index)
            candidate = item.widget() if item is not None else None
            if candidate is None:
                continue

            labels: list[QLabel] = []
            if isinstance(candidate, QLabel):
                labels.append(candidate)
            labels.extend(candidate.findChildren(QLabel))

            if not any(placeholder_text in label.text() for label in labels):
                continue

            page_layout.removeWidget(candidate)
            candidate.hide()
            candidate.setParent(None)
            candidate.deleteLater()

    def _build_inference_tab(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(8, 10, 12, 30)
        content_layout.setSpacing(18)

        content_layout.addWidget(self._create_inference_intro_card())
        content_layout.addWidget(self._create_inference_paths_card())
        content_layout.addWidget(self._create_inference_settings_card())
        content_layout.addWidget(self._create_inference_drawing_card())
        content_layout.addWidget(self._create_inference_action_card())
        content_layout.addWidget(self._create_inference_progress_card())
        content_layout.addWidget(self._create_inference_preview_card())
        content_layout.addWidget(self._create_inference_log_card())
        content_layout.addStretch()

        scroll_area.setWidget(content)

        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(scroll_area)
        return wrapper

    def _build_preview_tab(self) -> QWidget:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 10, 12, 30)
        layout.setSpacing(18)

        layout.addWidget(self._create_preview_intro_card())
        layout.addWidget(self._create_preview_dataset_card())
        layout.addWidget(self._create_preview_settings_card())
        layout.addWidget(self._create_preview_action_bar())
        layout.addWidget(self._create_preview_log_card())
        layout.addStretch()

        scroll_area.setWidget(content)

        wrapper = QWidget()
        wrapper_layout = QVBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 0, 0, 0)
        wrapper_layout.addWidget(scroll_area)
        return wrapper

    # ==================================================================
    # Inference kartları
    # ==================================================================

    def _create_inference_intro_card(self) -> QFrame:
        card, layout = self._create_card(
            "Eğitilmiş Model ile Gerçek Inference",
            (
                "runs/.../weights/best.pt veya başka bir YOLO Pose modeli seçin. "
                "Tek fotoğraf, klasör, video ve canlı webcam üzerinde tahmin "
                "çalıştırabilir; çizimli çıktıları kaydedebilirsiniz."
            ),
        )
        return card

    def _create_inference_paths_card(self) -> QFrame:
        card, layout = self._create_card(
            "Model, Kaynak ve Çıktı",
            "Model ve test kaynakları birbirinden bağımsız seçilir.",
        )

        layout.addWidget(
            self._create_path_field(
                title="Pose modeli (.pt)",
                description="Önerilen: eğitim sonucundaki weights/best.pt",
                line_edit=self.model_input,
                button=self.select_model_button,
            )
        )

        source_type_container = QWidget()
        source_type_layout = QVBoxLayout(source_type_container)
        source_type_layout.setContentsMargins(0, 0, 0, 0)
        source_type_layout.setSpacing(7)
        source_type_label = QLabel("Test kaynağı türü")
        source_type_label.setObjectName("inputLabel")
        source_type_layout.addWidget(source_type_label)
        source_type_layout.addWidget(self.source_type_combo)
        layout.addWidget(source_type_container)

        layout.addWidget(
            self._create_path_field(
                title="Test kaynağı",
                description=(
                    "Seçilen türe göre fotoğraf, görsel klasörü veya video dosyası"
                ),
                line_edit=self.source_input,
                button=self.select_source_button,
            )
        )

        webcam_container = QWidget()
        webcam_layout = QVBoxLayout(webcam_container)
        webcam_layout.setContentsMargins(0, 0, 0, 0)
        webcam_layout.setSpacing(7)
        webcam_label = QLabel("Webcam numarası")
        webcam_label.setObjectName("inputLabel")
        webcam_layout.addWidget(webcam_label)

        webcam_layout.addWidget(self.webcam_index_combo)
        webcam_layout.addWidget(self.camera_help_label)
        layout.addWidget(webcam_container)

        layout.addWidget(
            self._create_path_field(
                title="Inference çıktı klasörü",
                description=(
                    "İşlenmiş fotoğrafların ve videoların kaydedileceği ana klasör"
                ),
                line_edit=self.inference_output_input,
                button=self.select_inference_output_button,
            )
        )

        return card

    def _create_inference_settings_card(self) -> QFrame:
        card, layout = self._create_card(
            "Inference Ayarları",
            "Confidence, IoU, görüntü boyutu ve cihaz seçimini belirleyin.",
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(14)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        controls = (
            ("Confidence", self.confidence_spin),
            ("IoU", self.iou_spin),
            ("Görüntü boyutu", self.image_size_spin),
            ("Device", self.device_combo),
            ("Keypoint confidence", self.keypoint_confidence_spin),
            ("Çizgi kalınlığı", self.line_width_spin),
            ("Keypoint yarıçapı", self.point_radius_spin),
        )

        for index, (label_text, control) in enumerate(controls):
            grid.addWidget(
                self._create_labeled_control(label_text, control),
                index // 2,
                index % 2,
            )

        layout.addLayout(grid)
        return card

    def _create_inference_drawing_card(self) -> QFrame:
        card, layout = self._create_card(
            "Çizim ve Kayıt Ayarları",
            "Bounding box, keypoint, iskelet ve çıktı kaydı seçenekleri.",
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(12)

        checkboxes = (
            self.show_boxes_checkbox,
            self.show_labels_checkbox,
            self.show_confidence_checkbox,
            self.show_keypoints_checkbox,
            self.show_skeleton_checkbox,
            self.save_output_checkbox,
        )

        for index, checkbox in enumerate(checkboxes):
            grid.addWidget(checkbox, index // 2, index % 2)

        layout.addLayout(grid)
        return card

    def _create_inference_action_card(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("formCard")
        layout = QGridLayout(frame)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        layout.addWidget(self.validate_inference_button, 0, 0)
        layout.addWidget(self.start_inference_button, 0, 1)
        layout.addWidget(self.stop_inference_button, 0, 2)
        layout.addWidget(self.open_inference_output_button, 1, 0, 1, 2)
        layout.addWidget(self.clear_inference_button, 1, 2)
        return frame

    def _create_inference_progress_card(self) -> QFrame:
        card, layout = self._create_card(
            "Test İlerlemesi",
            "İşlenen kare/görsel, tespit sayısı, inference süresi ve FPS.",
        )
        layout.addWidget(self.inference_progress_bar)
        layout.addWidget(self.inference_status_label)
        layout.addWidget(self.inference_metric_label)
        return card

    def _create_inference_preview_card(self) -> QFrame:
        card, layout = self._create_card(
            "Canlı Test Önizlemesi",
            "Fotoğraf sonucu veya video/webcam kareleri burada gösterilir.",
        )
        layout.addWidget(self.inference_preview_label, 1)
        return card

    def _create_inference_log_card(self) -> QFrame:
        card, layout = self._create_card(
            "Model Testi Logları",
            "Model yükleme, kaynak işleme ve çıktı yolları.",
        )
        layout.addWidget(self.inference_log_output)
        return card

    # ==================================================================
    # Preview kartları — eski özellikler korunuyor
    # ==================================================================

    def _create_preview_intro_card(self) -> QFrame:
        card, layout = self._create_card(
            "Pose Label Önizleme",
            (
                "Augmentation veya etiketleme sonrasında oluşan YOLO Pose label "
                "dosyalarını görseller üzerinde çizer. Bounding box, class adı "
                "ve keypoint noktalarını görsel olarak kontrol eder."
            ),
        )
        return card

    def _create_preview_dataset_card(self) -> QFrame:
        card, layout = self._create_card(
            "Preview Dataset Girdileri",
            (
                "Her alan bağımsızdır. data.yaml, images, labels ve çıktı "
                "klasörünü farklı konumlardan seçebilirsiniz."
            ),
        )

        fields = (
            (
                "data.yaml",
                "Dataset sınıfları ve keypoint yapısını içeren YAML dosyası",
                self.data_yaml_input,
                "Dosya Seç",
                self.select_data_yaml,
            ),
            (
                "Images klasörü",
                "Önizleme üretilecek görsellerin bulunduğu klasör",
                self.images_input,
                "Klasör Seç",
                self.select_images_directory,
            ),
            (
                "Labels klasörü",
                "Görsellerle aynı ada sahip YOLO Pose .txt dosyaları",
                self.labels_input,
                "Klasör Seç",
                self.select_labels_directory,
            ),
            (
                "Çıktı klasörü",
                "pose_preview_... klasörünün oluşturulacağı konum",
                self.output_input,
                "Klasör Seç",
                self.select_output_directory,
            ),
        )

        for title, description, line_edit, button_text, callback in fields:
            button = QPushButton(button_text)
            self._configure_action_button(button, 120)
            button.setObjectName("secondaryButton")
            button.pressed.connect(callback)
            layout.addWidget(
                self._create_path_field(
                    title=title,
                    description=description,
                    line_edit=line_edit,
                    button=button,
                )
            )

        return card

    def _create_preview_settings_card(self) -> QFrame:
        card = QFrame()
        card.setObjectName("formCard")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(20)

        text_container = QWidget()
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(6)
        title_label = QLabel("Önizleme Ayarı")
        title_label.setObjectName("sectionTitle")
        info_label = QLabel(
            "İlk kaç image-label çifti için çizimli preview oluşturulacağını belirleyin."
        )
        info_label.setObjectName("inputLabel")
        info_label.setWordWrap(True)
        text_layout.addWidget(title_label)
        text_layout.addWidget(info_label)

        spin_container = self._create_labeled_control(
            "Maksimum görsel",
            self.max_images_spin,
        )
        layout.addWidget(text_container, 1)
        layout.addWidget(spin_container)
        return card

    def _create_preview_action_bar(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("formCard")
        layout = QGridLayout(frame)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setHorizontalSpacing(12)
        layout.setVerticalSpacing(12)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)

        clear_button = QPushButton("Formu Temizle")
        validate_button = QPushButton("Dataseti Doğrula")
        self._configure_action_button(clear_button, 150)
        self._configure_action_button(validate_button, 150)
        clear_button.setObjectName("dangerButton")
        validate_button.setObjectName("secondaryButton")

        clear_button.clicked.connect(self.clear_preview_form)
        validate_button.clicked.connect(self.validate_preview_with_message)
        self.open_preview_button.clicked.connect(self.open_preview_directory)
        self.generate_preview_button.clicked.connect(self.generate_preview)

        layout.addWidget(clear_button, 0, 0)
        layout.addWidget(validate_button, 0, 1)
        layout.addWidget(self.open_preview_button, 1, 0)
        layout.addWidget(self.generate_preview_button, 1, 1)
        return frame

    def _create_preview_log_card(self) -> QFrame:
        card, layout = self._create_card(
            "Preview İşlem Sonuçları",
            "Pose preview doğrulama ve üretim sonuçları.",
        )
        layout.addWidget(self.log_output)
        return card

    # ==================================================================
    # Ortak UI yardımcıları
    # ==================================================================

    @staticmethod
    def _create_card(title: str, description: str) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("formCard")
        card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 20, 24, 24)
        layout.setSpacing(13)

        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        description_label = QLabel(description)
        description_label.setObjectName("inputLabel")
        description_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(description_label)
        return card, layout

    @staticmethod
    def _create_path_field(
        *,
        title: str,
        description: str,
        line_edit: QLineEdit,
        button: QPushButton,
    ) -> QFrame:
        frame = QFrame()
        frame.setObjectName("settingFrame")
        frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(7)

        title_label = QLabel(title)
        title_label.setObjectName("inputLabel")
        description_label = QLabel(description)
        description_label.setObjectName("fieldDescription")
        description_label.setWordWrap(True)

        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 2, 0, 0)
        row_layout.setSpacing(10)
        row_layout.addWidget(line_edit, 1)
        row_layout.addWidget(button)

        layout.addWidget(title_label)
        layout.addWidget(description_label)
        layout.addWidget(row_widget)
        return frame

    @staticmethod
    def _create_labeled_control(label_text: str, control: QWidget) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        label = QLabel(label_text)
        label.setObjectName("inputLabel")
        layout.addWidget(label)
        layout.addWidget(control)
        return container

    @staticmethod
    def _configure_action_button(button: QPushButton, minimum_width: int) -> None:
        button.setMinimumWidth(minimum_width)
        button.setMinimumHeight(44)
        button.setCursor(Qt.CursorShape.PointingHandCursor)

    # ==================================================================
    # Inference sinyalleri ve seçimler
    # ==================================================================

    def _connect_inference_signals(self) -> None:
        # currentIndexChanged(int) bir argüman taşır. Lambda bu argümanı açıkça
        # karşılar; böylece webcam alanının aktifleşmesi PySide sürümüne bağlı kalmaz.
        self.source_type_combo.currentIndexChanged.connect(
            lambda _index: self._update_source_controls()
        )
        self.source_type_combo.activated.connect(
            lambda _index: self._update_source_controls()
        )
        self.select_model_button.clicked.connect(self.select_model_file)
        self.select_source_button.clicked.connect(self.select_inference_source)
        self.select_inference_output_button.clicked.connect(
            self.select_inference_output_directory
        )
        self.validate_inference_button.clicked.connect(
            self.validate_inference_with_message
        )
        self.start_inference_button.clicked.connect(self.start_inference)
        # Mouse basıldığı anda çalışır; ağır model/kamera işlemi sürerken
        # clicked sinyalinin mouse bırakılmasını beklemeyiz.
        self.stop_inference_button.pressed.connect(self.stop_inference)
        self.webcam_index_combo.currentIndexChanged.connect(
            self._on_webcam_index_changed
        )
        self.open_inference_output_button.clicked.connect(
            self.open_inference_output_directory
        )
        self.clear_inference_button.clicked.connect(self.clear_inference_form)

    @Slot()
    def _update_source_controls(self) -> None:
        mode = self.current_inference_mode()
        webcam_mode = mode == "webcam"
        editable = not self._is_inference_running()

        self.source_input.setEnabled(editable and not webcam_mode)
        self.select_source_button.setEnabled(editable and not webcam_mode)
        self.webcam_index_combo.setEnabled(editable and webcam_mode)

        if editable and webcam_mode:
            self.webcam_index_combo.setFocus(Qt.FocusReason.OtherFocusReason)

        if mode == "image":
            self.select_source_button.setText("Fotoğraf Seç")
            self.source_input.setPlaceholderText("Test edilecek fotoğrafı seçin")
        elif mode == "directory":
            self.select_source_button.setText("Klasör Seç")
            self.source_input.setPlaceholderText("Test edilecek görsel klasörünü seçin")
        elif mode == "video":
            self.select_source_button.setText("Video Seç")
            self.source_input.setPlaceholderText("Test edilecek videoyu seçin")
        else:
            self.select_source_button.setText("Webcam")
            self.source_input.setPlaceholderText(
                "Webcam modunda dosya kaynağı kullanılmaz"
            )

    @Slot(int)
    def _on_webcam_index_changed(self, _combo_index: int) -> None:
        raw_value = self.webcam_index_combo.currentData()
        if raw_value is None:
            return
        camera_index = self._selected_webcam_index()
        self.webcam_index_combo.setToolTip(
            f"Seçili webcam: Kamera {camera_index}."
        )
        if self.current_inference_mode() == "webcam":
            self.append_inference_log(
                f"Webcam seçimi değiştirildi: Kamera {camera_index}"
            )

    def _selected_webcam_index(self) -> int:
        raw_value = self.webcam_index_combo.currentData()
        if raw_value is None:
            return 0
        try:
            return int(raw_value)
        except (TypeError, ValueError):
            return 0

    def current_inference_mode(self) -> str:
        return str(self.source_type_combo.currentData())

    def select_model_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "YOLO Pose Modelini Seç",
            self._start_directory_for_path(self.model_input.text(), True),
            "YOLO Model Dosyaları (*.pt);;Tüm Dosyalar (*)",
        )
        if file_path:
            self.model_input.setText(file_path)
            self.append_inference_log(f"Model seçildi: {file_path}")

    def select_inference_source(self) -> None:
        mode = self.current_inference_mode()
        start_directory = self._start_directory_for_path(
            self.source_input.text(),
            use_parent=mode != "directory",
        )

        if mode == "image":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Test Fotoğrafını Seç",
                start_directory,
                "Görseller (*.jpg *.jpeg *.png *.bmp *.webp *.tif *.tiff);;Tüm Dosyalar (*)",
            )
        elif mode == "directory":
            path = QFileDialog.getExistingDirectory(
                self,
                "Test Görselleri Klasörünü Seç",
                start_directory,
                QFileDialog.Option.ShowDirsOnly,
            )
        elif mode == "video":
            path, _ = QFileDialog.getOpenFileName(
                self,
                "Test Videosunu Seç",
                start_directory,
                "Videolar (*.mp4 *.avi *.mov *.mkv *.m4v *.webm);;Tüm Dosyalar (*)",
            )
        else:
            return

        if path:
            self.source_input.setText(path)
            self.append_inference_log(f"Test kaynağı seçildi: {path}")

    def select_inference_output_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Inference Çıktı Klasörünü Seç",
            self._start_directory_for_path(
                self.inference_output_input.text(),
                False,
            ),
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self.inference_output_input.setText(directory)
            self.append_inference_log(f"Çıktı klasörü seçildi: {directory}")

    @staticmethod
    def _start_directory_for_path(raw_path: str, use_parent: bool) -> str:
        text = raw_path.strip()
        if not text:
            return str(Path.home())

        path = Path(text).expanduser()
        candidate = path.parent if use_parent else path
        if candidate.is_dir():
            return str(candidate.resolve())
        return str(Path.home())

    # ==================================================================
    # Inference ayarları, doğrulama ve thread
    # ==================================================================

    def _create_inference_settings(self) -> InferenceSettings:
        return InferenceSettings(
            model_path=self.model_input.text().strip(),
            output_directory=self.inference_output_input.text().strip(),
            confidence=self.confidence_spin.value(),
            iou=self.iou_spin.value(),
            image_size=self.image_size_spin.value(),
            device=str(self.device_combo.currentData()),
            keypoint_confidence=self.keypoint_confidence_spin.value(),
            line_width=self.line_width_spin.value(),
            point_radius=self.point_radius_spin.value(),
            show_boxes=self.show_boxes_checkbox.isChecked(),
            show_labels=self.show_labels_checkbox.isChecked(),
            show_confidence=self.show_confidence_checkbox.isChecked(),
            show_keypoints=self.show_keypoints_checkbox.isChecked(),
            show_skeleton=self.show_skeleton_checkbox.isChecked(),
            save_output=self.save_output_checkbox.isChecked(),
        )

    def validate_inference(self) -> tuple[InferenceSettings, str]:
        settings = self._create_inference_settings()
        if not settings.output_directory:
            raise ValueError("Inference çıktı klasörü belirtilmedi.")

        resolved_device = self.inference_service.validate_settings(settings)
        self.inference_service.validate_source(
            self.current_inference_mode(),
            self.source_input.text().strip(),
            self._selected_webcam_index(),
        )
        return settings, resolved_device

    def validate_inference_with_message(self) -> None:
        try:
            settings, resolved_device = self.validate_inference()
        except Exception as error:
            self.append_inference_log(f"DOĞRULAMA HATASI: {error}")
            QMessageBox.warning(self, "Model Testi Doğrulama Hatası", str(error))
            return

        mode_text = self.source_type_combo.currentText()
        camera_summary = (
            f"Webcam numarası: {self._selected_webcam_index()}\n"
            if self.current_inference_mode() == "webcam"
            else ""
        )
        summary = (
            f"Model: {settings.model_path}\n"
            f"Kaynak türü: {mode_text}\n"
            f"{camera_summary}"
            f"Cihaz: {resolved_device}\n"
            f"Confidence: {settings.confidence:.2f}\n"
            f"IoU: {settings.iou:.2f}\n"
            f"Görüntü boyutu: {settings.image_size}\n"
            f"Çıktı: {Path(settings.output_directory).expanduser()}"
        )
        self.append_inference_log("Ayarlar başarıyla doğrulandı.")
        self.append_inference_log(summary.replace("\n", " | "))
        QMessageBox.information(self, "Doğrulama Başarılı", summary)

    def start_inference(self) -> None:
        if self._is_inference_running():
            QMessageBox.warning(
                self,
                "Model Testi Devam Ediyor",
                "Aynı anda yalnızca bir model testi çalıştırılabilir.",
            )
            return

        try:
            settings, resolved_device = self.validate_inference()
        except Exception as error:
            self.append_inference_log(f"TEST BAŞLATILAMADI: {error}")
            QMessageBox.warning(self, "Test Başlatılamadı", str(error))
            return

        mode = self.current_inference_mode()
        source_path = self.source_input.text().strip()
        camera_index = self._selected_webcam_index()

        self.inference_service.reset_stop_request()
        self._stop_request_sent = False
        self._inference_active = True
        self.last_inference_result = None
        base_output = Path(settings.output_directory).expanduser().resolve()
        base_output.mkdir(parents=True, exist_ok=True)
        self.last_inference_output_directory = base_output
        self.inference_progress_bar.setValue(0)
        self.inference_progress_bar.setRange(0, 1000)
        self.inference_status_label.setText("Model testi hazırlanıyor...")
        self.inference_metric_label.setText("Henüz inference metriği alınmadı.")

        self.append_inference_log("")
        self.append_inference_log("=" * 68)
        self.append_inference_log(
            f"MODEL TESTİ BAŞLADI — {self.source_type_combo.currentText()}"
        )
        if mode == "webcam":
            self.append_inference_log(f"Seçilen webcam numarası: {camera_index}")
        self.append_inference_log(f"Cihaz: {resolved_device}")
        self.append_inference_log("=" * 68)

        self.inference_thread = QThread(self)
        self.inference_worker = InferenceWorker(
            service=self.inference_service,
            mode=mode,
            settings=settings,
            source_path=source_path,
            camera_index=camera_index,
        )
        self.inference_worker.moveToThread(self.inference_thread)

        self.inference_thread.started.connect(self.inference_worker.run)
        self.inference_worker.log_message.connect(self.append_inference_log)
        self.inference_worker.progress_changed.connect(
            self._handle_inference_progress
        )
        self.inference_worker.inference_finished.connect(
            self._handle_inference_finished
        )
        self.inference_worker.inference_failed.connect(
            self._handle_inference_failed
        )
        self.inference_worker.worker_finished.connect(
            self.inference_thread.quit
        )
        self.inference_worker.worker_finished.connect(
            self.inference_worker.deleteLater
        )
        self.inference_thread.finished.connect(
            self._handle_inference_thread_finished
        )
        self.inference_thread.finished.connect(
            self.inference_thread.deleteLater
        )

        self._set_inference_running(True)
        self.inference_thread.start()

    @Slot()
    def stop_inference(self) -> None:
        print(f"[TestingPage {BUILD_ID}] stop pressed", flush=True)

        if not self._is_inference_running():
            self.append_inference_log(
                "Testi Durdur çalıştı; ancak şu anda aktif model testi yok."
            )
            self.inference_status_label.setText("Aktif model testi bulunmuyor.")
            return

        # UI thread'i burada asla kamera/model kilidi beklemez. Servis yalnızca
        # stop event'ini set eder ve kaynak kapatmayı daemon thread'e bırakır.
        self.inference_service.request_stop()

        if not self._stop_request_sent:
            self._stop_request_sent = True
            self.append_inference_log(
                "TESTİ DURDUR: Stop event anında gönderildi; worker kapanıyor."
            )

        if self.inference_thread is not None:
            try:
                self.inference_thread.requestInterruption()
            except RuntimeError:
                pass

        # Butonu devre dışı bırakmıyoruz. Böylece kullanıcı tıklamanın işlendiğini
        # görür ve gerekirse stop isteğini tekrar gönderebilir.
        self.stop_inference_button.setText("Durduruluyor...")
        self.stop_inference_button.setEnabled(True)
        self.inference_status_label.setText(
            "Durdurma isteği alındı. Devam eden tek model çağrısı biter bitmez "
            "test kapanacak."
        )

    @Slot(object)
    def _handle_inference_progress(self, progress: InferenceProgress) -> None:
        # Stop tıklandıktan sonra kuyrukta kalmış eski webcam kareleri UI'ı
        # tekrar "çalışıyor" durumuna döndürmesin.
        if self._stop_request_sent:
            return

        if progress.percent is None:
            self.inference_progress_bar.setRange(0, 0)
            self.inference_progress_bar.setFormat("Canlı çalışma")
        else:
            if self.inference_progress_bar.maximum() == 0:
                self.inference_progress_bar.setRange(0, 1000)
            value = max(0, min(1000, int(progress.percent * 10)))
            self.inference_progress_bar.setValue(value)
            self.inference_progress_bar.setFormat(f"%{progress.percent:.1f}")

        self.inference_status_label.setText(progress.message)
        fps = 1000.0 / progress.inference_ms if progress.inference_ms > 0 else 0.0
        self.inference_metric_label.setText(
            f"Toplam tespit: {progress.detections} | "
            f"Inference: {progress.inference_ms:.2f} ms | "
            f"Yaklaşık FPS: {fps:.1f}"
        )

        if progress.preview_frame is not None:
            self._show_inference_frame(progress.preview_frame)

    @Slot(object)
    def _handle_inference_finished(self, result: InferenceRunResult) -> None:
        self.last_inference_result = result
        self.last_inference_output_directory = result.output_directory
        self.open_inference_output_button.setEnabled(True)

        if self.inference_progress_bar.maximum() == 0:
            self.inference_progress_bar.setRange(0, 1000)
        if not result.stopped:
            self.inference_progress_bar.setValue(1000)
            self.inference_progress_bar.setFormat("Test tamamlandı — %100")
        else:
            self.inference_progress_bar.setFormat("Test durduruldu")

        if result.last_preview_frame is not None:
            self._show_inference_frame(result.last_preview_frame)

        average_fps = (
            1000.0 / result.average_inference_ms
            if result.average_inference_ms > 0
            else 0.0
        )
        self.inference_status_label.setText(
            "Model testi kullanıcı isteğiyle durduruldu."
            if result.stopped
            else "Model testi başarıyla tamamlandı."
        )
        self.inference_metric_label.setText(
            f"İşlenen: {result.processed_count} | "
            f"Toplam tespit: {result.total_detections} | "
            f"Ortalama inference: {result.average_inference_ms:.2f} ms | "
            f"Yaklaşık FPS: {average_fps:.1f}"
        )

        self.append_inference_log("")
        self.append_inference_log("=" * 68)
        self.append_inference_log("MODEL TESTİ SONUCU")
        self.append_inference_log("=" * 68)
        self.append_inference_log(f"Mod: {result.mode}")
        self.append_inference_log(f"Cihaz: {result.resolved_device}")
        self.append_inference_log(f"İşlenen adet/kare: {result.processed_count}")
        self.append_inference_log(f"Toplam tespit: {result.total_detections}")
        self.append_inference_log(
            f"Ortalama inference: {result.average_inference_ms:.2f} ms"
        )
        self.append_inference_log(f"Toplam süre: {result.elapsed_seconds:.2f} sn")
        self.append_inference_log(f"Durduruldu: {result.stopped}")
        self.append_inference_log(f"Çıktı klasörü: {result.output_directory}")
        for output_path in result.output_paths[:10]:
            self.append_inference_log(f"- {output_path}")

        QMessageBox.information(
            self,
            "Model Testi Tamamlandı",
            (
                f"İşlenen adet/kare: {result.processed_count}\n"
                f"Toplam tespit: {result.total_detections}\n"
                f"Ortalama inference: {result.average_inference_ms:.2f} ms\n"
                f"Toplam süre: {result.elapsed_seconds:.2f} sn\n\n"
                f"Çıktı klasörü:\n{result.output_directory}"
            ),
        )

    @Slot(str)
    def _handle_inference_failed(self, message: str) -> None:
        if self.inference_progress_bar.maximum() == 0:
            self.inference_progress_bar.setRange(0, 1000)
        self.inference_progress_bar.setFormat("Test başarısız")
        self.inference_status_label.setText("Model testi hata nedeniyle tamamlanamadı.")
        self.inference_metric_label.setText(message)
        self.append_inference_log(f"MODEL TESTİ HATASI: {message}")
        QMessageBox.critical(self, "Model Testi Hatası", message)

    @Slot()
    def _handle_inference_thread_finished(self) -> None:
        self._inference_active = False
        self._stop_request_sent = False
        self.inference_thread = None
        self.inference_worker = None
        self._set_inference_running(False)

    def _is_inference_running(self) -> bool:
        if self._inference_active:
            return True
        if self.inference_thread is None:
            return False
        try:
            return self.inference_thread.isRunning()
        except RuntimeError:
            return False

    def _set_inference_running(self, running: bool) -> None:
        self._inference_active = running
        self.validate_inference_button.setEnabled(not running)
        self.start_inference_button.setEnabled(not running)
        # Aktif test boyunca buton daima tıklanabilir kalır. Stop gönderildikten
        # sonra tekrar basılması güvenlidir ve UI geri bildirimi sağlar.
        self.stop_inference_button.setEnabled(running)
        self.stop_inference_button.setText(
            "Durduruluyor..."
            if running and self._stop_request_sent
            else "Testi Durdur"
        )

        editable_controls = (
            self.model_input,
            self.select_model_button,
            self.source_type_combo,
            self.source_input,
            self.select_source_button,
            self.webcam_index_combo,
            self.inference_output_input,
            self.select_inference_output_button,
            self.confidence_spin,
            self.iou_spin,
            self.image_size_spin,
            self.device_combo,
            self.keypoint_confidence_spin,
            self.line_width_spin,
            self.point_radius_spin,
            self.show_boxes_checkbox,
            self.show_labels_checkbox,
            self.show_confidence_checkbox,
            self.show_keypoints_checkbox,
            self.show_skeleton_checkbox,
            self.save_output_checkbox,
            self.clear_inference_button,
        )
        for control in editable_controls:
            control.setEnabled(not running)

        # Çıktı klasörü işlem sırasında da açılabilir.
        self.open_inference_output_button.setEnabled(True)
        self.open_preview_button.setEnabled(True)

        if not running:
            self._update_source_controls()
        else:
            # Çalışma sırasında kaynak ve kamera numarası sabit kalır.
            self.webcam_index_combo.setEnabled(False)

    # ==================================================================
    # Görsel gösterme ve çıktı klasörü
    # ==================================================================

    def _show_inference_frame(self, frame: np.ndarray) -> None:
        if frame.ndim != 3 or frame.shape[2] != 3:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        bytes_per_line = channels * width
        image = QImage(
            rgb.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self._last_preview_pixmap = QPixmap.fromImage(image)
        self._refresh_inference_preview()

    def _refresh_inference_preview(self) -> None:
        if self._last_preview_pixmap is None:
            return
        available_size = self.inference_preview_label.size()
        scaled = self._last_preview_pixmap.scaled(
            available_size,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.inference_preview_label.setPixmap(scaled)

    def resizeEvent(self, event) -> None:  # type: ignore[override]
        super().resizeEvent(event)
        self._refresh_inference_preview()

    @Slot()
    def open_inference_output_directory(self) -> None:
        candidates: list[Path] = []

        if self.last_inference_output_directory is not None:
            candidates.append(self.last_inference_output_directory)

        raw_output = self.inference_output_input.text().strip() or "test_output"
        base_output = Path(raw_output).expanduser()
        if not base_output.is_absolute():
            base_output = Path.cwd() / base_output
        candidates.append(base_output)

        directory: Path | None = None
        for candidate in candidates:
            try:
                candidate = candidate.resolve()
                candidate.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            if candidate.is_dir():
                directory = candidate
                break

        if directory is None:
            QMessageBox.warning(
                self,
                "Çıktı Klasörü Açılamadı",
                "Inference çıktı klasörü oluşturulamadı veya bulunamadı.",
            )
            return

        self.last_inference_output_directory = directory
        self.append_inference_log(f"Çıktı klasörü açılıyor: {directory}")
        self._open_directory(directory, title="Inference Çıktı Klasörü")

    def _open_directory(self, directory: Path, *, title: str = "Klasör") -> bool:
        try:
            directory = directory.expanduser()
            if not directory.is_absolute():
                directory = Path.cwd() / directory
            directory = directory.resolve()
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            QMessageBox.warning(
                self,
                f"{title} Açılamadı",
                f"Klasör hazırlanamadı:\n{directory}\n\n{error}",
            )
            return False

        try:
            if sys.platform == "darwin":
                completed = subprocess.run(
                    ["/usr/bin/open", str(directory)],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                if completed.returncode == 0:
                    return True
                error_text = completed.stderr.strip() or "macOS open komutu başarısız oldu."
            elif sys.platform.startswith("win"):
                subprocess.Popen(["explorer", str(directory)])
                return True
            else:
                subprocess.Popen(["xdg-open", str(directory)])
                return True
        except OSError as error:
            error_text = str(error)

        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))
        if opened:
            return True

        QMessageBox.warning(
            self,
            f"{title} Açılamadı",
            f"Finder ile klasör açılamadı:\n{directory}\n\n{error_text}",
        )
        return False

    def clear_inference_form(self) -> None:
        if self._is_inference_running():
            QMessageBox.warning(
                self,
                "Test Devam Ediyor",
                "Model testi çalışırken form temizlenemez.",
            )
            return

        self.model_input.clear()
        self.source_input.clear()
        self.inference_output_input.setText("test_output")
        self.source_type_combo.setCurrentIndex(0)
        self.webcam_index_combo.setCurrentIndex(0)
        self.confidence_spin.setValue(0.25)
        self.iou_spin.setValue(0.70)
        self.image_size_spin.setValue(640)
        self.device_combo.setCurrentIndex(0)
        self.keypoint_confidence_spin.setValue(0.25)
        self.line_width_spin.setValue(2)
        self.point_radius_spin.setValue(4)
        self.show_boxes_checkbox.setChecked(True)
        self.show_labels_checkbox.setChecked(True)
        self.show_confidence_checkbox.setChecked(True)
        self.show_keypoints_checkbox.setChecked(True)
        self.show_skeleton_checkbox.setChecked(True)
        self.save_output_checkbox.setChecked(True)

        self.last_inference_result = None
        self.last_inference_output_directory = None
        self._last_preview_pixmap = None
        self.inference_preview_label.clear()
        self.inference_preview_label.setText("Test önizlemesi burada gösterilir.")
        self.inference_log_output.clear()
        self.inference_progress_bar.setRange(0, 1000)
        self.inference_progress_bar.setValue(0)
        self.inference_progress_bar.setFormat("%p%")
        self.inference_status_label.setText("Henüz model testi başlatılmadı.")
        self.inference_metric_label.setText(
            "Tespit sayısı, inference süresi ve FPS burada gösterilir."
        )
        self.open_inference_output_button.setEnabled(False)
        self.append_inference_log("Model testi formu temizlendi.")

    @Slot(str)
    def append_inference_log(self, message: str) -> None:
        self.inference_log_output.append(message)

    # ==================================================================
    # Pose Preview mevcut işlevleri
    # ==================================================================

    def select_data_yaml(self) -> None:
        current_text = self.data_yaml_input.text().strip()
        start_directory = self._start_directory_for_path(current_text, True)
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "data.yaml Dosyasını Seç",
            start_directory,
            "YAML Dosyaları (*.yaml *.yml);;Tüm Dosyalar (*)",
        )
        if file_path:
            self.data_yaml_input.setText(file_path)
            self.append_preview_log(f"data.yaml seçildi: {file_path}")

    def select_images_directory(self) -> None:
        folder_path = self._select_directory(
            "Images Klasörünü Seç",
            self.images_input.text().strip(),
        )
        if folder_path:
            self.images_input.setText(folder_path)
            self.append_preview_log(f"Images klasörü seçildi: {folder_path}")

    def select_labels_directory(self) -> None:
        folder_path = self._select_directory(
            "Labels Klasörünü Seç",
            self.labels_input.text().strip(),
        )
        if folder_path:
            self.labels_input.setText(folder_path)
            self.append_preview_log(f"Labels klasörü seçildi: {folder_path}")

    def select_output_directory(self) -> None:
        folder_path = self._select_directory(
            "Preview Çıktı Klasörünü Seç",
            self.output_input.text().strip(),
        )
        if folder_path:
            self.output_input.setText(folder_path)
            self.append_preview_log(f"Çıktı klasörü seçildi: {folder_path}")

    def _select_directory(self, title: str, current_text: str) -> str:
        current_path = Path(current_text).expanduser() if current_text else None
        start_directory = (
            str(current_path.resolve())
            if current_path is not None and current_path.is_dir()
            else str(Path.home())
        )
        return QFileDialog.getExistingDirectory(
            self,
            title,
            start_directory,
            QFileDialog.Option.ShowDirsOnly,
        )

    def validate_preview_form(self) -> tuple[bool, list[str]]:
        errors: list[str] = []

        data_yaml_path = self.data_yaml_input.text().strip()
        images_path = self.images_input.text().strip()
        labels_path = self.labels_input.text().strip()
        output_path = self.output_input.text().strip()

        if not data_yaml_path:
            errors.append("data.yaml dosyası seçilmedi.")
        elif not Path(data_yaml_path).expanduser().is_file():
            errors.append("Seçilen data.yaml dosyası bulunamadı.")
        elif Path(data_yaml_path).suffix.lower() not in {".yaml", ".yml"}:
            errors.append("Dataset yapılandırma dosyası YAML olmalıdır.")

        if not images_path:
            errors.append("Images klasörü seçilmedi.")
        elif not Path(images_path).expanduser().is_dir():
            errors.append("Seçilen images klasörü bulunamadı.")

        if not labels_path:
            errors.append("Labels klasörü seçilmedi.")
        elif not Path(labels_path).expanduser().is_dir():
            errors.append("Seçilen labels klasörü bulunamadı.")

        if not output_path:
            errors.append("Çıktı klasörü seçilmedi.")
        elif not Path(output_path).expanduser().is_dir():
            errors.append("Seçilen çıktı klasörü bulunamadı.")

        return not errors, errors

    def validate_preview_with_message(self) -> None:
        is_valid, errors = self.validate_preview_form()
        self.append_preview_log("")
        self.append_preview_log("=" * 60)
        self.append_preview_log("Pose preview dataset doğrulaması başlatıldı.")
        self.append_preview_log("=" * 60)

        if not is_valid:
            for error in errors:
                self.append_preview_log(f"HATA: {error}")
            QMessageBox.warning(self, "Doğrulama Hatası", "\n".join(errors))
            return

        self.append_preview_log("Dosya ve klasör yolları geçerli.")
        self.append_preview_log(
            f"Maksimum preview sayısı: {self.max_images_spin.value()}"
        )
        QMessageBox.information(
            self,
            "Doğrulama Başarılı",
            "Pose preview dataset yolları geçerli.",
        )

    def generate_preview(self) -> None:
        is_valid, errors = self.validate_preview_form()
        if not is_valid:
            self.append_preview_log("")
            self.append_preview_log("Pose preview oluşturulamadı.")
            for error in errors:
                self.append_preview_log(f"HATA: {error}")
            QMessageBox.warning(self, "Doğrulama Hatası", "\n".join(errors))
            return

        self.generate_preview_button.setEnabled(False)
        self.generate_preview_button.setText("Preview Oluşturuluyor...")
        self.open_preview_button.setEnabled(True)

        self.append_preview_log("")
        self.append_preview_log("=" * 60)
        self.append_preview_log("Pose label önizleme işlemi başlatıldı.")
        self.append_preview_log("=" * 60)
        self.append_preview_log(f"Maksimum görsel: {self.max_images_spin.value()}")

        try:
            result = self.preview_service.generate_preview_dataset(
                data_yaml_path=self.data_yaml_input.text().strip(),
                images_directory=self.images_input.text().strip(),
                labels_directory=self.labels_input.text().strip(),
                output_directory=self.output_input.text().strip(),
                max_images=self.max_images_spin.value(),
            )
        except Exception as error:
            self.append_preview_log(f"HATA: {error}")
            QMessageBox.critical(self, "Pose Preview Hatası", str(error))
            return
        finally:
            self.generate_preview_button.setEnabled(True)
            self.generate_preview_button.setText("Pose Preview Oluştur")

        self.last_preview_directory = result.output_directory
        self.open_preview_button.setEnabled(True)

        self.append_preview_log("")
        self.append_preview_log("=" * 60)
        self.append_preview_log("Pose preview başarıyla tamamlandı.")
        self.append_preview_log("=" * 60)
        self.append_preview_log(f"Üretilen preview: {result.preview_count}")
        self.append_preview_log(f"Atlanan dosya: {result.skipped_count}")
        self.append_preview_log(f"Çıktı klasörü: {result.output_directory}")

        if result.preview_image_paths:
            self.append_preview_log("")
            self.append_preview_log("İlk preview dosyaları:")
            for preview_path in result.preview_image_paths[:5]:
                self.append_preview_log(f"- {preview_path.name}")

        QMessageBox.information(
            self,
            "Pose Preview Tamamlandı",
            (
                "Pose label önizleme işlemi tamamlandı.\n\n"
                f"Üretilen preview: {result.preview_count}\n"
                f"Atlanan: {result.skipped_count}\n\n"
                f"Çıktı klasörü:\n{result.output_directory}"
            ),
        )

    @Slot()
    def open_preview_directory(self) -> None:
        candidates: list[Path] = []

        if self.last_preview_directory is not None:
            candidates.append(self.last_preview_directory)

        raw_output = self.output_input.text().strip()
        if raw_output:
            base_output = Path(raw_output).expanduser()
            if not base_output.is_absolute():
                base_output = Path.cwd() / base_output
            if base_output.is_dir():
                child_directories = [
                    path for path in base_output.iterdir() if path.is_dir()
                ]
                if child_directories:
                    newest = max(
                        child_directories,
                        key=lambda path: path.stat().st_mtime,
                    )
                    candidates.append(newest)
            candidates.append(base_output)

        if not candidates:
            QMessageBox.warning(
                self,
                "Preview Klasörü Belirtilmedi",
                "Önce preview çıktı klasörünü seçin veya preview oluşturun.",
            )
            return

        directory: Path | None = None
        for candidate in candidates:
            try:
                candidate = candidate.expanduser()
                if not candidate.is_absolute():
                    candidate = Path.cwd() / candidate
                candidate = candidate.resolve()
                candidate.mkdir(parents=True, exist_ok=True)
            except OSError:
                continue
            if candidate.is_dir():
                directory = candidate
                break

        if directory is None:
            QMessageBox.warning(
                self,
                "Preview Klasörü Açılamadı",
                "Preview çıktı klasörü bulunamadı veya oluşturulamadı.",
            )
            return

        self.last_preview_directory = directory
        self.append_preview_log(f"Preview klasörü açılıyor: {directory}")
        self._open_directory(directory, title="Preview Klasörü")

    def clear_preview_form(self) -> None:
        self.data_yaml_input.clear()
        self.images_input.clear()
        self.labels_input.clear()
        self.output_input.clear()
        self.max_images_spin.setValue(10)
        self.last_preview_directory = None
        self.open_preview_button.setEnabled(True)
        self.log_output.clear()
        self.append_preview_log("Pose preview formu temizlendi.")

    @Slot(str)
    def append_preview_log(self, message: str) -> None:
        self.log_output.append(message)

    # Eski dış çağrılarla uyumluluk için mevcut isim korunuyor.
    def append_log(self, message: str) -> None:
        self.append_preview_log(message)

    # ==================================================================
    # Kapanış
    # ==================================================================

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._is_inference_running():
            self.inference_service.request_stop()
            QMessageBox.information(
                self,
                "Model Testi Durduruluyor",
                "Aktif video/webcam testi için durdurma isteği gönderildi. "
                "İşlem kapandıktan sonra uygulamayı yeniden kapatın.",
            )
            event.ignore()
            return
        super().closeEvent(event)
