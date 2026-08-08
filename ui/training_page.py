from __future__ import annotations

from pathlib import Path
import subprocess
from typing import Callable

from PySide6.QtCore import QObject, QThread, Qt, QUrl, Signal, Slot
from PySide6.QtGui import QCloseEvent, QDesktopServices
from PySide6.QtWidgets import (
    QApplication,
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
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.training_service import (
    PoseTrainingService,
    TrainingProgress,
    TrainingResult,
    TrainingSettings,
)
from ui.base_page import BasePage


BUILD_ID = "2026-07-31-pose-recovery-v2-state-consistency"


class TrainingWorker(QObject):
    """YOLO Pose eğitimini arka planda çalıştıran worker."""

    log_message = Signal(str)
    progress_changed = Signal(object)
    training_finished = Signal(object)
    training_failed = Signal(str)
    worker_finished = Signal()

    def __init__(
        self,
        *,
        service: PoseTrainingService,
        settings: TrainingSettings,
    ) -> None:
        super().__init__()
        self.service = service
        self.settings = settings

    @Slot()
    def run(self) -> None:
        try:
            result = self.service.train(
                self.settings,
                log_callback=self.log_message.emit,
                progress_callback=self.progress_changed.emit,
            )
        except Exception as error:
            self.training_failed.emit(str(error))
        else:
            self.training_finished.emit(result)
        finally:
            self.worker_finished.emit()


class TrainingPage(BasePage):
    """YOLO Pose model eğitimi ekranı."""

    def __init__(self) -> None:
        super().__init__(
            title="YOLO Pose Model Eğitimi",
            description=(
                "Dataset, başlangıç modeli ve eğitim ayarları seçilerek "
                "YOLO Pose modeli eğitilir. Eğitim arka planda çalıştığı "
                "için arayüz işlem sırasında kullanılabilir durumda kalır."
            ),
        )

        self.training_service = PoseTrainingService()
        self.training_thread: QThread | None = None
        self.training_worker: TrainingWorker | None = None
        self.last_training_result: TrainingResult | None = None
        self.stop_request_sent = False

        self.data_yaml_input = QLineEdit()
        self.model_input = QLineEdit("yolo11n-pose.pt")
        self.output_directory_input = QLineEdit("runs")
        self.run_name_input = QLineEdit("pose_training")

        self.resume_checkbox = QCheckBox(
            "Kesintiye uğrayan eğitime checkpoint'ten devam et"
        )
        self.resume_checkpoint_input = QLineEdit()
        self.select_resume_button = QPushButton("last.pt Seç")
        self.find_latest_resume_button = QPushButton("En Son Checkpoint'i Bul")
        self.clear_resume_button = QPushButton("Yeni Eğitim Moduna Geç")

        self.model_preset_combo = QComboBox()
        self.epochs_spin = QSpinBox()
        self.image_size_spin = QSpinBox()
        self.batch_size_spin = QSpinBox()
        self.device_combo = QComboBox()
        self.workers_spin = QSpinBox()
        self.patience_spin = QSpinBox()
        self.optimizer_combo = QComboBox()
        self.seed_spin = QSpinBox()
        self.save_period_spin = QSpinBox()

        self.box_loss_spin = QDoubleSpinBox()
        self.cls_loss_spin = QDoubleSpinBox()
        self.dfl_loss_spin = QDoubleSpinBox()
        self.pose_loss_spin = QDoubleSpinBox()
        self.kobj_loss_spin = QDoubleSpinBox()
        self.rle_loss_spin = QDoubleSpinBox()
        self.nominal_batch_spin = QSpinBox()

        self.lr0_spin = QDoubleSpinBox()
        self.lrf_spin = QDoubleSpinBox()
        self.momentum_spin = QDoubleSpinBox()
        self.weight_decay_spin = QDoubleSpinBox()
        self.warmup_epochs_spin = QDoubleSpinBox()
        self.warmup_momentum_spin = QDoubleSpinBox()
        self.warmup_bias_lr_spin = QDoubleSpinBox()
        self.close_mosaic_spin = QSpinBox()
        self.cos_lr_checkbox = QCheckBox("Cosine learning-rate")
        self.automatic_recovery_checkbox = QCheckBox(
            "Elektrik kesintisi için otomatik recovery kaydı"
        )

        self.pretrained_checkbox = QCheckBox("Pretrained ağırlıkları kullan")
        self.deterministic_checkbox = QCheckBox("Deterministik eğitim")
        self.cache_checkbox = QCheckBox("Dataset cache")
        self.amp_checkbox = QCheckBox("AMP")
        self.plots_checkbox = QCheckBox("Grafikleri üret")
        self.exist_ok_checkbox = QCheckBox("Aynı çalışma klasörünü kullan")

        self.validation_summary = QTextEdit()
        self.training_log = QTextEdit()
        self.progress_bar = QProgressBar()
        self.epoch_status_label = QLabel("Henüz eğitim başlatılmadı.")
        self.metric_status_label = QLabel("Metrikler eğitim sırasında burada gösterilir.")

        self.validate_button = QPushButton("Ayarları Doğrula")
        self.start_button = QPushButton("Eğitimi Başlat")
        self.stop_button = QPushButton("Epoch Sonunda Durdur")
        self.open_run_button = QPushButton("Eğitim Klasörünü Aç")
        self.open_weights_button = QPushButton("Weights Klasörünü Aç")
        self.clear_log_button = QPushButton("Logu Temizle")

        self._configure_controls()
        self._build_ui()
        self._connect_signals()
        self._set_training_state(False)
        self._append_log(f"TrainingPage build: {BUILD_ID}")
        self._detect_recovery_state(show_message=False)

    # ------------------------------------------------------------------
    # UI kurulumu
    # ------------------------------------------------------------------

    def _configure_controls(self) -> None:
        path_inputs = (
            self.data_yaml_input,
            self.model_input,
            self.output_directory_input,
            self.run_name_input,
            self.resume_checkpoint_input,
        )

        for line_edit in path_inputs:
            line_edit.setMinimumHeight(40)
            line_edit.setClearButtonEnabled(True)

        self.data_yaml_input.setPlaceholderText(
            "Split edilmiş datasetin data.yaml dosyasını seçin"
        )
        self.model_input.setPlaceholderText(
            "Örnek: yolo11n-pose.pt veya yerel bir .pt dosyası"
        )
        self.output_directory_input.setPlaceholderText(
            "Eğitim çıktılarının kaydedileceği klasör"
        )
        self.run_name_input.setPlaceholderText("Eğitim çalışma adı")
        self.resume_checkpoint_input.setPlaceholderText(
            "Örnek: runs/pose_training/weights/last.pt"
        )

        self.model_preset_combo.addItem("YOLO11 Nano Pose", "yolo11n-pose.pt")
        self.model_preset_combo.addItem("YOLO11 Small Pose", "yolo11s-pose.pt")
        self.model_preset_combo.addItem("YOLO11 Medium Pose", "yolo11m-pose.pt")
        self.model_preset_combo.addItem("YOLO11 Large Pose", "yolo11l-pose.pt")
        self.model_preset_combo.addItem("YOLO11 XLarge Pose", "yolo11x-pose.pt")
        self.model_preset_combo.addItem("Yerel / Özel Model", "__custom__")
        self.model_preset_combo.setMinimumHeight(40)

        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(100)

        self.image_size_spin.setRange(64, 4096)
        self.image_size_spin.setSingleStep(32)
        self.image_size_spin.setValue(640)

        self.batch_size_spin.setRange(-1, 1024)
        self.batch_size_spin.setValue(2)
        self.batch_size_spin.setSpecialValueText("Otomatik (-1)")

        self.device_combo.addItem("Otomatik", "auto")
        self.device_combo.addItem("Apple MPS", "mps")
        self.device_combo.addItem("CPU", "cpu")
        self.device_combo.addItem("CUDA GPU 0", "0")
        self.device_combo.setMinimumHeight(40)

        self.workers_spin.setRange(0, 64)
        self.workers_spin.setValue(0)

        self.patience_spin.setRange(0, 10000)
        self.patience_spin.setValue(50)

        for optimizer in (
            "auto",
            "SGD",
            "Adam",
            "AdamW",
            "NAdam",
            "RAdam",
            "RMSProp",
        ):
            self.optimizer_combo.addItem(optimizer, optimizer)
        self.optimizer_combo.setMinimumHeight(40)

        self.seed_spin.setRange(0, 2_147_483_647)
        self.seed_spin.setValue(42)

        self.save_period_spin.setRange(-1, 10000)
        self.save_period_spin.setValue(1)
        self.save_period_spin.setSpecialValueText("Yalnızca last.pt (-1)")

        self._configure_double_spin(
            self.box_loss_spin, 0.0, 100.0, 7.5, 0.1, 3
        )
        self._configure_double_spin(
            self.cls_loss_spin, 0.0, 100.0, 0.5, 0.1, 3
        )
        self._configure_double_spin(
            self.dfl_loss_spin, 0.0, 100.0, 1.5, 0.1, 3
        )
        self._configure_double_spin(
            self.pose_loss_spin, 0.0, 100.0, 12.0, 0.5, 3
        )
        self._configure_double_spin(
            self.kobj_loss_spin, 0.0, 100.0, 1.0, 0.1, 3
        )
        self._configure_double_spin(
            self.rle_loss_spin, 0.0, 100.0, 1.0, 0.1, 3
        )
        self.nominal_batch_spin.setRange(1, 4096)
        self.nominal_batch_spin.setValue(64)

        self._configure_double_spin(
            self.lr0_spin, 0.000001, 10.0, 0.01, 0.001, 6
        )
        self._configure_double_spin(
            self.lrf_spin, 0.000001, 1.0, 0.01, 0.001, 6
        )
        self._configure_double_spin(
            self.momentum_spin, 0.0, 1.0, 0.937, 0.001, 4
        )
        self._configure_double_spin(
            self.weight_decay_spin, 0.0, 1.0, 0.0005, 0.0001, 6
        )
        self._configure_double_spin(
            self.warmup_epochs_spin, 0.0, 1000.0, 3.0, 0.5, 2
        )
        self._configure_double_spin(
            self.warmup_momentum_spin, 0.0, 1.0, 0.8, 0.01, 3
        )
        self._configure_double_spin(
            self.warmup_bias_lr_spin, 0.0, 10.0, 0.1, 0.01, 4
        )
        self.close_mosaic_spin.setRange(0, 10000)
        self.close_mosaic_spin.setValue(10)

        for spin_box in (
            self.epochs_spin,
            self.image_size_spin,
            self.batch_size_spin,
            self.workers_spin,
            self.patience_spin,
            self.seed_spin,
            self.save_period_spin,
            self.nominal_batch_spin,
            self.close_mosaic_spin,
        ):
            spin_box.setMinimumHeight(40)

        self.automatic_recovery_checkbox.setChecked(True)
        self.cos_lr_checkbox.setChecked(False)
        self.pretrained_checkbox.setChecked(True)
        self.deterministic_checkbox.setChecked(True)
        self.cache_checkbox.setChecked(False)
        self.amp_checkbox.setChecked(False)
        self.plots_checkbox.setChecked(True)
        self.exist_ok_checkbox.setChecked(False)

        self.validation_summary.setReadOnly(True)
        self.validation_summary.setMinimumHeight(150)
        self.validation_summary.setPlaceholderText(
            "Dataset ve eğitim ayarlarının doğrulama sonucu burada gösterilir."
        )

        self.training_log.setReadOnly(True)
        self.training_log.setMinimumHeight(260)
        self.training_log.setPlaceholderText(
            "Eğitim logları ve checkpoint bilgileri burada gösterilir."
        )

        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setMinimumHeight(24)

        self.epoch_status_label.setWordWrap(True)
        self.metric_status_label.setWordWrap(True)

        for button in (
            self.validate_button,
            self.start_button,
            self.stop_button,
            self.open_run_button,
            self.open_weights_button,
            self.clear_log_button,
            self.select_resume_button,
            self.find_latest_resume_button,
            self.clear_resume_button,
        ):
            button.setMinimumHeight(42)
            button.setCursor(Qt.CursorShape.PointingHandCursor)

        self.validate_button.setObjectName("secondaryButton")
        self.start_button.setObjectName("primaryButton")
        self.stop_button.setObjectName("dangerButton")
        self.open_run_button.setObjectName("secondaryButton")
        self.open_weights_button.setObjectName("secondaryButton")
        self.select_resume_button.setObjectName("secondaryButton")
        self.find_latest_resume_button.setObjectName("secondaryButton")
        self.resume_checkbox.setChecked(False)
        # Checkpoint seçim/bulma butonları her zaman kullanılabilir.
        # Bir checkpoint seçildiğinde resume modu otomatik açılır.
        self.resume_checkpoint_input.setEnabled(True)
        self.select_resume_button.setEnabled(True)
        self.find_latest_resume_button.setEnabled(True)

    @staticmethod
    def _configure_double_spin(
        control: QDoubleSpinBox,
        minimum: float,
        maximum: float,
        value: float,
        step: float,
        decimals: int,
    ) -> None:
        control.setRange(minimum, maximum)
        control.setDecimals(decimals)
        control.setSingleStep(step)
        control.setValue(value)
        control.setMinimumHeight(40)

    def _build_ui(self) -> None:
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(4, 4, 12, 20)
        content_layout.setSpacing(16)

        content_layout.addWidget(self._create_paths_card())
        content_layout.addWidget(self._create_resume_card())
        content_layout.addWidget(self._create_basic_settings_card())
        content_layout.addWidget(self._create_pose_loss_settings_card())
        content_layout.addWidget(self._create_optimizer_settings_card())
        content_layout.addWidget(self._create_advanced_settings_card())
        content_layout.addWidget(self._create_validation_card())
        content_layout.addWidget(self._create_progress_card())
        content_layout.addLayout(self._create_action_layout())
        content_layout.addWidget(self._create_log_card())
        content_layout.addStretch()

        scroll_area.setWidget(content)
        self._insert_into_base_layout(scroll_area)

    def _insert_into_base_layout(self, widget: QWidget) -> None:
        page_layout = self.layout()

        if page_layout is None:
            page_layout = QVBoxLayout(self)
            page_layout.setContentsMargins(32, 28, 32, 28)
            page_layout.setSpacing(16)

        # BasePage başlangıçta sayfanın ortasına geçici bir placeholder kartı
        # ekliyor. Eğitim arayüzü hazır olduğu için bu kartı kaldırıyoruz.
        # Başlık ve açıklama etiketlerine dokunulmuyor.
        self._remove_base_placeholder(page_layout)

        insert_index = page_layout.count()
        if insert_index > 0:
            last_item = page_layout.itemAt(insert_index - 1)
            if last_item is not None and last_item.spacerItem() is not None:
                insert_index -= 1

        page_layout.insertWidget(insert_index, widget, 1)

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

            contains_placeholder = any(
                placeholder_text in label.text().strip()
                for label in labels
            )

            if not contains_placeholder:
                continue

            page_layout.removeWidget(candidate)
            candidate.hide()
            candidate.setParent(None)
            candidate.deleteLater()

    def _create_paths_card(self) -> QFrame:
        card, layout = self._create_card(
            title="Dataset ve Çıktı Yolları",
            description=(
                "Split edilmiş data.yaml dosyasını, başlangıç modelini ve "
                "eğitim çıktılarının kaydedileceği klasörü seçin."
            ),
        )

        layout.addWidget(
            self._create_path_field(
                label_text="data.yaml",
                line_edit=self.data_yaml_input,
                button_text="Dosya Seç",
                callback=self._select_data_yaml,
            )
        )

        model_field = QWidget()
        model_layout = QVBoxLayout(model_field)
        model_layout.setContentsMargins(0, 0, 0, 0)
        model_layout.setSpacing(7)

        model_label = QLabel("Başlangıç modeli")
        model_label.setObjectName("inputLabel")

        preset_row = QHBoxLayout()
        preset_row.setContentsMargins(0, 0, 0, 0)
        preset_row.setSpacing(10)
        preset_label = QLabel("Hazır model:")
        preset_label.setMinimumWidth(92)
        preset_row.addWidget(preset_label)
        preset_row.addWidget(self.model_preset_combo, 1)

        model_path_row = self._create_path_row(
            line_edit=self.model_input,
            button_text="Model Seç",
            callback=self._select_model_file,
        )

        model_layout.addWidget(model_label)
        model_layout.addLayout(preset_row)
        model_layout.addWidget(model_path_row)
        layout.addWidget(model_field)

        layout.addWidget(
            self._create_path_field(
                label_text="Çıktı klasörü",
                line_edit=self.output_directory_input,
                button_text="Klasör Seç",
                callback=self._select_output_directory,
            )
        )

        run_name_container = QWidget()
        run_name_layout = QVBoxLayout(run_name_container)
        run_name_layout.setContentsMargins(0, 0, 0, 0)
        run_name_layout.setSpacing(7)
        run_name_label = QLabel("Eğitim çalışma adı")
        run_name_label.setObjectName("inputLabel")
        run_name_layout.addWidget(run_name_label)
        run_name_layout.addWidget(self.run_name_input)
        layout.addWidget(run_name_container)

        return card

    def _create_resume_card(self) -> QFrame:
        card, layout = self._create_card(
            title="Kesintiden Devam / Checkpoint Kurtarma",
            description=(
                "Elektrik kesintisi veya kullanıcı durdurması sonrasında "
                "last.pt seçilerek son tamamlanan epoch'tan devam edilir. "
                "Optimizer, scheduler ve epoch bilgisi checkpoint içinden yüklenir."
            ),
        )

        layout.addWidget(self.resume_checkbox)
        layout.addWidget(self.automatic_recovery_checkbox)

        checkpoint_label = QLabel("Devam checkpoint'i")
        checkpoint_label.setObjectName("inputLabel")
        layout.addWidget(checkpoint_label)

        checkpoint_row = QHBoxLayout()
        checkpoint_row.setContentsMargins(0, 0, 0, 0)
        checkpoint_row.setSpacing(10)
        checkpoint_row.addWidget(self.resume_checkpoint_input, 1)
        checkpoint_row.addWidget(self.select_resume_button)
        checkpoint_row.addWidget(self.find_latest_resume_button)
        layout.addLayout(checkpoint_row)
        layout.addWidget(self.clear_resume_button)

        note = QLabel(
            "Not: Resume yalnızca son tamamlanan epoch'tan devam eder. "
            "Yarım kalan epoch yeniden çalıştırılır. best.pt yerine last.pt kullanın."
        )
        note.setObjectName("inputLabel")
        note.setWordWrap(True)
        layout.addWidget(note)

        return card

    def _create_basic_settings_card(self) -> QFrame:
        card, layout = self._create_card(
            title="Temel Eğitim Ayarları",
            description=(
                "İlk denemelerde küçük epoch ve batch değerleri kullanın. "
                "Gerçek eğitimde dataset büyüklüğüne göre artırabilirsiniz."
            ),
        )

        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(14)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)

        controls = (
            ("Epoch", self.epochs_spin),
            ("Görüntü boyutu", self.image_size_spin),
            ("Batch size", self.batch_size_spin),
            ("Device", self.device_combo),
            ("Workers", self.workers_spin),
            ("Patience", self.patience_spin),
            ("Optimizer", self.optimizer_combo),
            ("Random seed", self.seed_spin),
        )

        for index, (label_text, control) in enumerate(controls):
            row = index // 2
            column = index % 2
            grid.addWidget(
                self._create_labeled_control(label_text, control),
                row,
                column,
            )

        layout.addLayout(grid)
        return card

    def _create_pose_loss_settings_card(self) -> QFrame:
        card, layout = self._create_card(
            title="Detection ve Pose Loss Ayarları",
            description=(
                "Egzersiz asistanında insan bounding box doğruluğu ile anatomik "
                "keypoint doğruluğu birlikte eğitilir. Varsayılanlar Ultralytics "
                "Pose yapılandırmasıdır."
            ),
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)
        controls = (
            ("Box loss gain", self.box_loss_spin),
            ("Classification loss gain", self.cls_loss_spin),
            ("DFL loss gain", self.dfl_loss_spin),
            ("Pose loss gain", self.pose_loss_spin),
            ("Keypoint objectness gain", self.kobj_loss_spin),
            ("RLE loss gain", self.rle_loss_spin),
            ("Nominal batch size", self.nominal_batch_spin),
        )
        for index, (label, control) in enumerate(controls):
            grid.addWidget(
                self._create_labeled_control(label, control),
                index // 3,
                index % 3,
            )
        layout.addLayout(grid)
        return card

    def _create_optimizer_settings_card(self) -> QFrame:
        card, layout = self._create_card(
            title="Learning Rate ve Warmup Ayarları",
            description=(
                "Optimizer davranışını ve eğitimin ilk epoch'lardaki ısınma "
                "sürecini kontrol eder."
            ),
        )
        grid = QGridLayout()
        grid.setHorizontalSpacing(14)
        grid.setVerticalSpacing(12)
        controls = (
            ("Initial LR (lr0)", self.lr0_spin),
            ("Final LR fraction (lrf)", self.lrf_spin),
            ("Momentum / beta1", self.momentum_spin),
            ("Weight decay", self.weight_decay_spin),
            ("Warmup epochs", self.warmup_epochs_spin),
            ("Warmup momentum", self.warmup_momentum_spin),
            ("Warmup bias LR", self.warmup_bias_lr_spin),
            ("Close mosaic", self.close_mosaic_spin),
        )
        for index, (label, control) in enumerate(controls):
            grid.addWidget(
                self._create_labeled_control(label, control),
                index // 3,
                index % 3,
            )
        layout.addLayout(grid)
        layout.addWidget(self.cos_lr_checkbox)
        return card

    def _create_advanced_settings_card(self) -> QFrame:
        card, layout = self._create_card(
            title="Gelişmiş Ayarlar",
            description=(
                "MPS testinde AMP kapalı tutulabilir. last.pt her tamamlanan "
                "epoch sonunda güncellenir. Checkpoint aralığı 1 yapılırsa "
                "epoch*.pt yedekleri de saklanır."
            ),
        )

        save_period_container = self._create_labeled_control(
            "Checkpoint kayıt aralığı",
            self.save_period_spin,
        )
        layout.addWidget(save_period_container)

        checkbox_grid = QGridLayout()
        checkbox_grid.setHorizontalSpacing(18)
        checkbox_grid.setVerticalSpacing(12)

        checkboxes = (
            self.pretrained_checkbox,
            self.deterministic_checkbox,
            self.cache_checkbox,
            self.amp_checkbox,
            self.plots_checkbox,
            self.exist_ok_checkbox,
        )

        for index, checkbox in enumerate(checkboxes):
            checkbox_grid.addWidget(checkbox, index // 2, index % 2)

        layout.addLayout(checkbox_grid)
        return card

    def _create_validation_card(self) -> QFrame:
        card, layout = self._create_card(
            title="Eğitim Öncesi Doğrulama",
            description=(
                "Train, validation ve test yolları ile cihaz seçimi eğitim "
                "başlatılmadan önce kontrol edilir."
            ),
        )
        layout.addWidget(self.validation_summary)
        return card

    def _create_progress_card(self) -> QFrame:
        card, layout = self._create_card(
            title="Eğitim İlerlemesi",
            description="Epoch ilerlemesi ve son metrik özeti canlı güncellenir.",
        )
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.epoch_status_label)
        layout.addWidget(self.metric_status_label)
        return card

    def _create_action_layout(self) -> QGridLayout:
        layout = QGridLayout()
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        layout.setColumnStretch(0, 1)
        layout.setColumnStretch(1, 1)
        layout.setColumnStretch(2, 1)

        layout.addWidget(self.validate_button, 0, 0)
        layout.addWidget(self.start_button, 0, 1)
        layout.addWidget(self.stop_button, 0, 2)
        layout.addWidget(self.open_run_button, 1, 0)
        layout.addWidget(self.open_weights_button, 1, 1)
        layout.addWidget(self.clear_log_button, 1, 2)

        return layout

    def _create_log_card(self) -> QFrame:
        card, layout = self._create_card(
            title="Canlı Eğitim Logları",
            description=(
                "Ultralytics eğitim başlangıcı, epoch tamamlanması, checkpoint "
                "kayıtları ve hata mesajları burada gösterilir."
            ),
        )
        layout.addWidget(self.training_log)
        return card

    @staticmethod
    def _create_card(
        *,
        title: str,
        description: str,
    ) -> tuple[QFrame, QVBoxLayout]:
        card = QFrame()
        card.setObjectName("formCard")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(22, 20, 22, 22)
        layout.setSpacing(13)

        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")

        description_label = QLabel(description)
        description_label.setObjectName("inputLabel")
        description_label.setWordWrap(True)

        layout.addWidget(title_label)
        layout.addWidget(description_label)

        return card, layout

    def _create_path_field(
        self,
        *,
        label_text: str,
        line_edit: QLineEdit,
        button_text: str,
        callback: Callable[[], None],
    ) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(7)

        label = QLabel(label_text)
        label.setObjectName("inputLabel")

        layout.addWidget(label)
        layout.addWidget(
            self._create_path_row(
                line_edit=line_edit,
                button_text=button_text,
                callback=callback,
            )
        )
        return container

    @staticmethod
    def _create_path_row(
        *,
        line_edit: QLineEdit,
        button_text: str,
        callback: Callable[[], None],
    ) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        button = QPushButton(button_text)
        button.setMinimumWidth(118)
        button.setMinimumHeight(40)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(callback)

        layout.addWidget(line_edit, 1)
        layout.addWidget(button)
        return container

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

    # ------------------------------------------------------------------
    # Sinyaller ve seçim pencereleri
    # ------------------------------------------------------------------

    def _connect_signals(self) -> None:
        """Arayüz sinyallerini doğrudan ve tek kez bağlar."""

        self.model_preset_combo.currentIndexChanged.connect(
            self._apply_model_preset
        )
        self.validate_button.pressed.connect(self.validate_settings)
        self.start_button.pressed.connect(self.start_training)

        # Bu üç kritik düğmede clicked(bool) veya lambda kullanılmıyor.
        # pressed sinyali macOS/PySide6 üzerinde doğrudan tetiklenir.
        self.stop_button.pressed.connect(self.stop_training)
        self.open_run_button.pressed.connect(self.open_run_directory)
        self.open_weights_button.pressed.connect(self.open_weights_directory)

        self.clear_log_button.pressed.connect(self.training_log.clear)
        self.resume_checkbox.toggled.connect(self._toggle_resume_mode)
        self.select_resume_button.pressed.connect(
            self._select_resume_checkpoint
        )
        self.find_latest_resume_button.pressed.connect(
            self._find_latest_resume_checkpoint
        )
        self.clear_resume_button.pressed.connect(
            self._clear_resume_selection
        )
        self.output_directory_input.editingFinished.connect(
            lambda: self._detect_recovery_state(show_message=False)
        )

    @Slot(int)
    def _apply_model_preset(self, _index: int = -1) -> None:
        """Hazır model seçimini model alanına uygular."""

        model_name = self.model_preset_combo.currentData()
        if model_name and model_name != "__custom__":
            self.model_input.setText(str(model_name))

    def _select_data_yaml(self) -> None:
        start_directory = self._start_directory_for_path(
            self.data_yaml_input.text(),
            use_parent=True,
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Split Dataset data.yaml Dosyasını Seç",
            start_directory,
            "YAML Dosyaları (*.yaml *.yml);;Tüm Dosyalar (*)",
        )
        if file_path:
            self.data_yaml_input.setText(file_path)

    def _select_model_file(self) -> None:
        start_directory = self._start_directory_for_path(
            self.model_input.text(),
            use_parent=True,
        )
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "YOLO Pose Modelini Seç",
            start_directory,
            "YOLO Model Dosyaları (*.pt *.yaml *.yml);;Tüm Dosyalar (*)",
        )
        if file_path:
            self.model_input.setText(file_path)
            custom_index = self.model_preset_combo.findData("__custom__")
            if custom_index >= 0:
                self.model_preset_combo.setCurrentIndex(custom_index)
            self._append_log(f"Başlangıç modeli seçildi: {file_path}")

    def _detect_recovery_state(self, *, show_message: bool) -> bool:
        if self._is_training_running():
            return False
        output_directory = self.output_directory_input.text().strip() or "runs"
        recovery = self.training_service.find_recovery_state(output_directory)
        if recovery is None:
            if show_message:
                QMessageBox.information(
                    self,
                    "Recovery Bulunamadı",
                    "Geçerli last.pt içeren yarım kalmış eğitim bulunamadı.",
                )
            return False

        recovered_settings = recovery.settings
        recovered_data_yaml = str(
            recovered_settings.get("data_yaml_path", "")
        ).strip()
        if recovered_data_yaml and Path(recovered_data_yaml).expanduser().is_file():
            self.data_yaml_input.setText(recovered_data_yaml)

        recovered_output = str(
            recovered_settings.get("output_directory", "")
        ).strip()
        if recovered_output:
            self.output_directory_input.setText(recovered_output)

        recovered_run_name = str(recovered_settings.get("run_name", "")).strip()
        if recovered_run_name:
            self.run_name_input.setText(recovered_run_name)

        self.resume_checkpoint_input.setText(str(recovery.checkpoint_path))
        self.resume_checkbox.setChecked(True)
        self._append_log(
            "Yarım kalmış eğitim bulundu: "
            f"epoch={recovery.last_completed_epoch}/{recovery.total_epochs}, "
            f"checkpoint={recovery.checkpoint_path}"
        )
        self._append_log(
            "Recovery modu aktifleştirildi. Resume sırasında arayüzdeki epoch "
            "değeri kullanılmaz; checkpoint içindeki özgün hedef epoch korunur."
        )
        if show_message:
            QMessageBox.information(
                self,
                "Eğitim Recovery Bulundu",
                (
                    "Elektrik kesintisi veya beklenmeyen kapanma sonrası "
                    "kullanılabilecek checkpoint bulundu.\n\n"
                    f"Son tamamlanan epoch: {recovery.last_completed_epoch}\n"
                    f"Checkpoint:\n{recovery.checkpoint_path}"
                ),
            )
        return True

    def _select_resume_checkpoint(self) -> None:
        if self._is_training_running():
            QMessageBox.warning(
                self,
                "Eğitim Devam Ediyor",
                "Eğitim sürerken checkpoint seçilemez.",
            )
            return

        current_checkpoint = self.resume_checkpoint_input.text().strip()
        output_directory = self.output_directory_input.text().strip() or "runs"

        if current_checkpoint:
            start_path = Path(current_checkpoint).expanduser()
            start_directory = (
                start_path.parent
                if start_path.parent.is_dir()
                else Path.home()
            )
        else:
            output_path = Path(output_directory).expanduser()
            start_directory = (
                output_path.resolve()
                if output_path.is_dir()
                else Path.cwd()
            )

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Devam Checkpoint'ini Seç",
            str(start_directory),
            "YOLO Checkpoint (*.pt);;Tüm Dosyalar (*)",
        )
        if not file_path:
            self._append_log("Checkpoint seçimi iptal edildi.")
            return

        self._apply_resume_checkpoint(Path(file_path))

    def _find_latest_resume_checkpoint(self) -> None:
        if self._is_training_running():
            QMessageBox.warning(
                self,
                "Eğitim Devam Ediyor",
                "Eğitim sürerken checkpoint aranamaz.",
            )
            return

        search_roots = self._checkpoint_search_roots()
        candidates: list[Path] = []

        for root in search_roots:
            checkpoint = self.training_service.find_latest_checkpoint(str(root))
            if checkpoint is not None and checkpoint.is_file():
                candidates.append(checkpoint.resolve())

        if not candidates:
            searched_text = "\n".join(f"- {path}" for path in search_roots)
            QMessageBox.warning(
                self,
                "Checkpoint Bulunamadı",
                (
                    "Aşağıdaki klasörlerde last.pt veya epoch*.pt bulunamadı:\n\n"
                    f"{searched_text}"
                ),
            )
            self._append_log("En son checkpoint bulunamadı.")
            return

        checkpoint = max(
            set(candidates),
            key=lambda path: path.stat().st_mtime,
        )
        self._apply_resume_checkpoint(checkpoint)
        QMessageBox.information(
            self,
            "Checkpoint Bulundu",
            f"En son checkpoint seçildi:\n\n{checkpoint}",
        )

    def _checkpoint_search_roots(self) -> list[Path]:
        roots: list[Path] = []

        raw_output = self.output_directory_input.text().strip() or "runs"
        output_path = Path(raw_output).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        roots.append(output_path.resolve())

        run_name = self.run_name_input.text().strip()
        if run_name:
            roots.insert(0, (output_path / run_name).resolve())

        selected_checkpoint = self.resume_checkpoint_input.text().strip()
        if selected_checkpoint:
            checkpoint_path = Path(selected_checkpoint).expanduser()
            if checkpoint_path.is_file():
                roots.insert(0, checkpoint_path.resolve().parent.parent)

        default_runs = (Path.cwd() / "runs").resolve()
        roots.append(default_runs)

        unique_roots: list[Path] = []
        seen: set[Path] = set()
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            unique_roots.append(root)

        return unique_roots

    def _apply_resume_checkpoint(self, checkpoint: Path) -> None:
        checkpoint = checkpoint.expanduser().resolve()

        if not checkpoint.is_file():
            QMessageBox.warning(
                self,
                "Checkpoint Bulunamadı",
                f"Checkpoint dosyası bulunamadı:\n{checkpoint}",
            )
            return

        if checkpoint.suffix.lower() != ".pt":
            QMessageBox.warning(
                self,
                "Geçersiz Checkpoint",
                "Checkpoint bir .pt dosyası olmalıdır.",
            )
            return

        if checkpoint.name == "best.pt":
            QMessageBox.warning(
                self,
                "best.pt Kullanılamaz",
                "Resume için best.pt yerine last.pt veya epoch*.pt seçin.",
            )
            return

        self.resume_checkpoint_input.setText(str(checkpoint))

        # Checkpoint seçildiğinde devam modu otomatik açılır.
        if not self.resume_checkbox.isChecked():
            self.resume_checkbox.setChecked(True)

        weights_directory = checkpoint.parent
        run_directory = (
            weights_directory.parent
            if weights_directory.name == "weights"
            else weights_directory
        )
        output_directory = run_directory.parent

        self.output_directory_input.setText(str(output_directory))
        self.run_name_input.setText(run_directory.name)

        args_yaml = run_directory / "args.yaml"
        if args_yaml.is_file():
            try:
                import yaml

                args_data = yaml.safe_load(
                    args_yaml.read_text(encoding="utf-8")
                )
                if isinstance(args_data, dict):
                    data_value = args_data.get("data")
                    if data_value:
                        data_path = Path(str(data_value)).expanduser()
                        if data_path.is_file():
                            self.data_yaml_input.setText(
                                str(data_path.resolve())
                            )
            except (OSError, yaml.YAMLError):
                pass

        self._append_log(f"Resume checkpoint seçildi: {checkpoint}")
        self._append_log(f"Eğitim klasörü algılandı: {run_directory}")

    @Slot(bool)
    def _toggle_resume_mode(self, checked: bool) -> None:
        if checked and not self.resume_checkpoint_input.text().strip():
            checkpoint = self.training_service.find_latest_checkpoint(
                self.output_directory_input.text().strip() or "runs"
            )
            if checkpoint is not None:
                self.resume_checkpoint_input.setText(str(checkpoint))

        self.start_button.setText(
            "Eğitime Devam Et" if checked else "Eğitimi Başlat"
        )
        self._apply_editable_control_state(
            running=self._is_training_running()
        )

    @Slot()
    def _clear_resume_selection(self) -> None:
        """Checkpoint devamını kapatır ve yeni eğitim moduna döner."""

        if self._is_training_running():
            QMessageBox.warning(
                self,
                "Eğitim Devam Ediyor",
                "Eğitim sürerken resume modu değiştirilemez.",
            )
            return

        self.resume_checkbox.setChecked(False)
        self.resume_checkpoint_input.clear()
        self.start_button.setText("Eğitimi Başlat")
        self.epoch_status_label.setText(
            "Yeni eğitim modu seçildi. Epoch ve Pose parametreleri arayüzden kullanılacak."
        )
        self._append_log(
            "Resume seçimi temizlendi. Yeni eğitim modu aktif; epoch ve Pose "
            "parametreleri arayüzden alınacak."
        )

    def _select_output_directory(self) -> None:
        start_directory = self._start_directory_for_path(
            self.output_directory_input.text(),
            use_parent=False,
        )
        directory = QFileDialog.getExistingDirectory(
            self,
            "Eğitim Çıktı Klasörünü Seç",
            start_directory,
            QFileDialog.Option.ShowDirsOnly,
        )
        if directory:
            self.output_directory_input.setText(directory)

    @staticmethod
    def _start_directory_for_path(raw_path: str, *, use_parent: bool) -> str:
        text = raw_path.strip()
        if not text:
            return str(Path.home())

        path = Path(text).expanduser()
        if use_parent:
            candidate = path.parent
        else:
            candidate = path

        if candidate.is_dir():
            return str(candidate.resolve())
        return str(Path.home())

    # ------------------------------------------------------------------
    # Ayar üretme ve doğrulama
    # ------------------------------------------------------------------

    def _create_settings(self) -> TrainingSettings:
        data_yaml_path = self.data_yaml_input.text().strip()
        model_path = self.model_input.text().strip()
        output_directory = self.output_directory_input.text().strip()
        run_name = self.run_name_input.text().strip()

        resume_enabled = self.resume_checkbox.isChecked()
        resume_checkpoint_path = self.resume_checkpoint_input.text().strip()

        if not data_yaml_path:
            raise ValueError("data.yaml seçilmedi.")
        if not output_directory:
            raise ValueError("Çıktı klasörü belirtilmedi.")

        if resume_enabled:
            if not resume_checkpoint_path:
                raise ValueError(
                    "Devam modu açık ancak last.pt checkpoint'i seçilmedi."
                )
        else:
            if not model_path:
                raise ValueError("Başlangıç modeli seçilmedi.")
            if not run_name:
                raise ValueError("Eğitim çalışma adı belirtilmedi.")

        return TrainingSettings(
            data_yaml_path=data_yaml_path,
            model_path=model_path,
            output_directory=output_directory,
            run_name=run_name,
            epochs=self.epochs_spin.value(),
            image_size=self.image_size_spin.value(),
            batch_size=self.batch_size_spin.value(),
            device=str(self.device_combo.currentData()),
            workers=self.workers_spin.value(),
            patience=self.patience_spin.value(),
            optimizer=str(self.optimizer_combo.currentData()),
            seed=self.seed_spin.value(),
            deterministic=self.deterministic_checkbox.isChecked(),
            pretrained=self.pretrained_checkbox.isChecked(),
            cache=self.cache_checkbox.isChecked(),
            amp=self.amp_checkbox.isChecked(),
            plots=self.plots_checkbox.isChecked(),
            save_period=self.save_period_spin.value(),
            exist_ok=self.exist_ok_checkbox.isChecked(),
            box_loss_gain=self.box_loss_spin.value(),
            cls_loss_gain=self.cls_loss_spin.value(),
            dfl_loss_gain=self.dfl_loss_spin.value(),
            pose_loss_gain=self.pose_loss_spin.value(),
            keypoint_objectness_gain=self.kobj_loss_spin.value(),
            rle_loss_gain=self.rle_loss_spin.value(),
            nominal_batch_size=self.nominal_batch_spin.value(),
            initial_learning_rate=self.lr0_spin.value(),
            final_learning_rate_fraction=self.lrf_spin.value(),
            momentum=self.momentum_spin.value(),
            weight_decay=self.weight_decay_spin.value(),
            warmup_epochs=self.warmup_epochs_spin.value(),
            warmup_momentum=self.warmup_momentum_spin.value(),
            warmup_bias_learning_rate=self.warmup_bias_lr_spin.value(),
            cosine_learning_rate=self.cos_lr_checkbox.isChecked(),
            close_mosaic=self.close_mosaic_spin.value(),
            automatic_recovery=self.automatic_recovery_checkbox.isChecked(),
            resume=resume_enabled,
            resume_checkpoint_path=resume_checkpoint_path,
        )

    @Slot()
    def validate_settings(self) -> bool:
        try:
            settings = self._create_settings()
            result = self.training_service.validate_settings(settings)
        except Exception as error:
            self.validation_summary.setPlainText(f"HATA: {error}")
            QMessageBox.warning(
                self,
                "Eğitim Ayarları Geçersiz",
                str(error),
            )
            return False

        lines = [
            "Eğitim ayarları geçerli.",
            "",
            f"Mod: {'Checkpoint devamı' if result.resume_mode else 'Yeni eğitim'}",
            f"Model: {settings.model_path}",
            f"Çalışma adı: {settings.run_name}",
            (
                f"Arayüz epoch değeri: {settings.epochs} "
                "(resume modunda kullanılmaz)"
                if settings.resume
                else f"Epoch: {settings.epochs}"
            ),
            f"Görüntü boyutu: {settings.image_size}",
            f"Batch size: {settings.batch_size}",
            f"Cihaz: {result.resolved_device}",
            f"Workers: {settings.workers}",
            f"Patience: {settings.patience}",
            f"Optimizer: {settings.optimizer}",
            f"Seed: {settings.seed}",
            f"Checkpoint aralığı: {1 if settings.automatic_recovery else settings.save_period}",
            f"Keypoint yapısı: {result.keypoint_count} nokta × {result.keypoint_dimensions} boyut",
            f"Anatomik keypoint adları: {result.keypoint_names_defined}",
            f"Box / Cls / DFL: {settings.box_loss_gain} / {settings.cls_loss_gain} / {settings.dfl_loss_gain}",
            f"Pose / KObj / RLE: {settings.pose_loss_gain} / {settings.keypoint_objectness_gain} / {settings.rle_loss_gain}",
            f"lr0 / lrf: {settings.initial_learning_rate} / {settings.final_learning_rate_fraction}",
            f"Momentum / weight decay: {settings.momentum} / {settings.weight_decay}",
            f"Warmup epochs: {settings.warmup_epochs}",
            f"Otomatik recovery: {settings.automatic_recovery}",
            f"Pretrained: {settings.pretrained}",
            f"Deterministik: {settings.deterministic}",
            f"Cache: {settings.cache}",
            f"AMP: {settings.amp}",
            f"Plots: {settings.plots}",
            f"Exist OK: {settings.exist_ok}",
            "",
            f"Train yolu: {result.train_summary.resolved_path}",
            f"Train görsel: {result.train_summary.image_count}",
            f"Validation yolu: {result.val_summary.resolved_path}",
            f"Validation görsel: {result.val_summary.image_count}",
        ]

        if result.resume_checkpoint_path is not None:
            lines.extend(
                [
                    "",
                    f"Devam checkpoint'i: {result.resume_checkpoint_path}",
                ]
            )

        if result.test_summary is not None:
            lines.extend(
                [
                    f"Test yolu: {result.test_summary.resolved_path}",
                    f"Test görsel: {result.test_summary.image_count}",
                ]
            )

        if result.warnings:
            lines.append("")
            lines.append("Uyarılar:")
            lines.extend(f"- {warning}" for warning in result.warnings)

        self.validation_summary.setPlainText("\n".join(lines))
        self._append_log("Eğitim ayarları doğrulandı.")

        QMessageBox.information(
            self,
            "Doğrulama Başarılı",
            (
                "Eğitim ayarları geçerli.\n\n"
                f"Cihaz: {result.resolved_device}\n"
                f"Train: {result.train_summary.image_count}\n"
                f"Validation: {result.val_summary.image_count}"
            ),
        )
        return True

    # ------------------------------------------------------------------
    # Eğitim thread yönetimi
    # ------------------------------------------------------------------

    @Slot()
    def start_training(self) -> None:
        if self._is_training_running():
            QMessageBox.warning(
                self,
                "Eğitim Devam Ediyor",
                "Aynı anda yalnızca bir eğitim başlatılabilir.",
            )
            return

        try:
            settings = self._create_settings()
            validation = self.training_service.validate_settings(settings)
        except Exception as error:
            self.validation_summary.setPlainText(f"HATA: {error}")
            QMessageBox.warning(
                self,
                "Eğitim Başlatılamadı",
                str(error),
            )
            return

        if settings.resume:
            confirmation_text = (
                "Kesintiye uğrayan YOLO Pose eğitimi son tamamlanan "
                "epoch'tan devam ettirilecek.\n\n"
                f"Checkpoint:\n{validation.resume_checkpoint_path}\n\n"
                f"Cihaz: {validation.resolved_device}\n\n"
                "Devam edilsin mi?"
            )
            confirmation_title = "Eğitime Devam Et"
        else:
            confirmation_text = (
                f"{settings.epochs} epoch YOLO Pose eğitimi "
                f"{validation.resolved_device} cihazında başlatılacak.\n\n"
                "Devam edilsin mi?"
            )
            confirmation_title = "Eğitimi Başlat"

        if settings.resume or settings.epochs > 1:
            question = QMessageBox.question(
                self,
                confirmation_title,
                confirmation_text,
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes,
            )
            if question != QMessageBox.StandardButton.Yes:
                return

        # Durdurma bayrağı thread başlamadan önce sıfırlanır.
        # Böylece kullanıcı Başlat'a basar basmaz Durdur'a basarsa istek kaybolmaz.
        self.training_service.reset_stop_request()
        self.stop_request_sent = False

        self.last_training_result = None
        self.progress_bar.setValue(0)
        self.epoch_status_label.setText("Eğitim hazırlanıyor...")
        self.metric_status_label.setText("Henüz epoch metriği alınmadı.")
        self._append_log("")
        self._append_log("=" * 68)
        self._append_log(
            "Checkpoint'ten eğitim devam ettiriliyor."
            if settings.resume
            else "YOLO Pose eğitimi arka planda başlatılıyor."
        )
        self._append_log(
            "Seçilen ayarlar: "
            f"model={settings.model_path}, run={settings.run_name}, "
            f"epochs={settings.epochs}, imgsz={settings.image_size}, "
            f"batch={settings.batch_size}, device={settings.device}, "
            f"workers={settings.workers}, patience={settings.patience}, "
            f"optimizer={settings.optimizer}, seed={settings.seed}, "
            f"save_period={settings.save_period}, "
            f"pretrained={settings.pretrained}, cache={settings.cache}, "
            f"amp={settings.amp}, plots={settings.plots}, "
            f"deterministic={settings.deterministic}, "
            f"exist_ok={settings.exist_ok}, "
            f"box={settings.box_loss_gain}, cls={settings.cls_loss_gain}, "
            f"dfl={settings.dfl_loss_gain}, pose={settings.pose_loss_gain}, "
            f"kobj={settings.keypoint_objectness_gain}, rle={settings.rle_loss_gain}, "
            f"lr0={settings.initial_learning_rate}, "
            f"lrf={settings.final_learning_rate_fraction}, "
            f"auto_recovery={settings.automatic_recovery}"
        )
        self._append_log("=" * 68)

        self.training_thread = QThread(self)
        self.training_worker = TrainingWorker(
            service=self.training_service,
            settings=settings,
        )
        self.training_worker.moveToThread(self.training_thread)

        self.training_thread.started.connect(self.training_worker.run)
        self.training_worker.log_message.connect(self._append_log)
        self.training_worker.progress_changed.connect(self._handle_progress)
        self.training_worker.training_finished.connect(
            self._handle_training_finished
        )
        self.training_worker.training_failed.connect(
            self._handle_training_failed
        )
        self.training_worker.worker_finished.connect(self.training_thread.quit)
        self.training_worker.worker_finished.connect(
            self.training_worker.deleteLater
        )
        self.training_thread.finished.connect(self._handle_thread_finished)
        self.training_thread.finished.connect(self.training_thread.deleteLater)

        self._set_training_state(True)
        self.training_thread.start()

    @Slot()
    def stop_training(self) -> None:
        """Aktif eğitimi mevcut epoch tamamlanınca güvenli biçimde durdurur."""

        print(f"[TrainingPage {BUILD_ID}] stop button pressed", flush=True)
        self._append_log("[BUTON] Epoch Sonunda Durdur düğmesine basıldı.")

        if not self._is_training_running():
            self.epoch_status_label.setText(
                "Durdurulacak aktif bir eğitim bulunmuyor."
            )
            self._append_log("Aktif eğitim olmadığı için durdurma uygulanmadı.")
            return

        if self.stop_request_sent:
            self.epoch_status_label.setText(
                "Durdurma isteği zaten gönderildi; epoch sonu bekleniyor."
            )
            self._append_log("Durdurma isteği daha önce gönderilmişti.")
            return

        trainer_available = self.training_service.request_stop()
        self.stop_request_sent = True
        self.stop_button.setEnabled(False)
        self.stop_button.setText("Epoch Sonunda Durdurulacak")
        self.epoch_status_label.setText(
            "Durdurma isteği alındı. Mevcut epoch, validation ve last.pt "
            "kaydı tamamlanınca eğitim duracak."
        )
        self._append_log(
            "DURDURMA İSTEĞİ KABUL EDİLDİ: mevcut epoch tamamlanacak, "
            "last.pt kaydedilecek ve sonraki epoch başlamayacak."
        )
        self._append_log(
            "Aktif Ultralytics trainer bulundu."
            if trainer_available
            else "Trainer henüz kurulmadı; istek ilk epoch sonunda uygulanacak."
        )
        QApplication.processEvents()

    @Slot(object)
    def _handle_progress(self, progress: TrainingProgress) -> None:
        value = max(0, min(1000, int(progress.percent * 10)))
        self.progress_bar.setValue(value)
        self.progress_bar.setFormat(
            f"Epoch {progress.epoch}/{progress.total_epochs} — "
            f"%{progress.percent:.1f}"
        )
        self.epoch_status_label.setText(progress.message)

        if progress.metrics:
            metric_parts = [
                f"{name}: {value:.4f}"
                for name, value in list(progress.metrics.items())[:12]
            ]
            self.metric_status_label.setText(" | ".join(metric_parts))
        else:
            self.metric_status_label.setText(
                "Bu epoch için metrik bilgisi alınamadı."
            )

    @Slot(object)
    def _handle_training_finished(self, result: TrainingResult) -> None:
        self.last_training_result = result
        self.progress_bar.setValue(1000)
        self.progress_bar.setFormat("Eğitim tamamlandı — %100")

        if result.stopped_early:
            self.epoch_status_label.setText(
                "Eğitim kullanıcı isteğiyle erken durduruldu."
            )
        else:
            self.epoch_status_label.setText("Eğitim başarıyla tamamlandı.")

        if result.metrics:
            metric_parts = [
                f"{name}: {value:.4f}"
                for name, value in list(result.metrics.items())[:12]
            ]
            self.metric_status_label.setText(" | ".join(metric_parts))

        self._append_log("")
        self._append_log("=" * 68)
        self._append_log("EĞİTİM SONUCU")
        self._append_log("=" * 68)
        self._append_log(f"Cihaz: {result.resolved_device}")
        self._append_log(f"Çalışma klasörü: {result.run_directory}")
        self._append_log(f"Süre: {result.elapsed_seconds:.2f} saniye")
        self._append_log(f"Erken durduruldu: {result.stopped_early}")

        if result.best_model_path is not None:
            self._append_log(f"best.pt: {result.best_model_path}")
        else:
            self._append_log("best.pt bulunamadı.")

        if result.last_model_path is not None:
            self._append_log(f"last.pt: {result.last_model_path}")
            self.resume_checkpoint_input.setText(str(result.last_model_path))

        if result.resumed:
            self._append_log("Bu çalışma bir checkpoint'ten devam ettirildi.")

        if result.results_csv_path is not None:
            self._append_log(f"results.csv: {result.results_csv_path}")
        if result.recovery_state_path is not None and result.stopped_early:
            self._append_log(
                f"Recovery durumu: {result.recovery_state_path}"
            )

        self.open_run_button.setEnabled(True)
        self.open_weights_button.setEnabled(result.weights_directory.is_dir())

        completion_text = (
            "Eğitim checkpoint kaydedilerek durduruldu. Daha sonra last.pt "
            "ile kaldığınız yerden devam edebilirsiniz."
            if result.stopped_early
            else "Eğitim tamamlandı."
        )

        QMessageBox.information(
            self,
            "YOLO Pose Eğitimi Tamamlandı",
            (
                f"{completion_text}\n\n"
                f"Süre: {result.elapsed_seconds:.2f} saniye\n"
                f"Çalışma klasörü:\n{result.run_directory}\n\n"
                f"last.pt:\n{result.last_model_path or 'Bulunamadı'}\n\n"
                f"best.pt:\n{result.best_model_path or 'Bulunamadı'}"
            ),
        )

    @Slot(str)
    def _handle_training_failed(self, message: str) -> None:
        self.progress_bar.setFormat("Eğitim başarısız")
        self.epoch_status_label.setText("Eğitim hata nedeniyle tamamlanamadı.")
        self.metric_status_label.setText(message)
        self._append_log("")
        self._append_log(f"EĞİTİM HATASI: {message}")

        QMessageBox.critical(
            self,
            "YOLO Pose Eğitim Hatası",
            message,
        )

    @Slot()
    def _handle_thread_finished(self) -> None:
        self.training_thread = None
        self.training_worker = None
        self.stop_request_sent = False
        self._set_training_state(False)

    def _set_training_state(self, running: bool) -> None:
        self.validate_button.setEnabled(not running)
        self.start_button.setEnabled(not running)

        # Buton her zaman tıklanabilir. Aktif eğitim yoksa kullanıcıya
        # açıklayıcı mesaj gösterilir; eğitim varsa güvenli durdurma istenir.
        self.stop_button.setEnabled(
            not running or not self.stop_request_sent
        )
        self.stop_button.setText(
            "Epoch Sonunda Durdurulacak"
            if running and self.stop_request_sent
            else "Epoch Sonunda Durdur"
        )

        self._apply_editable_control_state(running=running)

        # Klasör düğmeleri eğitim sırasında da kullanılabilir. Eğitim devam
        # ederken mevcut çalışma klasörü ve weights klasörü Finder'da açılır.
        self.open_run_button.setEnabled(True)
        self.open_weights_button.setEnabled(True)

    def _apply_editable_control_state(self, *, running: bool) -> None:
        """Eğitim sürmüyorsa bütün ayar alanlarını düzenlenebilir yapar.

        Resume modu yalnızca eğitimin başlatılma biçimini değiştirir. Arayüzde
        model, çalışma adı, temel ayarlar veya gelişmiş ayarlar kilitlenmez.
        Böylece kullanıcı istediği değeri görebilir ve değiştirebilir.
        """

        editable_controls = (
            self.data_yaml_input,
            self.model_input,
            self.output_directory_input,
            self.run_name_input,
            self.model_preset_combo,
            self.epochs_spin,
            self.image_size_spin,
            self.batch_size_spin,
            self.device_combo,
            self.workers_spin,
            self.patience_spin,
            self.optimizer_combo,
            self.seed_spin,
            self.save_period_spin,
            self.box_loss_spin,
            self.cls_loss_spin,
            self.dfl_loss_spin,
            self.pose_loss_spin,
            self.kobj_loss_spin,
            self.rle_loss_spin,
            self.nominal_batch_spin,
            self.lr0_spin,
            self.lrf_spin,
            self.momentum_spin,
            self.weight_decay_spin,
            self.warmup_epochs_spin,
            self.warmup_momentum_spin,
            self.warmup_bias_lr_spin,
            self.close_mosaic_spin,
            self.cos_lr_checkbox,
            self.automatic_recovery_checkbox,
            self.pretrained_checkbox,
            self.deterministic_checkbox,
            self.cache_checkbox,
            self.amp_checkbox,
            self.plots_checkbox,
            self.exist_ok_checkbox,
            self.resume_checkbox,
            self.resume_checkpoint_input,
            self.select_resume_button,
            self.find_latest_resume_button,
        )

        for control in editable_controls:
            control.setEnabled(not running)

        # QLineEdit alanlarının salt okunur kalma ihtimalini de temizliyoruz.
        for line_edit in (
            self.data_yaml_input,
            self.model_input,
            self.output_directory_input,
            self.run_name_input,
            self.resume_checkpoint_input,
        ):
            line_edit.setReadOnly(running)

        if not running:
            self.start_button.setText(
                "Eğitime Devam Et"
                if self.resume_checkbox.isChecked()
                else "Eğitimi Başlat"
            )

    def _is_training_running(self) -> bool:
        # Thread start() çağrısı ile isRunning() değerinin True olması arasında
        # çok kısa bir pencere olabilir. Worker ve thread oluşturulduysa eğitim
        # başlatılmış kabul edilir; böylece Durdur butonuna hemen basıldığında
        # istek kaybolmaz.
        return (
            self.training_thread is not None
            and self.training_worker is not None
        )

    # ------------------------------------------------------------------
    # Çıktı klasörleri ve yardımcılar
    # ------------------------------------------------------------------

    @Slot()
    def open_run_directory(self) -> None:
        """Mevcut/son eğitim klasörünü Finder'da açar."""

        print(f"[TrainingPage {BUILD_ID}] open run pressed", flush=True)
        self._append_log("[BUTON] Eğitim Klasörünü Aç düğmesine basıldı.")

        run_directory = self._resolve_run_directory()
        if run_directory is None:
            output_path = self._normalized_output_directory()
            run_name = self.run_name_input.text().strip() or "pose_training"
            run_directory = output_path / run_name

        try:
            run_directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self._append_log(f"Eğitim klasörü oluşturulamadı: {error}")
            QMessageBox.critical(
                self,
                "Eğitim Klasörü Açılamadı",
                f"Klasör hazırlanamadı:\n{run_directory}\n\n{error}",
            )
            return

        self.epoch_status_label.setText(
            f"Finder'da eğitim klasörü açılıyor: {run_directory}"
        )
        self._append_log(f"Açılacak eğitim klasörü: {run_directory}")
        self._open_local_path(run_directory)

    @Slot()
    def open_weights_directory(self) -> None:
        """Mevcut/son weights klasörünü Finder'da açar."""

        print(f"[TrainingPage {BUILD_ID}] open weights pressed", flush=True)
        self._append_log("[BUTON] Weights Klasörünü Aç düğmesine basıldı.")

        weights_directory = self._resolve_weights_directory()
        if weights_directory is None:
            run_directory = self._resolve_run_directory()
            if run_directory is None:
                output_path = self._normalized_output_directory()
                run_name = self.run_name_input.text().strip() or "pose_training"
                run_directory = output_path / run_name
            weights_directory = run_directory / "weights"

        try:
            weights_directory.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            self._append_log(f"Weights klasörü oluşturulamadı: {error}")
            QMessageBox.critical(
                self,
                "Weights Klasörü Açılamadı",
                f"Klasör hazırlanamadı:\n{weights_directory}\n\n{error}",
            )
            return

        self.epoch_status_label.setText(
            f"Finder'da weights klasörü açılıyor: {weights_directory}"
        )
        self._append_log(f"Açılacak weights klasörü: {weights_directory}")
        self._open_local_path(weights_directory)

    def _normalized_output_directory(self) -> Path:
        raw_output = self.output_directory_input.text().strip() or "runs"
        output_path = Path(raw_output).expanduser()
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        return output_path.resolve()

    @staticmethod
    def _directory_mtime(path: Path) -> float:
        try:
            return path.stat().st_mtime
        except OSError:
            return 0.0

    def _resolve_run_directory(self) -> Path | None:
        candidates: list[Path] = []

        if self.last_training_result is not None:
            candidates.append(self.last_training_result.run_directory)

        checkpoint_text = self.resume_checkpoint_input.text().strip()
        if checkpoint_text:
            checkpoint = Path(checkpoint_text).expanduser()
            if checkpoint.is_file():
                checkpoint = checkpoint.resolve()
                candidates.append(
                    checkpoint.parent.parent
                    if checkpoint.parent.name == "weights"
                    else checkpoint.parent
                )

        output_path = self._normalized_output_directory()
        run_name = self.run_name_input.text().strip()
        if run_name:
            candidates.append(output_path / run_name)

        # Ultralytics aynı isim mevcutsa pose_training2 gibi yeni bir klasör
        # oluşturabilir. Bu yüzden çıktı kökü altındaki gerçek run klasörlerini
        # de tarayıp en yeni olanları aday listesine ekliyoruz.
        discovered_runs: list[Path] = []
        if output_path.is_dir():
            for directory in output_path.rglob("*"):
                if not directory.is_dir():
                    continue
                if directory.name == "weights":
                    discovered_runs.append(directory.parent)
                    continue
                if (directory / "args.yaml").is_file():
                    discovered_runs.append(directory)
                    continue
                if (directory / "results.csv").is_file():
                    discovered_runs.append(directory)

        discovered_runs.sort(
            key=self._directory_mtime,
            reverse=True,
        )
        candidates.extend(discovered_runs)

        # Hiç eğitim klasörü yoksa en azından seçilmiş ana çıktı klasörü açılır.
        candidates.append(output_path)

        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_dir():
                return resolved

        return None

    def _resolve_weights_directory(self) -> Path | None:
        candidates: list[Path] = []

        if self.last_training_result is not None:
            candidates.append(self.last_training_result.weights_directory)

        checkpoint_text = self.resume_checkpoint_input.text().strip()
        if checkpoint_text:
            checkpoint = Path(checkpoint_text).expanduser()
            if checkpoint.is_file():
                checkpoint = checkpoint.resolve()
                candidates.append(
                    checkpoint.parent
                    if checkpoint.parent.name == "weights"
                    else checkpoint.parent / "weights"
                )

        output_path = self._normalized_output_directory()
        run_name = self.run_name_input.text().strip()
        if run_name:
            candidates.append(output_path / run_name / "weights")

        run_directory = self._resolve_run_directory()
        if run_directory is not None:
            candidates.append(run_directory / "weights")

        discovered_weights: list[Path] = []
        if output_path.is_dir():
            for directory in output_path.rglob("weights"):
                if not directory.is_dir():
                    continue
                if any(directory.glob("*.pt")):
                    discovered_weights.append(directory)

        discovered_weights.sort(
            key=self._directory_mtime,
            reverse=True,
        )
        candidates.extend(discovered_weights)

        seen: set[Path] = set()
        for candidate in candidates:
            resolved = candidate.expanduser().resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_dir():
                return resolved

        return None

    def _open_local_path(self, path: Path) -> None:
        """Klasörü macOS Finder'da bloklamadan açar."""

        resolved_path = path.expanduser().resolve()
        if not resolved_path.is_dir():
            self._append_log(f"Klasör bulunamadı: {resolved_path}")
            QMessageBox.warning(
                self,
                "Klasör Bulunamadı",
                f"Klasör bulunamadı:\n{resolved_path}",
            )
            return

        try:
            # run()/wait() kullanılmıyor; GUI thread'i bloklanmadan Finder açılır.
            subprocess.Popen(
                ["/usr/bin/open", str(resolved_path)],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except OSError as error:
            opened = QDesktopServices.openUrl(
                QUrl.fromLocalFile(str(resolved_path))
            )
            if not opened:
                self._append_log(
                    f"Finder açılamadı: {resolved_path} | {error}"
                )
                QMessageBox.critical(
                    self,
                    "Klasör Açılamadı",
                    f"Finder ile açılamadı:\n{resolved_path}\n\n{error}",
                )
                return

        self._append_log(f"Finder açma komutu gönderildi: {resolved_path}")
        self.epoch_status_label.setText(f"Finder'da açıldı: {resolved_path}")

    @Slot(str)
    def _append_log(self, message: str) -> None:
        self.training_log.append(message)
        scroll_bar = self.training_log.verticalScrollBar()
        scroll_bar.setValue(scroll_bar.maximum())

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._is_training_running():
            answer = QMessageBox.question(
                self,
                "Eğitim Devam Ediyor",
                (
                    "Eğitim hâlâ devam ediyor. Mevcut epoch tamamlanıp "
                    "checkpoint kaydedildikten sonra durdurulsun mu?"
                ),
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )

            if answer == QMessageBox.StandardButton.No:
                event.ignore()
                return

            self.training_service.request_stop()
            event.ignore()
            QMessageBox.information(
                self,
                "Durdurma İsteği Gönderildi",
                "Mevcut epoch tamamlanıp last.pt kaydedildikten sonra "
                "uygulamayı kapatabilirsiniz.",
            )
            return

        super().closeEvent(event)
