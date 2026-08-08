from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from services.augmentation_service import (
    AugmentationService,
    AugmentationSettings,
)


class AugmentationPage(QWidget):
    """
    YOLO Pose datasetleri için Data Augmentation sayfası.

    Bu sayfa üzerinden:

    - data.yaml seçilir.
    - images ve labels klasörleri seçilir.
    - augmentation ayarları yapılır.
    - yeni augmented dataset oluşturulur.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.augmentation_service = AugmentationService()

        # Dosya ve klasör alanları
        self.data_yaml_input = QLineEdit()
        self.images_input = QLineEdit()
        self.labels_input = QLineEdit()
        self.output_input = QLineEdit()

        # Genel ayarlar
        self.copies_spin = QSpinBox()
        self.seed_spin = QSpinBox()

        # Horizontal flip ayarları
        self.flip_checkbox = QCheckBox("Horizontal Flip")
        self.flip_probability_spin = QDoubleSpinBox()

        # Brightness ve contrast ayarları
        self.brightness_checkbox = QCheckBox(
            "Brightness / Contrast"
        )
        self.brightness_limit_spin = QDoubleSpinBox()
        self.contrast_limit_spin = QDoubleSpinBox()

        # Rotation ayarları
        self.rotation_checkbox = QCheckBox("Rotation")
        self.rotation_limit_spin = QDoubleSpinBox()

        # Scale ayarları
        self.scale_checkbox = QCheckBox("Scale")
        self.scale_limit_spin = QDoubleSpinBox()

        # Translation ayarları
        self.translation_checkbox = QCheckBox("Translation")
        self.translation_limit_spin = QDoubleSpinBox()

        # Blur ayarları
        self.blur_checkbox = QCheckBox("Gaussian Blur")
        self.blur_probability_spin = QDoubleSpinBox()

        # Noise ayarları
        self.noise_checkbox = QCheckBox("Gaussian Noise")
        self.noise_probability_spin = QDoubleSpinBox()

        # Log alanı
        self.log_output = QTextEdit()

        # İşlem butonu
        self.augment_button = QPushButton(
            "Augmentation Başlat"
        )

        self.setup_ui()
        self.configure_inputs()
        self.connect_signals()
        self.update_enabled_states()

    def setup_ui(self) -> None:
        """Sayfanın kaydırılabilir ve responsive arayüzünü oluşturur."""

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        content_widget = QWidget()
        content_widget.setObjectName("augmentationContent")

        main_layout = QVBoxLayout(content_widget)
        main_layout.setContentsMargins(40, 35, 40, 40)
        main_layout.setSpacing(18)

        title_label = QLabel("Data Augmentation")
        title_label.setObjectName("pageTitle")

        description_label = QLabel(
            "YOLO Pose datasetine görüntü ve koordinat uyumlu "
            "augmentation işlemleri uygular. Görseller değiştirildiğinde "
            "bounding box ve keypoint koordinatları da otomatik olarak "
            "güncellenir."
        )
        description_label.setObjectName("pageDescription")
        description_label.setWordWrap(True)

        main_layout.addWidget(title_label)
        main_layout.addWidget(description_label)
        main_layout.addWidget(self.create_input_card())
        main_layout.addWidget(self.create_general_settings_card())
        main_layout.addWidget(self.create_augmentation_settings_card())
        main_layout.addLayout(self.create_action_buttons())
        main_layout.addWidget(self.create_log_card())
        main_layout.addStretch()

        scroll_area.setWidget(content_widget)
        root_layout.addWidget(scroll_area)


    def create_input_card(self) -> QFrame:
        """Dataset dosya ve klasör seçim kartını oluşturur."""

        card = QFrame()
        card.setObjectName("formCard")
        card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(15)

        title_label = QLabel("Dataset Girdileri")
        title_label.setObjectName("sectionTitle")

        info_label = QLabel(
            "Augmentation uygulanacak data.yaml, images ve labels "
            "klasörlerini seçin. Oluşturulan yeni dataset çıktı "
            "klasörünün altında ayrı bir klasöre kaydedilecektir."
        )
        info_label.setObjectName("inputLabel")
        info_label.setWordWrap(True)

        form_layout = QFormLayout()
        form_layout.setHorizontalSpacing(20)
        form_layout.setVerticalSpacing(14)
        form_layout.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
        )
        form_layout.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow
        )

        form_layout.addRow(
            "data.yaml:",
            self.create_path_row(
                line_edit=self.data_yaml_input,
                button_text="Dosya Seç",
                callback=self.select_data_yaml,
            ),
        )

        form_layout.addRow(
            "Images klasörü:",
            self.create_path_row(
                line_edit=self.images_input,
                button_text="Klasör Seç",
                callback=self.select_images_folder,
            ),
        )

        form_layout.addRow(
            "Labels klasörü:",
            self.create_path_row(
                line_edit=self.labels_input,
                button_text="Klasör Seç",
                callback=self.select_labels_folder,
            ),
        )

        form_layout.addRow(
            "Çıktı klasörü:",
            self.create_path_row(
                line_edit=self.output_input,
                button_text="Klasör Seç",
                callback=self.select_output_folder,
            ),
        )

        layout.addWidget(title_label)
        layout.addWidget(info_label)
        layout.addLayout(form_layout)

        return card

    def create_general_settings_card(self) -> QFrame:
        """Kopya sayısı ve random seed ayarlarını oluşturur."""

        card = QFrame()
        card.setObjectName("formCard")
        card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(15)

        title_label = QLabel("Genel Ayarlar")
        title_label.setObjectName("sectionTitle")

        info_label = QLabel(
            "Kopya sayısı, her kaynak görselden kaç yeni augmented "
            "görsel üretileceğini belirler. Orijinal görseller de çıktı "
            "datasetine eklenir."
        )
        info_label.setObjectName("inputLabel")
        info_label.setWordWrap(True)

        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(20)

        settings_layout.addWidget(
            self.create_control_group(
                title="Her görsel için kopya",
                widget=self.copies_spin,
            )
        )

        settings_layout.addWidget(
            self.create_control_group(
                title="Random Seed",
                widget=self.seed_spin,
            )
        )

        settings_layout.addStretch()

        layout.addWidget(title_label)
        layout.addWidget(info_label)
        layout.addLayout(settings_layout)

        return card

    def create_augmentation_settings_card(self) -> QFrame:
        """Bütün augmentation seçeneklerini oluşturur."""

        card = QFrame()
        card.setObjectName("formCard")
        card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 26)
        layout.setSpacing(18)

        title_label = QLabel("Augmentation İşlemleri")
        title_label.setObjectName("sectionTitle")

        info_label = QLabel(
            "Aktif etmek istediğiniz işlemleri seçin. Geometrik "
            "işlemlerde görsel ile birlikte bounding box ve pose "
            "keypoint koordinatları da dönüştürülür."
        )
        info_label.setObjectName("inputLabel")
        info_label.setWordWrap(True)

        grid_layout = QGridLayout()
        grid_layout.setContentsMargins(0, 6, 0, 0)
        grid_layout.setHorizontalSpacing(18)
        grid_layout.setVerticalSpacing(16)
        grid_layout.setColumnStretch(0, 1)
        grid_layout.setColumnStretch(1, 1)

        grid_layout.addWidget(
            self.create_flip_group(),
            0,
            0,
        )

        grid_layout.addWidget(
            self.create_brightness_group(),
            0,
            1,
        )

        grid_layout.addWidget(
            self.create_rotation_group(),
            1,
            0,
        )

        grid_layout.addWidget(
            self.create_scale_group(),
            1,
            1,
        )

        grid_layout.addWidget(
            self.create_translation_group(),
            2,
            0,
        )

        grid_layout.addWidget(
            self.create_blur_group(),
            2,
            1,
        )

        grid_layout.addWidget(
            self.create_noise_group(),
            3,
            0,
        )

        layout.addWidget(title_label)
        layout.addWidget(info_label)
        layout.addLayout(grid_layout)

        return card

    def create_flip_group(self) -> QFrame:
        """Horizontal flip ayar grubunu oluşturur."""

        frame = self.create_setting_frame()

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        probability_row = self.create_labeled_row(
            label_text="Uygulama olasılığı:",
            widget=self.flip_probability_spin,
        )

        layout.addWidget(self.flip_checkbox)
        layout.addWidget(probability_row)

        return frame

    def create_brightness_group(self) -> QFrame:
        """Brightness ve contrast ayar grubunu oluşturur."""

        frame = self.create_setting_frame()

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        brightness_row = self.create_labeled_row(
            label_text="Brightness sınırı:",
            widget=self.brightness_limit_spin,
        )

        contrast_row = self.create_labeled_row(
            label_text="Contrast sınırı:",
            widget=self.contrast_limit_spin,
        )

        layout.addWidget(self.brightness_checkbox)
        layout.addWidget(brightness_row)
        layout.addWidget(contrast_row)

        return frame

    def create_rotation_group(self) -> QFrame:
        """Rotation ayar grubunu oluşturur."""

        frame = self.create_setting_frame()

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        rotation_row = self.create_labeled_row(
            label_text="Maksimum açı:",
            widget=self.rotation_limit_spin,
        )

        layout.addWidget(self.rotation_checkbox)
        layout.addWidget(rotation_row)

        return frame

    def create_scale_group(self) -> QFrame:
        """Scale ayar grubunu oluşturur."""

        frame = self.create_setting_frame()

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        scale_row = self.create_labeled_row(
            label_text="Scale sınırı:",
            widget=self.scale_limit_spin,
        )

        layout.addWidget(self.scale_checkbox)
        layout.addWidget(scale_row)

        return frame

    def create_translation_group(self) -> QFrame:
        """Translation ayar grubunu oluşturur."""

        frame = self.create_setting_frame()

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        translation_row = self.create_labeled_row(
            label_text="Kaydırma sınırı:",
            widget=self.translation_limit_spin,
        )

        layout.addWidget(self.translation_checkbox)
        layout.addWidget(translation_row)

        return frame

    def create_blur_group(self) -> QFrame:
        """Blur ayar grubunu oluşturur."""

        frame = self.create_setting_frame()

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        probability_row = self.create_labeled_row(
            label_text="Uygulama olasılığı:",
            widget=self.blur_probability_spin,
        )

        layout.addWidget(self.blur_checkbox)
        layout.addWidget(probability_row)

        return frame

    def create_noise_group(self) -> QFrame:
        """Noise ayar grubunu oluşturur."""

        frame = self.create_setting_frame()

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(10)

        probability_row = self.create_labeled_row(
            label_text="Uygulama olasılığı:",
            widget=self.noise_probability_spin,
        )

        layout.addWidget(self.noise_checkbox)
        layout.addWidget(probability_row)

        return frame

    def create_action_buttons(self) -> QHBoxLayout:
        """Temizleme, doğrulama ve başlatma butonlarını oluşturur."""

        layout = QHBoxLayout()
        layout.setSpacing(12)

        clear_button = QPushButton("Formu Temizle")
        clear_button.setObjectName("dangerButton")
        clear_button.setMinimumHeight(42)
        clear_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        clear_button.clicked.connect(self.clear_form)

        validate_button = QPushButton("Ayarları Doğrula")
        validate_button.setObjectName("secondaryButton")
        validate_button.setMinimumHeight(42)
        validate_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        validate_button.clicked.connect(
            self.validate_form_with_message
        )

        self.augment_button.setObjectName("primaryButton")
        self.augment_button.setMinimumHeight(42)
        self.augment_button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        self.augment_button.clicked.connect(
            self.start_augmentation
        )

        layout.addStretch()
        layout.addWidget(clear_button)
        layout.addWidget(validate_button)
        layout.addWidget(self.augment_button)

        return layout

    def create_log_card(self) -> QFrame:
        """İşlem sonuçları alanını oluşturur."""

        card = QFrame()
        card.setObjectName("formCard")
        card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(12)

        title_label = QLabel("İşlem Sonuçları")
        title_label.setObjectName("sectionTitle")

        self.log_output.setReadOnly(True)
        self.log_output.setMinimumHeight(220)
        self.log_output.setMaximumHeight(320)
        self.log_output.setPlaceholderText(
            "Augmentation doğrulama ve işlem sonuçları burada "
            "gösterilecektir."
        )

        layout.addWidget(title_label)
        layout.addWidget(self.log_output)

        return card

    def configure_inputs(self) -> None:
        """Bütün giriş alanlarının başlangıç ayarlarını yapar."""

        path_inputs = [
            self.data_yaml_input,
            self.images_input,
            self.labels_input,
            self.output_input,
        ]

        for line_edit in path_inputs:
            line_edit.setReadOnly(False)
            line_edit.setClearButtonEnabled(True)
            line_edit.setMinimumHeight(40)
            line_edit.setSizePolicy(
                QSizePolicy.Policy.Expanding,
                QSizePolicy.Policy.Fixed,
            )

        self.data_yaml_input.setPlaceholderText(
            "data.yaml dosyasını seçin veya yolunu yazın"
        )
        self.images_input.setPlaceholderText(
            "images klasörünü seçin veya yolunu yazın"
        )
        self.labels_input.setPlaceholderText(
            "labels klasörünü seçin veya yolunu yazın"
        )
        self.output_input.setPlaceholderText(
            "Çıktı klasörünü seçin veya yolunu yazın"
        )

        self.copies_spin.setRange(1, 20)
        self.copies_spin.setValue(1)
        self.copies_spin.setMinimumHeight(40)
        self.copies_spin.setMinimumWidth(150)

        self.seed_spin.setRange(0, 999999)
        self.seed_spin.setValue(42)
        self.seed_spin.setMinimumHeight(40)
        self.seed_spin.setMinimumWidth(150)

        self.flip_checkbox.setChecked(True)
        self.flip_probability_spin.setRange(0.0, 1.0)
        self.flip_probability_spin.setSingleStep(0.05)
        self.flip_probability_spin.setDecimals(2)
        self.flip_probability_spin.setValue(0.50)

        self.brightness_checkbox.setChecked(True)

        self.brightness_limit_spin.setRange(0.0, 1.0)
        self.brightness_limit_spin.setSingleStep(0.05)
        self.brightness_limit_spin.setDecimals(2)
        self.brightness_limit_spin.setValue(0.20)

        self.contrast_limit_spin.setRange(0.0, 1.0)
        self.contrast_limit_spin.setSingleStep(0.05)
        self.contrast_limit_spin.setDecimals(2)
        self.contrast_limit_spin.setValue(0.20)

        self.rotation_checkbox.setChecked(True)
        self.rotation_limit_spin.setRange(0.0, 180.0)
        self.rotation_limit_spin.setSingleStep(1.0)
        self.rotation_limit_spin.setDecimals(1)
        self.rotation_limit_spin.setValue(10.0)
        self.rotation_limit_spin.setSuffix("°")

        self.scale_checkbox.setChecked(True)
        self.scale_limit_spin.setRange(0.0, 0.95)
        self.scale_limit_spin.setSingleStep(0.05)
        self.scale_limit_spin.setDecimals(2)
        self.scale_limit_spin.setValue(0.10)

        self.translation_checkbox.setChecked(True)
        self.translation_limit_spin.setRange(0.0, 0.95)
        self.translation_limit_spin.setSingleStep(0.01)
        self.translation_limit_spin.setDecimals(2)
        self.translation_limit_spin.setValue(0.05)

        self.blur_checkbox.setChecked(False)
        self.blur_probability_spin.setRange(0.0, 1.0)
        self.blur_probability_spin.setSingleStep(0.05)
        self.blur_probability_spin.setDecimals(2)
        self.blur_probability_spin.setValue(0.20)

        self.noise_checkbox.setChecked(False)
        self.noise_probability_spin.setRange(0.0, 1.0)
        self.noise_probability_spin.setSingleStep(0.05)
        self.noise_probability_spin.setDecimals(2)
        self.noise_probability_spin.setValue(0.20)

        double_spinboxes = [
            self.flip_probability_spin,
            self.brightness_limit_spin,
            self.contrast_limit_spin,
            self.rotation_limit_spin,
            self.scale_limit_spin,
            self.translation_limit_spin,
            self.blur_probability_spin,
            self.noise_probability_spin,
        ]

        for spinbox in double_spinboxes:
            spinbox.setMinimumHeight(38)
            spinbox.setMinimumWidth(130)

    def connect_signals(self) -> None:
        """Checkbox durumlarını ilgili ayarlara bağlar."""

        self.flip_checkbox.toggled.connect(
            self.update_enabled_states
        )
        self.brightness_checkbox.toggled.connect(
            self.update_enabled_states
        )
        self.rotation_checkbox.toggled.connect(
            self.update_enabled_states
        )
        self.scale_checkbox.toggled.connect(
            self.update_enabled_states
        )
        self.translation_checkbox.toggled.connect(
            self.update_enabled_states
        )
        self.blur_checkbox.toggled.connect(
            self.update_enabled_states
        )
        self.noise_checkbox.toggled.connect(
            self.update_enabled_states
        )

    def update_enabled_states(self) -> None:
        """Kapalı augmentation ayarlarının alanlarını pasifleştirir."""

        self.flip_probability_spin.setEnabled(
            self.flip_checkbox.isChecked()
        )

        brightness_enabled = (
            self.brightness_checkbox.isChecked()
        )

        self.brightness_limit_spin.setEnabled(
            brightness_enabled
        )
        self.contrast_limit_spin.setEnabled(
            brightness_enabled
        )

        self.rotation_limit_spin.setEnabled(
            self.rotation_checkbox.isChecked()
        )

        self.scale_limit_spin.setEnabled(
            self.scale_checkbox.isChecked()
        )

        self.translation_limit_spin.setEnabled(
            self.translation_checkbox.isChecked()
        )

        self.blur_probability_spin.setEnabled(
            self.blur_checkbox.isChecked()
        )

        self.noise_probability_spin.setEnabled(
            self.noise_checkbox.isChecked()
        )

    @staticmethod
    def create_setting_frame() -> QFrame:
        """Augmentation ayarları için ortak kutu oluşturur."""

        frame = QFrame()
        frame.setObjectName("settingFrame")
        frame.setMinimumHeight(128)
        frame.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )

        return frame

    @staticmethod
    def create_control_group(
        *,
        title: str,
        widget: QWidget,
    ) -> QWidget:
        """Başlık ve form kontrolünü dikey biçimde oluşturur."""

        container = QWidget()
        container.setMinimumWidth(220)

        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        label = QLabel(title)
        label.setObjectName("inputLabel")

        layout.addWidget(label)
        layout.addWidget(widget)

        return container

    @staticmethod
    def create_labeled_row(
        *,
        label_text: str,
        widget: QWidget,
    ) -> QWidget:
        """Ayar başlığı ve kontrol alanını yan yana oluşturur."""

        container = QWidget()

        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        label = QLabel(label_text)
        label.setObjectName("inputLabel")

        layout.addWidget(label)
        layout.addStretch()
        layout.addWidget(widget)

        return container

    @staticmethod
    def create_path_row(
        *,
        line_edit: QLineEdit,
        button_text: str,
        callback: Callable[[], None],
    ) -> QWidget:
        """Dosya yolu alanı ve seçim butonu oluşturur."""

        container = QWidget()

        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(10)

        button = QPushButton(button_text)
        button.setMinimumWidth(115)
        button.setMinimumHeight(40)
        button.setCursor(
            Qt.CursorShape.PointingHandCursor
        )
        button.clicked.connect(callback)

        layout.addWidget(line_edit, stretch=1)
        layout.addWidget(button)

        return container

    def select_data_yaml(self) -> None:
        """Yalnızca data.yaml dosyasını seçer."""

        current_path = self.data_yaml_input.text().strip()

        if current_path:
            current_directory = str(
                Path(current_path).expanduser().parent
            )
        else:
            current_directory = str(Path.home())

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "data.yaml Dosyasını Seç",
            current_directory,
            "YAML Dosyaları (*.yaml *.yml);;Tüm Dosyalar (*)",
        )

        if not file_path:
            return

        self.data_yaml_input.setText(file_path)
        self.append_log(f"data.yaml seçildi: {file_path}")

    def select_images_folder(self) -> None:
        """Kullanıcının istediği images klasörünü seçmesini sağlar."""

        current_path = self.images_input.text().strip()
        start_directory = (
            current_path
            if current_path and Path(current_path).is_dir()
            else str(Path.home())
        )

        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Images Klasörünü Seç",
            start_directory,
            QFileDialog.Option.ShowDirsOnly,
        )

        if not folder_path:
            return

        self.images_input.setText(folder_path)
        self.append_log(
            f"Images klasörü seçildi: {folder_path}"
        )

    def select_labels_folder(self) -> None:
        """Kullanıcının istediği labels klasörünü seçmesini sağlar."""

        current_path = self.labels_input.text().strip()
        start_directory = (
            current_path
            if current_path and Path(current_path).is_dir()
            else str(Path.home())
        )

        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Labels Klasörünü Seç",
            start_directory,
            QFileDialog.Option.ShowDirsOnly,
        )

        if not folder_path:
            return

        self.labels_input.setText(folder_path)
        self.append_log(
            f"Labels klasörü seçildi: {folder_path}"
        )

    def select_output_folder(self) -> None:
        """Kullanıcının istediği çıktı klasörünü seçmesini sağlar."""

        current_path = self.output_input.text().strip()
        start_directory = (
            current_path
            if current_path and Path(current_path).is_dir()
            else str(Path.home())
        )

        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Çıktı Klasörünü Seç",
            start_directory,
            QFileDialog.Option.ShowDirsOnly,
        )

        if not folder_path:
            return

        self.output_input.setText(folder_path)
        self.append_log(
            f"Çıktı klasörü seçildi: {folder_path}"
        )

    def validate_form(self) -> tuple[bool, list[str]]:
        """
        Form alanlarını ve augmentation seçeneklerini kontrol eder.

        Dönüş:
            (geçerli_mi, hata_listesi)
        """

        errors: list[str] = []

        data_yaml_path = self.data_yaml_input.text().strip()
        images_path = self.images_input.text().strip()
        labels_path = self.labels_input.text().strip()
        output_path = self.output_input.text().strip()

        if not data_yaml_path:
            errors.append(
                "data.yaml dosyası seçilmedi."
            )
        elif not Path(data_yaml_path).is_file():
            errors.append(
                "Seçilen data.yaml dosyası bulunamadı."
            )
        elif Path(data_yaml_path).suffix.lower() not in {
            ".yaml",
            ".yml",
        }:
            errors.append(
                "Dataset yapılandırma dosyası YAML olmalıdır."
            )

        if not images_path:
            errors.append(
                "Images klasörü seçilmedi."
            )
        elif not Path(images_path).is_dir():
            errors.append(
                "Seçilen images klasörü bulunamadı."
            )

        if not labels_path:
            errors.append(
                "Labels klasörü seçilmedi."
            )
        elif not Path(labels_path).is_dir():
            errors.append(
                "Seçilen labels klasörü bulunamadı."
            )

        if not output_path:
            errors.append(
                "Çıktı klasörü seçilmedi."
            )
        elif not Path(output_path).is_dir():
            errors.append(
                "Seçilen çıktı klasörü bulunamadı."
            )

        if images_path and output_path:
            if (
                Path(images_path).resolve()
                == Path(output_path).resolve()
            ):
                errors.append(
                    "Çıktı klasörü images klasörüyle aynı olamaz."
                )

        if labels_path and output_path:
            if (
                Path(labels_path).resolve()
                == Path(output_path).resolve()
            ):
                errors.append(
                    "Çıktı klasörü labels klasörüyle aynı olamaz."
                )

        enabled_operation_count = sum(
            [
                self.flip_checkbox.isChecked(),
                self.brightness_checkbox.isChecked(),
                self.rotation_checkbox.isChecked(),
                self.scale_checkbox.isChecked(),
                self.translation_checkbox.isChecked(),
                self.blur_checkbox.isChecked(),
                self.noise_checkbox.isChecked(),
            ]
        )

        if enabled_operation_count == 0:
            errors.append(
                "En az bir augmentation işlemi seçilmelidir."
            )

        return not errors, errors

    def validate_form_with_message(self) -> None:
        """Doğrulama sonucunu kullanıcıya gösterir."""

        is_valid, errors = self.validate_form()

        self.append_log("")
        self.append_log("=" * 60)
        self.append_log("Augmentation ayarları doğrulanıyor.")
        self.append_log("=" * 60)

        if not is_valid:
            for error in errors:
                self.append_log(
                    f"HATA: {error}"
                )

            QMessageBox.warning(
                self,
                "Doğrulama Hatası",
                "\n".join(errors),
            )
            return

        settings = self.create_settings()

        self.append_log(
            "Dosya yolları ve augmentation ayarları geçerli."
        )
        self.append_log(
            f"Her görsel için üretilecek kopya: "
            f"{settings.copies_per_image}"
        )
        self.append_log(
            f"Random seed: {settings.seed}"
        )

        QMessageBox.information(
            self,
            "Doğrulama Başarılı",
            (
                "Dosya yolları ve augmentation ayarları "
                "başarıyla doğrulandı."
            ),
        )

    def create_settings(self) -> AugmentationSettings:
        """Arayüzdeki değerlerden servis ayar nesnesini oluşturur."""

        return AugmentationSettings(
            copies_per_image=self.copies_spin.value(),

            horizontal_flip=self.flip_checkbox.isChecked(),
            flip_probability=(
                self.flip_probability_spin.value()
            ),

            brightness_contrast=(
                self.brightness_checkbox.isChecked()
            ),
            brightness_limit=(
                self.brightness_limit_spin.value()
            ),
            contrast_limit=(
                self.contrast_limit_spin.value()
            ),

            rotation=self.rotation_checkbox.isChecked(),
            rotation_limit=(
                self.rotation_limit_spin.value()
            ),

            scale=self.scale_checkbox.isChecked(),
            scale_limit=self.scale_limit_spin.value(),

            translation=(
                self.translation_checkbox.isChecked()
            ),
            translation_limit=(
                self.translation_limit_spin.value()
            ),

            blur=self.blur_checkbox.isChecked(),
            blur_probability=(
                self.blur_probability_spin.value()
            ),

            noise=self.noise_checkbox.isChecked(),
            noise_probability=(
                self.noise_probability_spin.value()
            ),

            seed=self.seed_spin.value(),
        )

    def start_augmentation(self) -> None:
        """Augmentation servisini çalıştırır."""

        is_valid, errors = self.validate_form()

        if not is_valid:
            self.append_log("")
            self.append_log("=" * 60)
            self.append_log(
                "Augmentation başlatılamadı."
            )
            self.append_log("=" * 60)

            for error in errors:
                self.append_log(
                    f"HATA: {error}"
                )

            QMessageBox.warning(
                self,
                "Doğrulama Hatası",
                "\n".join(errors),
            )
            return

        settings = self.create_settings()

        self.append_log("")
        self.append_log("=" * 60)
        self.append_log(
            "YOLO Pose Data Augmentation başlatıldı."
        )
        self.append_log("=" * 60)

        self.append_log(
            f"Her görsel için yeni kopya: "
            f"{settings.copies_per_image}"
        )
        self.append_log(
            f"Random seed: {settings.seed}"
        )

        self.log_selected_operations(settings)

        self.augment_button.setEnabled(False)
        self.augment_button.setText("İşlem Yapılıyor...")

        try:
            result = (
                self.augmentation_service.augment_dataset(
                    data_yaml_path=(
                        self.data_yaml_input.text().strip()
                    ),
                    images_directory=(
                        self.images_input.text().strip()
                    ),
                    labels_directory=(
                        self.labels_input.text().strip()
                    ),
                    output_directory=(
                        self.output_input.text().strip()
                    ),
                    settings=settings,
                )
            )

        except (
            FileNotFoundError,
            PermissionError,
            ValueError,
            OSError,
        ) as error:
            self.append_log("")
            self.append_log(
                f"HATA: {error}"
            )

            QMessageBox.critical(
                self,
                "Augmentation Hatası",
                str(error),
            )

            return

        except Exception as error:
            self.append_log("")
            self.append_log(
                f"BEKLENMEYEN HATA: {error}"
            )

            QMessageBox.critical(
                self,
                "Beklenmeyen Hata",
                (
                    "Augmentation sırasında beklenmeyen "
                    "bir hata oluştu.\n\n"
                    f"{error}"
                ),
            )

            return

        finally:
            self.augment_button.setEnabled(True)
            self.augment_button.setText(
                "Augmentation Başlat"
            )

        self.append_log("")
        self.append_log("=" * 60)
        self.append_log(
            "Augmentation başarıyla tamamlandı."
        )
        self.append_log("=" * 60)

        self.append_log(
            f"Kaynak eşleşen görsel: "
            f"{result.source_image_count}"
        )
        self.append_log(
            f"Üretilen toplam görsel: "
            f"{result.generated_image_count}"
        )
        self.append_log(
            f"Üretilen toplam label: "
            f"{result.generated_label_count}"
        )
        self.append_log(
            f"Atlanan görsel: "
            f"{result.skipped_image_count}"
        )

        self.append_log("")
        self.append_log(
            f"Çıktı klasörü: "
            f"{result.output_directory}"
        )
        self.append_log(
            f"Yeni data.yaml: "
            f"{result.data_yaml_path}"
        )
        self.append_log(
            f"Augmentation raporu: "
            f"{result.report_path}"
        )

        QMessageBox.information(
            self,
            "Augmentation Tamamlandı",
            (
                "Dataset augmentation işlemi tamamlandı.\n\n"
                f"Kaynak görsel: "
                f"{result.source_image_count}\n"
                f"Toplam çıktı görseli: "
                f"{result.generated_image_count}\n"
                f"Atlanan görsel: "
                f"{result.skipped_image_count}\n\n"
                f"Çıktı klasörü:\n"
                f"{result.output_directory}"
            ),
        )

    def log_selected_operations(
        self,
        settings: AugmentationSettings,
    ) -> None:
        """Seçilen augmentation ayarlarını log alanına yazar."""

        self.append_log("")
        self.append_log("Aktif işlemler:")

        if settings.horizontal_flip:
            self.append_log(
                f"- Horizontal Flip "
                f"(olasılık: {settings.flip_probability:.2f})"
            )

        if settings.brightness_contrast:
            self.append_log(
                "- Brightness / Contrast "
                f"(brightness: ±{settings.brightness_limit:.2f}, "
                f"contrast: ±{settings.contrast_limit:.2f})"
            )

        if settings.rotation:
            self.append_log(
                f"- Rotation "
                f"(±{settings.rotation_limit:.1f} derece)"
            )

        if settings.scale:
            self.append_log(
                f"- Scale "
                f"(±{settings.scale_limit:.2f})"
            )

        if settings.translation:
            self.append_log(
                f"- Translation "
                f"(±{settings.translation_limit:.2f})"
            )

        if settings.blur:
            self.append_log(
                f"- Gaussian Blur "
                f"(olasılık: {settings.blur_probability:.2f})"
            )

        if settings.noise:
            self.append_log(
                f"- Gaussian Noise "
                f"(olasılık: {settings.noise_probability:.2f})"
            )

    def clear_form(self) -> None:
        """Formu varsayılan değerlere döndürür."""

        self.data_yaml_input.clear()
        self.images_input.clear()
        self.labels_input.clear()
        self.output_input.clear()

        self.copies_spin.setValue(1)
        self.seed_spin.setValue(42)

        self.flip_checkbox.setChecked(True)
        self.flip_probability_spin.setValue(0.50)

        self.brightness_checkbox.setChecked(True)
        self.brightness_limit_spin.setValue(0.20)
        self.contrast_limit_spin.setValue(0.20)

        self.rotation_checkbox.setChecked(True)
        self.rotation_limit_spin.setValue(10.0)

        self.scale_checkbox.setChecked(True)
        self.scale_limit_spin.setValue(0.10)

        self.translation_checkbox.setChecked(True)
        self.translation_limit_spin.setValue(0.05)

        self.blur_checkbox.setChecked(False)
        self.blur_probability_spin.setValue(0.20)

        self.noise_checkbox.setChecked(False)
        self.noise_probability_spin.setValue(0.20)

        self.log_output.clear()
        self.append_log(
            "Augmentation formu temizlendi."
        )

    def append_log(self, message: str) -> None:
        """İşlem sonuçları alanına mesaj ekler."""

        self.log_output.append(message)