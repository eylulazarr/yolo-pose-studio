from __future__ import annotations

import random
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


class SplitterPage(QWidget):
    """YOLO Pose datasetini train, validation ve test olarak böler."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self.setObjectName("splitterPage")

        self.yaml_path_edit: QLineEdit
        self.images_path_edit: QLineEdit
        self.labels_path_edit: QLineEdit
        self.output_path_edit: QLineEdit

        self.train_spin: QSpinBox
        self.val_spin: QSpinBox
        self.test_spin: QSpinBox
        self.seed_spin: QSpinBox

        self.result_text: QTextEdit
        self.progress_bar: QProgressBar
        self.status_label: QLabel

        self.setup_ui()

    # =========================================================
    # UI
    # =========================================================

    def setup_ui(self) -> None:
        page_layout = QVBoxLayout(self)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        scroll_area = QScrollArea()
        scroll_area.setObjectName("splitterScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )

        content = QWidget()
        content.setObjectName("splitterContent")
        content.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.MinimumExpanding,
        )

        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(40, 30, 40, 42)
        content_layout.setSpacing(22)

        content_layout.addWidget(self.create_page_header())
        content_layout.addWidget(self.create_inputs_card())
        content_layout.addWidget(self.create_settings_card())
        content_layout.addWidget(self.create_actions_card())
        content_layout.addWidget(self.create_results_card())

        content_layout.addStretch()

        scroll_area.setWidget(content)
        page_layout.addWidget(scroll_area)

    def create_page_header(self) -> QFrame:
        frame = QFrame()
        frame.setObjectName("pageIntroFrame")

        layout = QVBoxLayout(frame)
        layout.setContentsMargins(2, 0, 2, 0)
        layout.setSpacing(7)

        eyebrow = QLabel("DATASET YÖNETİMİ")
        eyebrow.setObjectName("sectionEyebrow")

        title = QLabel("Dataset Bölme")
        title.setObjectName("pageTitle")

        description = QLabel(
            "YOLO Pose datasetini doğrular ve belirlediğiniz oranlarda "
            "train, validation ve test bölümlerine ayırır. Kaynak dosyalar "
            "değiştirilmeden yeni bir çıktı klasörüne kopyalanır."
        )
        description.setObjectName("pageDescription")
        description.setWordWrap(True)

        layout.addWidget(eyebrow)
        layout.addWidget(title)
        layout.addWidget(description)

        return frame

    def create_inputs_card(self) -> QFrame:
        card = self.create_section_card()

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Dataset Girdileri")
        title.setObjectName("sectionTitle")

        description = QLabel(
            "data.yaml dosyasını, kaynak images ve labels klasörlerini "
            "seçin. Çıktı için kaynak dataset dışında farklı bir klasör kullanın."
        )
        description.setObjectName("sectionDescription")
        description.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addSpacing(4)

        form_grid = QGridLayout()
        form_grid.setContentsMargins(0, 0, 0, 0)
        form_grid.setHorizontalSpacing(12)
        form_grid.setVerticalSpacing(14)

        form_grid.setColumnMinimumWidth(0, 110)
        form_grid.setColumnStretch(0, 0)
        form_grid.setColumnStretch(1, 1)
        form_grid.setColumnStretch(2, 0)

        self.yaml_path_edit = self.create_path_edit(
            "data.yaml dosyasını seçin"
        )
        self.images_path_edit = self.create_path_edit(
            "Kaynak images klasörünü seçin"
        )
        self.labels_path_edit = self.create_path_edit(
            "Kaynak labels klasörünü seçin"
        )
        self.output_path_edit = self.create_path_edit(
            "Çıktının oluşturulacağı ana klasörü seçin"
        )

        yaml_button = self.create_select_button("Dosya Seç")
        images_button = self.create_select_button("Klasör Seç")
        labels_button = self.create_select_button("Klasör Seç")
        output_button = self.create_select_button("Klasör Seç")

        yaml_button.clicked.connect(self.select_yaml_file)
        images_button.clicked.connect(self.select_images_directory)
        labels_button.clicked.connect(self.select_labels_directory)
        output_button.clicked.connect(self.select_output_directory)

        self.add_path_row(
            form_grid,
            row=0,
            label_text="data.yaml:",
            line_edit=self.yaml_path_edit,
            button=yaml_button,
        )
        self.add_path_row(
            form_grid,
            row=1,
            label_text="Images klasörü:",
            line_edit=self.images_path_edit,
            button=images_button,
        )
        self.add_path_row(
            form_grid,
            row=2,
            label_text="Labels klasörü:",
            line_edit=self.labels_path_edit,
            button=labels_button,
        )
        self.add_path_row(
            form_grid,
            row=3,
            label_text="Çıktı klasörü:",
            line_edit=self.output_path_edit,
            button=output_button,
        )

        layout.addLayout(form_grid)

        return card

    def create_settings_card(self) -> QFrame:
        card = self.create_section_card()

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(16)

        title = QLabel("Dataset Bölme Ayarları")
        title.setObjectName("sectionTitle")

        description = QLabel(
            "Train, validation ve test oranlarının toplamı 100 olmalıdır. "
            "Random seed aynı datasetin tekrar aynı şekilde bölünmesini sağlar."
        )
        description.setObjectName("sectionDescription")
        description.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addSpacing(4)

        settings_grid = QGridLayout()
        settings_grid.setContentsMargins(0, 0, 0, 0)
        settings_grid.setHorizontalSpacing(16)
        settings_grid.setVerticalSpacing(8)

        self.train_spin = self.create_ratio_spinbox(70)
        self.val_spin = self.create_ratio_spinbox(15)
        self.test_spin = self.create_ratio_spinbox(15)

        self.seed_spin = QSpinBox()
        self.seed_spin.setRange(0, 999_999)
        self.seed_spin.setValue(42)
        self.seed_spin.setMinimumHeight(44)

        self.add_setting_column(
            settings_grid,
            column=0,
            label_text="Train (%)",
            widget=self.train_spin,
        )
        self.add_setting_column(
            settings_grid,
            column=1,
            label_text="Validation (%)",
            widget=self.val_spin,
        )
        self.add_setting_column(
            settings_grid,
            column=2,
            label_text="Test (%)",
            widget=self.test_spin,
        )
        self.add_setting_column(
            settings_grid,
            column=3,
            label_text="Random Seed",
            widget=self.seed_spin,
        )

        for column in range(4):
            settings_grid.setColumnStretch(column, 1)

        self.ratio_status_label = QLabel()
        self.ratio_status_label.setObjectName("ratioStatusLabel")

        self.train_spin.valueChanged.connect(self.update_ratio_status)
        self.val_spin.valueChanged.connect(self.update_ratio_status)
        self.test_spin.valueChanged.connect(self.update_ratio_status)

        layout.addLayout(settings_grid)
        layout.addWidget(self.ratio_status_label)

        self.update_ratio_status()

        return card

    def create_actions_card(self) -> QFrame:
        card = self.create_section_card()

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(0, 0, 0, 0)
        top_layout.setSpacing(12)

        self.status_label = QLabel("İşlem bekleniyor.")
        self.status_label.setObjectName("operationStatusLabel")
        self.status_label.setWordWrap(True)

        clear_button = QPushButton("Formu Temizle")
        clear_button.setObjectName("secondaryButton")
        clear_button.setMinimumHeight(44)
        clear_button.setCursor(Qt.CursorShape.PointingHandCursor)

        validate_button = QPushButton("Dataseti Doğrula")
        validate_button.setObjectName("secondaryButton")
        validate_button.setMinimumHeight(44)
        validate_button.setCursor(Qt.CursorShape.PointingHandCursor)

        split_button = QPushButton("Dataseti Böl")
        split_button.setObjectName("primaryButton")
        split_button.setMinimumHeight(44)
        split_button.setCursor(Qt.CursorShape.PointingHandCursor)

        clear_button.clicked.connect(self.clear_form)
        validate_button.clicked.connect(self.validate_dataset_clicked)
        split_button.clicked.connect(self.split_dataset_clicked)

        top_layout.addWidget(self.status_label, 1)
        top_layout.addWidget(clear_button)
        top_layout.addWidget(validate_button)
        top_layout.addWidget(split_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setObjectName("splitProgressBar")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setMinimumHeight(18)

        layout.addLayout(top_layout)
        layout.addWidget(self.progress_bar)

        return card

    def create_results_card(self) -> QFrame:
        card = self.create_section_card()

        layout = QVBoxLayout(card)
        layout.setContentsMargins(24, 22, 24, 24)
        layout.setSpacing(14)

        title = QLabel("İşlem Sonuçları")
        title.setObjectName("sectionTitle")

        description = QLabel(
            "Dataset doğrulama ve bölme sonuçları burada gösterilir."
        )
        description.setObjectName("sectionDescription")
        description.setWordWrap(True)

        self.result_text = QTextEdit()
        self.result_text.setObjectName("splitResultText")
        self.result_text.setReadOnly(True)
        self.result_text.setMinimumHeight(240)
        self.result_text.setPlaceholderText(
            "Henüz bir doğrulama veya bölme işlemi yapılmadı."
        )

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(self.result_text)

        return card

    @staticmethod
    def create_section_card() -> QFrame:
        card = QFrame()
        card.setObjectName("sectionCard")
        card.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Minimum,
        )
        return card

    @staticmethod
    def create_path_edit(placeholder: str) -> QLineEdit:
        line_edit = QLineEdit()
        line_edit.setPlaceholderText(placeholder)
        line_edit.setMinimumHeight(44)
        line_edit.setClearButtonEnabled(True)
        line_edit.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Fixed,
        )
        return line_edit

    @staticmethod
    def create_select_button(text: str) -> QPushButton:
        button = QPushButton(text)
        button.setObjectName("pathSelectButton")
        button.setMinimumSize(116, 44)
        button.setMaximumWidth(126)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        return button

    @staticmethod
    def create_ratio_spinbox(value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(0, 100)
        spinbox.setValue(value)
        spinbox.setSuffix(" %")
        spinbox.setMinimumHeight(44)
        return spinbox

    @staticmethod
    def add_path_row(
        grid: QGridLayout,
        row: int,
        label_text: str,
        line_edit: QLineEdit,
        button: QPushButton,
    ) -> None:
        label = QLabel(label_text)
        label.setObjectName("formLabel")
        label.setAlignment(
            Qt.AlignmentFlag.AlignLeft
            | Qt.AlignmentFlag.AlignVCenter
        )
        label.setMinimumWidth(110)

        grid.addWidget(label, row, 0)
        grid.addWidget(line_edit, row, 1)
        grid.addWidget(button, row, 2)

    @staticmethod
    def add_setting_column(
        grid: QGridLayout,
        column: int,
        label_text: str,
        widget: QWidget,
    ) -> None:
        label = QLabel(label_text)
        label.setObjectName("formLabel")

        grid.addWidget(label, 0, column)
        grid.addWidget(widget, 1, column)

    # =========================================================
    # FILE DIALOGS
    # =========================================================

    def select_yaml_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "data.yaml Dosyasını Seç",
            "",
            "YAML Dosyaları (*.yaml *.yml)",
        )

        if not file_path:
            return

        self.yaml_path_edit.setText(file_path)
        self.try_autofill_dataset_paths(Path(file_path))

    def select_images_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Images Klasörünü Seç",
        )

        if directory:
            self.images_path_edit.setText(directory)

    def select_labels_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Labels Klasörünü Seç",
        )

        if directory:
            self.labels_path_edit.setText(directory)

    def select_output_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(
            self,
            "Çıktı Ana Klasörünü Seç",
        )

        if directory:
            self.output_path_edit.setText(directory)

    def try_autofill_dataset_paths(self, yaml_path: Path) -> None:
        dataset_root = yaml_path.parent

        images_candidates = [
            dataset_root / "images",
            dataset_root / "images" / "train",
        ]
        labels_candidates = [
            dataset_root / "labels",
            dataset_root / "labels" / "train",
        ]

        for candidate in images_candidates:
            if candidate.is_dir():
                self.images_path_edit.setText(str(candidate))
                break

        for candidate in labels_candidates:
            if candidate.is_dir():
                self.labels_path_edit.setText(str(candidate))
                break

    # =========================================================
    # VALIDATION
    # =========================================================

    def update_ratio_status(self) -> None:
        total = (
            self.train_spin.value()
            + self.val_spin.value()
            + self.test_spin.value()
        )

        if total == 100:
            self.ratio_status_label.setText(
                "✓ Oranların toplamı 100."
            )
            self.ratio_status_label.setProperty("valid", True)
        else:
            self.ratio_status_label.setText(
                f"⚠ Oranların toplamı {total}. Toplam 100 olmalıdır."
            )
            self.ratio_status_label.setProperty("valid", False)

        self.ratio_status_label.style().unpolish(
            self.ratio_status_label
        )
        self.ratio_status_label.style().polish(
            self.ratio_status_label
        )

    def validate_form(self) -> tuple[
        Path,
        Path,
        Path,
        Path,
        dict[str, Any],
    ]:
        yaml_path = Path(
            self.yaml_path_edit.text().strip()
        ).expanduser()

        images_dir = Path(
            self.images_path_edit.text().strip()
        ).expanduser()

        labels_dir = Path(
            self.labels_path_edit.text().strip()
        ).expanduser()

        output_dir = Path(
            self.output_path_edit.text().strip()
        ).expanduser()

        if not self.yaml_path_edit.text().strip():
            raise ValueError("data.yaml dosyasını seçmelisiniz.")

        if yaml_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                "Dataset yapılandırma dosyası .yaml veya .yml olmalıdır."
            )

        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"data.yaml bulunamadı:\n{yaml_path}"
            )

        if not images_dir.is_dir():
            raise FileNotFoundError(
                f"Images klasörü bulunamadı:\n{images_dir}"
            )

        if not labels_dir.is_dir():
            raise FileNotFoundError(
                f"Labels klasörü bulunamadı:\n{labels_dir}"
            )

        if not self.output_path_edit.text().strip():
            raise ValueError("Çıktı klasörünü seçmelisiniz.")

        ratio_total = (
            self.train_spin.value()
            + self.val_spin.value()
            + self.test_spin.value()
        )

        if ratio_total != 100:
            raise ValueError(
                f"Train, validation ve test oranlarının toplamı "
                f"100 olmalıdır. Mevcut toplam: {ratio_total}"
            )

        if self.train_spin.value() <= 0:
            raise ValueError(
                "Train oranı 0'dan büyük olmalıdır."
            )

        with yaml_path.open(
            "r",
            encoding="utf-8",
        ) as yaml_file:
            yaml_data = yaml.safe_load(yaml_file) or {}

        if not isinstance(yaml_data, dict):
            raise ValueError(
                "data.yaml içeriği geçerli bir YAML sözlüğü değil."
            )

        if "names" not in yaml_data:
            raise ValueError(
                "data.yaml içinde 'names' alanı bulunamadı."
            )

        if "kpt_shape" not in yaml_data:
            raise ValueError(
                "Bu bir YOLO Pose datasetidir. "
                "data.yaml içinde 'kpt_shape' bulunmalıdır."
            )

        return (
            yaml_path,
            images_dir,
            labels_dir,
            output_dir,
            yaml_data,
        )

    def collect_image_label_pairs(
        self,
        images_dir: Path,
        labels_dir: Path,
    ) -> tuple[
        list[tuple[Path, Path, Path]],
        list[Path],
    ]:
        image_files = sorted(
            file_path
            for file_path in images_dir.rglob("*")
            if file_path.is_file()
            and file_path.suffix.lower() in IMAGE_EXTENSIONS
        )

        valid_pairs: list[tuple[Path, Path, Path]] = []
        missing_labels: list[Path] = []

        for image_path in image_files:
            relative_image_path = image_path.relative_to(images_dir)
            relative_label_path = relative_image_path.with_suffix(".txt")
            label_path = labels_dir / relative_label_path

            if label_path.is_file():
                valid_pairs.append(
                    (
                        image_path,
                        label_path,
                        relative_image_path,
                    )
                )
            else:
                missing_labels.append(image_path)

        return valid_pairs, missing_labels

    def validate_pose_labels(
        self,
        pairs: list[tuple[Path, Path, Path]],
        yaml_data: dict[str, Any],
    ) -> list[str]:
        errors: list[str] = []

        kpt_shape = yaml_data.get("kpt_shape")

        if (
            not isinstance(kpt_shape, list)
            or len(kpt_shape) != 2
        ):
            return [
                "kpt_shape değeri [keypoint_sayısı, boyut] "
                "biçiminde olmalıdır."
            ]

        keypoint_count = int(kpt_shape[0])
        keypoint_dimensions = int(kpt_shape[1])

        expected_column_count = (
            1
            + 4
            + keypoint_count * keypoint_dimensions
        )

        for _, label_path, _ in pairs:
            try:
                lines = label_path.read_text(
                    encoding="utf-8"
                ).splitlines()
            except OSError as error:
                errors.append(
                    f"{label_path.name}: okunamadı ({error})"
                )
                continue

            for line_number, line in enumerate(lines, start=1):
                stripped_line = line.strip()

                if not stripped_line:
                    continue

                columns = stripped_line.split()

                if len(columns) != expected_column_count:
                    errors.append(
                        f"{label_path.name}, satır {line_number}: "
                        f"{len(columns)} değer bulundu; "
                        f"{expected_column_count} bekleniyordu."
                    )
                    continue

                try:
                    numeric_values = [
                        float(value)
                        for value in columns
                    ]
                except ValueError:
                    errors.append(
                        f"{label_path.name}, satır {line_number}: "
                        "sayısal olmayan değer içeriyor."
                    )
                    continue

                normalized_values = numeric_values[1:]

                for value in normalized_values:
                    if value < 0:
                        errors.append(
                            f"{label_path.name}, satır {line_number}: "
                            "negatif koordinat bulundu."
                        )
                        break

        return errors

    # =========================================================
    # BUTTON ACTIONS
    # =========================================================

    def validate_dataset_clicked(self) -> None:
        self.progress_bar.setValue(0)
        self.result_text.clear()

        try:
            (
                yaml_path,
                images_dir,
                labels_dir,
                _,
                yaml_data,
            ) = self.validate_form()

            self.progress_bar.setValue(20)

            pairs, missing_labels = self.collect_image_label_pairs(
                images_dir,
                labels_dir,
            )

            self.progress_bar.setValue(55)

            label_errors = self.validate_pose_labels(
                pairs,
                yaml_data,
            )

            self.progress_bar.setValue(100)

            self.result_text.append("DATASET DOĞRULAMA SONUCU")
            self.result_text.append("=" * 54)
            self.result_text.append(
                f"data.yaml: {yaml_path}"
            )
            self.result_text.append(
                f"Images klasörü: {images_dir}"
            )
            self.result_text.append(
                f"Labels klasörü: {labels_dir}"
            )
            self.result_text.append("")
            self.result_text.append(
                f"Eşleşen image-label çifti: {len(pairs)}"
            )
            self.result_text.append(
                f"Label bulunamayan görsel: {len(missing_labels)}"
            )
            self.result_text.append(
                f"Hatalı pose label satırı: {len(label_errors)}"
            )

            if missing_labels:
                self.result_text.append("")
                self.result_text.append(
                    "LABEL BULUNAMAYAN İLK DOSYALAR"
                )

                for image_path in missing_labels[:20]:
                    self.result_text.append(
                        f"- {image_path.name}"
                    )

            if label_errors:
                self.result_text.append("")
                self.result_text.append(
                    "POSE LABEL HATALARI"
                )

                for error in label_errors[:30]:
                    self.result_text.append(f"- {error}")

            if not pairs:
                raise ValueError(
                    "Hiçbir image-label çifti bulunamadı."
                )

            if label_errors:
                self.status_label.setText(
                    "Dataset doğrulandı ancak bazı label hataları bulundu."
                )
            elif missing_labels:
                self.status_label.setText(
                    "Dataset doğrulandı ancak bazı görsellerin label dosyası yok."
                )
            else:
                self.status_label.setText(
                    "Dataset doğrulaması başarıyla tamamlandı."
                )

        except Exception as error:
            self.progress_bar.setValue(0)
            self.status_label.setText("Dataset doğrulanamadı.")
            self.show_error(str(error))

    def split_dataset_clicked(self) -> None:
        self.progress_bar.setValue(0)
        self.result_text.clear()

        try:
            (
                yaml_path,
                images_dir,
                labels_dir,
                output_root,
                yaml_data,
            ) = self.validate_form()

            pairs, missing_labels = self.collect_image_label_pairs(
                images_dir,
                labels_dir,
            )

            if not pairs:
                raise ValueError(
                    "Bölünecek geçerli image-label çifti bulunamadı."
                )

            label_errors = self.validate_pose_labels(
                pairs,
                yaml_data,
            )

            if label_errors:
                raise ValueError(
                    "Dataset içinde hatalı pose label dosyaları var. "
                    "Önce 'Dataseti Doğrula' işlemini çalıştırıp "
                    "hataları düzeltin."
                )

            self.progress_bar.setValue(10)

            shuffled_pairs = list(pairs)

            random_generator = random.Random(
                self.seed_spin.value()
            )
            random_generator.shuffle(shuffled_pairs)

            total_count = len(shuffled_pairs)

            train_count = int(
                total_count
                * self.train_spin.value()
                / 100
            )

            val_count = int(
                total_count
                * self.val_spin.value()
                / 100
            )

            test_count = total_count - train_count - val_count

            if train_count <= 0:
                raise ValueError(
                    "Train bölümüne hiç görüntü düşmüyor. "
                    "Train oranını veya dataset sayısını artırın."
                )

            timestamp = datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

            split_root = (
                output_root
                / f"split_dataset_{timestamp}"
            )

            splits: dict[
                str,
                list[tuple[Path, Path, Path]],
            ] = {
                "train": shuffled_pairs[:train_count],
                "val": shuffled_pairs[
                    train_count:
                    train_count + val_count
                ],
                "test": shuffled_pairs[
                    train_count + val_count:
                ],
            }

            for split_name in splits:
                (split_root / "images" / split_name).mkdir(
                    parents=True,
                    exist_ok=True,
                )
                (split_root / "labels" / split_name).mkdir(
                    parents=True,
                    exist_ok=True,
                )

            copied_count = 0

            for split_name, split_pairs in splits.items():
                for (
                    image_path,
                    label_path,
                    relative_image_path,
                ) in split_pairs:
                    destination_image = (
                        split_root
                        / "images"
                        / split_name
                        / relative_image_path
                    )

                    destination_label = (
                        split_root
                        / "labels"
                        / split_name
                        / relative_image_path.with_suffix(".txt")
                    )

                    destination_image.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    destination_label.parent.mkdir(
                        parents=True,
                        exist_ok=True,
                    )

                    shutil.copy2(
                        image_path,
                        destination_image,
                    )
                    shutil.copy2(
                        label_path,
                        destination_label,
                    )

                    copied_count += 1

                    progress_value = 10 + int(
                        75
                        * copied_count
                        / total_count
                    )
                    self.progress_bar.setValue(
                        progress_value
                    )

            output_yaml_data = dict(yaml_data)

            output_yaml_data["path"] = str(
                split_root.resolve()
            )
            output_yaml_data["train"] = "images/train"
            output_yaml_data["val"] = "images/val"

            if test_count > 0:
                output_yaml_data["test"] = "images/test"
            else:
                output_yaml_data["test"] = ""

            output_yaml_path = split_root / "data.yaml"

            with output_yaml_path.open(
                "w",
                encoding="utf-8",
            ) as output_yaml_file:
                yaml.safe_dump(
                    output_yaml_data,
                    output_yaml_file,
                    allow_unicode=True,
                    sort_keys=False,
                )

            report_path = split_root / "split_report.txt"

            report_lines = [
                "YOLO POSE DATASET SPLIT REPORT",
                "=" * 56,
                f"Kaynak YAML: {yaml_path}",
                f"Kaynak images: {images_dir}",
                f"Kaynak labels: {labels_dir}",
                f"Çıktı: {split_root}",
                "",
                f"Toplam geçerli çift: {total_count}",
                f"Train: {train_count}",
                f"Validation: {val_count}",
                f"Test: {test_count}",
                f"Label bulunamayan görsel: {len(missing_labels)}",
                f"Random seed: {self.seed_spin.value()}",
                "",
                "Kaynak dosyalar değiştirilmemiştir.",
            ]

            report_path.write_text(
                "\n".join(report_lines),
                encoding="utf-8",
            )

            self.progress_bar.setValue(100)

            self.result_text.append("DATASET BÖLME TAMAMLANDI")
            self.result_text.append("=" * 56)
            self.result_text.append(
                f"Çıktı klasörü:\n{split_root}"
            )
            self.result_text.append("")
            self.result_text.append(
                f"Toplam geçerli image-label çifti: {total_count}"
            )
            self.result_text.append(
                f"Train: {train_count}"
            )
            self.result_text.append(
                f"Validation: {val_count}"
            )
            self.result_text.append(
                f"Test: {test_count}"
            )
            self.result_text.append("")
            self.result_text.append(
                f"Yeni data.yaml:\n{output_yaml_path}"
            )
            self.result_text.append(
                f"Rapor:\n{report_path}"
            )

            self.status_label.setText(
                "Dataset başarıyla bölündü."
            )

            QMessageBox.information(
                self,
                "Dataset Bölme Tamamlandı",
                "Dataset başarıyla bölündü.\n\n"
                f"Çıktı:\n{split_root}",
            )

        except Exception as error:
            self.progress_bar.setValue(0)
            self.status_label.setText(
                "Dataset bölme işlemi başarısız oldu."
            )
            self.show_error(str(error))

    def clear_form(self) -> None:
        self.yaml_path_edit.clear()
        self.images_path_edit.clear()
        self.labels_path_edit.clear()
        self.output_path_edit.clear()

        self.train_spin.setValue(70)
        self.val_spin.setValue(15)
        self.test_spin.setValue(15)
        self.seed_spin.setValue(42)

        self.result_text.clear()
        self.progress_bar.setValue(0)
        self.status_label.setText("İşlem bekleniyor.")

    def show_error(self, message: str) -> None:
        self.result_text.append("HATA")
        self.result_text.append("=" * 54)
        self.result_text.append(message)

        QMessageBox.critical(
            self,
            "Dataset Bölme Hatası",
            message,
        )