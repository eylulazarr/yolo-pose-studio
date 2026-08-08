from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ValidationIssue:
    """Tek bir dataset doğrulama hatasını temsil eder."""

    file_path: Path
    message: str
    line_number: int | None = None

    def format_message(self) -> str:
        if self.line_number is None:
            return f"{self.file_path}: {self.message}"

        return (
            f"{self.file_path}, satır {self.line_number}: "
            f"{self.message}"
        )


@dataclass
class PoseValidationResult:
    """Pose dataset doğrulamasının toplu sonucudur."""

    is_valid: bool
    checked_label_count: int
    checked_object_count: int
    valid_label_count: int
    invalid_label_count: int
    empty_label_count: int
    issues: list[ValidationIssue] = field(default_factory=list)


class PoseLabelValidator:
    """
    YOLO Pose label dosyalarını data.yaml bilgilerine göre doğrular.

    Beklenen satır yapısı:

    class_id
    x_center y_center width height
    x1 y1 visibility1
    x2 y2 visibility2
    ...
    """

    def validate_dataset(
        self,
        *,
        data_yaml_path: str,
        labels_directory: str,
    ) -> PoseValidationResult:
        yaml_path = Path(data_yaml_path).expanduser().resolve()
        labels_path = Path(labels_directory).expanduser().resolve()

        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"data.yaml dosyası bulunamadı: {yaml_path}"
            )

        if not labels_path.is_dir():
            raise FileNotFoundError(
                f"Labels klasörü bulunamadı: {labels_path}"
            )

        yaml_data = self._read_yaml(yaml_path)

        class_ids = self._extract_class_ids(yaml_data)
        keypoint_count, keypoint_dimensions = self._extract_kpt_shape(
            yaml_data
        )

        label_files = sorted(
            path
            for path in labels_path.rglob("*.txt")
            if path.is_file()
        )

        issues: list[ValidationIssue] = []
        checked_object_count = 0
        valid_label_count = 0
        invalid_label_count = 0
        empty_label_count = 0

        for label_path in label_files:
            file_issues, object_count, is_empty = self._validate_label_file(
                label_path=label_path,
                class_ids=class_ids,
                keypoint_count=keypoint_count,
                keypoint_dimensions=keypoint_dimensions,
            )

            checked_object_count += object_count

            if is_empty:
                empty_label_count += 1

            if file_issues:
                invalid_label_count += 1
                issues.extend(file_issues)
            else:
                valid_label_count += 1

        return PoseValidationResult(
            is_valid=not issues,
            checked_label_count=len(label_files),
            checked_object_count=checked_object_count,
            valid_label_count=valid_label_count,
            invalid_label_count=invalid_label_count,
            empty_label_count=empty_label_count,
            issues=issues,
        )

    @staticmethod
    def _read_yaml(yaml_path: Path) -> dict[str, Any]:
        try:
            with yaml_path.open("r", encoding="utf-8") as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)
        except yaml.YAMLError as error:
            raise ValueError(
                f"data.yaml okunamadı: {error}"
            ) from error

        if not isinstance(yaml_data, dict):
            raise ValueError(
                "data.yaml geçerli bir YAML sözlüğü değildir."
            )

        return yaml_data

    @staticmethod
    def _extract_class_ids(
        yaml_data: dict[str, Any],
    ) -> set[int]:
        names = yaml_data.get("names")

        if isinstance(names, list):
            return set(range(len(names)))

        if isinstance(names, dict):
            class_ids: set[int] = set()

            for raw_id in names:
                try:
                    class_ids.add(int(raw_id))
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Geçersiz class ID: {raw_id}"
                    ) from error

            return class_ids

        raise ValueError(
            "data.yaml içerisinde geçerli bir 'names' alanı bulunamadı."
        )

    @staticmethod
    def _extract_kpt_shape(
        yaml_data: dict[str, Any],
    ) -> tuple[int, int]:
        kpt_shape = yaml_data.get("kpt_shape")

        if not isinstance(kpt_shape, (list, tuple)):
            raise ValueError(
                "data.yaml içerisinde 'kpt_shape' bulunamadı."
            )

        if len(kpt_shape) != 2:
            raise ValueError(
                "kpt_shape iki değer içermelidir: "
                "[keypoint_sayısı, boyut]."
            )

        try:
            keypoint_count = int(kpt_shape[0])
            keypoint_dimensions = int(kpt_shape[1])
        except (TypeError, ValueError) as error:
            raise ValueError(
                "kpt_shape değerleri tam sayı olmalıdır."
            ) from error

        if keypoint_count <= 0:
            raise ValueError(
                "Keypoint sayısı sıfırdan büyük olmalıdır."
            )

        if keypoint_dimensions not in {2, 3}:
            raise ValueError(
                "YOLO Pose keypoint boyutu 2 veya 3 olmalıdır."
            )

        return keypoint_count, keypoint_dimensions

    def _validate_label_file(
        self,
        *,
        label_path: Path,
        class_ids: set[int],
        keypoint_count: int,
        keypoint_dimensions: int,
    ) -> tuple[list[ValidationIssue], int, bool]:
        issues: list[ValidationIssue] = []

        try:
            raw_text = label_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as error:
            return (
                [
                    ValidationIssue(
                        file_path=label_path,
                        message="Dosya UTF-8 olarak okunamadı.",
                    )
                ],
                0,
                False,
            )

        lines = raw_text.splitlines()

        non_empty_lines = [
            line
            for line in lines
            if line.strip()
        ]

        if not non_empty_lines:
            issues.append(
                ValidationIssue(
                    file_path=label_path,
                    message="Label dosyası boş.",
                )
            )

            return issues, 0, True

        object_count = 0

        expected_value_count = (
            5 + keypoint_count * keypoint_dimensions
        )

        for line_number, raw_line in enumerate(lines, start=1):
            line = raw_line.strip()

            if not line:
                continue

            object_count += 1
            values = line.split()

            if len(values) != expected_value_count:
                issues.append(
                    ValidationIssue(
                        file_path=label_path,
                        line_number=line_number,
                        message=(
                            f"Değer sayısı yanlış. Beklenen "
                            f"{expected_value_count}, bulunan {len(values)}."
                        ),
                    )
                )
                continue

            parsed_values = self._parse_numeric_values(
                label_path=label_path,
                line_number=line_number,
                values=values,
                issues=issues,
            )

            if parsed_values is None:
                continue

            class_id = self._validate_class_id(
                label_path=label_path,
                line_number=line_number,
                raw_value=parsed_values[0],
                class_ids=class_ids,
                issues=issues,
            )

            self._validate_bounding_box(
                label_path=label_path,
                line_number=line_number,
                values=parsed_values[1:5],
                issues=issues,
            )

            self._validate_keypoints(
                label_path=label_path,
                line_number=line_number,
                values=parsed_values[5:],
                keypoint_count=keypoint_count,
                keypoint_dimensions=keypoint_dimensions,
                issues=issues,
            )

            if class_id is None:
                continue

        return issues, object_count, False

    @staticmethod
    def _parse_numeric_values(
        *,
        label_path: Path,
        line_number: int,
        values: list[str],
        issues: list[ValidationIssue],
    ) -> list[float] | None:
        try:
            return [float(value) for value in values]
        except ValueError:
            issues.append(
                ValidationIssue(
                    file_path=label_path,
                    line_number=line_number,
                    message="Satırda sayısal olmayan değer bulunuyor.",
                )
            )

            return None

    @staticmethod
    def _validate_class_id(
        *,
        label_path: Path,
        line_number: int,
        raw_value: float,
        class_ids: set[int],
        issues: list[ValidationIssue],
    ) -> int | None:
        if not raw_value.is_integer():
            issues.append(
                ValidationIssue(
                    file_path=label_path,
                    line_number=line_number,
                    message="Class ID tam sayı olmalıdır.",
                )
            )
            return None

        class_id = int(raw_value)

        if class_id not in class_ids:
            issues.append(
                ValidationIssue(
                    file_path=label_path,
                    line_number=line_number,
                    message=(
                        f"Class ID data.yaml içinde tanımlı değil: "
                        f"{class_id}"
                    ),
                )
            )

        return class_id

    @staticmethod
    def _validate_bounding_box(
        *,
        label_path: Path,
        line_number: int,
        values: list[float],
        issues: list[ValidationIssue],
    ) -> None:
        names = (
            "x_center",
            "y_center",
            "width",
            "height",
        )

        for name, value in zip(names, values):
            if not 0.0 <= value <= 1.0:
                issues.append(
                    ValidationIssue(
                        file_path=label_path,
                        line_number=line_number,
                        message=(
                            f"Bounding box {name} değeri "
                            f"0 ile 1 arasında olmalıdır: {value}"
                        ),
                    )
                )

        width = values[2]
        height = values[3]

        if width <= 0:
            issues.append(
                ValidationIssue(
                    file_path=label_path,
                    line_number=line_number,
                    message="Bounding box width sıfırdan büyük olmalıdır.",
                )
            )

        if height <= 0:
            issues.append(
                ValidationIssue(
                    file_path=label_path,
                    line_number=line_number,
                    message="Bounding box height sıfırdan büyük olmalıdır.",
                )
            )

    @staticmethod
    def _validate_keypoints(
        *,
        label_path: Path,
        line_number: int,
        values: list[float],
        keypoint_count: int,
        keypoint_dimensions: int,
        issues: list[ValidationIssue],
    ) -> None:
        for keypoint_index in range(keypoint_count):
            start_index = keypoint_index * keypoint_dimensions
            keypoint_values = values[
                start_index:start_index + keypoint_dimensions
            ]

            x_value = keypoint_values[0]
            y_value = keypoint_values[1]

            if not 0.0 <= x_value <= 1.0:
                issues.append(
                    ValidationIssue(
                        file_path=label_path,
                        line_number=line_number,
                        message=(
                            f"Keypoint {keypoint_index} x değeri "
                            f"0 ile 1 arasında olmalıdır: {x_value}"
                        ),
                    )
                )

            if not 0.0 <= y_value <= 1.0:
                issues.append(
                    ValidationIssue(
                        file_path=label_path,
                        line_number=line_number,
                        message=(
                            f"Keypoint {keypoint_index} y değeri "
                            f"0 ile 1 arasında olmalıdır: {y_value}"
                        ),
                    )
                )

            if keypoint_dimensions == 3:
                visibility_value = keypoint_values[2]

                if visibility_value not in {0.0, 1.0, 2.0}:
                    issues.append(
                        ValidationIssue(
                            file_path=label_path,
                            line_number=line_number,
                            message=(
                                f"Keypoint {keypoint_index} visibility "
                                f"değeri 0, 1 veya 2 olmalıdır: "
                                f"{visibility_value}"
                            ),
                        )
                    )