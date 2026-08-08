from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import yaml


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
class PosePreviewObject:
    """Tek bir YOLO Pose nesnesini temsil eder."""

    class_id: int
    bbox: list[float]
    keypoints: list[list[float]]


@dataclass
class PosePreviewResult:
    """Önizleme üretim sonucunu döndürür."""

    output_directory: Path
    preview_count: int
    skipped_count: int
    preview_image_paths: list[Path]


class PosePreviewService:
    """
    YOLO Pose datasetindeki görsellerin üzerine
    bbox + class name + keypoint çizerek önizleme üretir.
    """

    def generate_preview_dataset(
        self,
        *,
        data_yaml_path: str,
        images_directory: str,
        labels_directory: str,
        output_directory: str,
        max_images: int | None = None,
    ) -> PosePreviewResult:
        yaml_path = Path(data_yaml_path).expanduser().resolve()
        images_path = Path(images_directory).expanduser().resolve()
        labels_path = Path(labels_directory).expanduser().resolve()
        output_parent = Path(output_directory).expanduser().resolve()

        self._validate_inputs(
            yaml_path=yaml_path,
            images_path=images_path,
            labels_path=labels_path,
            output_parent=output_parent,
            max_images=max_images,
        )

        yaml_data = self._read_yaml(yaml_path)
        class_names = self._extract_class_names(yaml_data)
        keypoint_count, keypoint_dimensions = self._extract_kpt_shape(
            yaml_data
        )

        pairs = self._match_images_and_labels(
            images_directory=images_path,
            labels_directory=labels_path,
        )

        if not pairs:
            raise ValueError(
                "Eşleşen hiçbir image-label çifti bulunamadı."
            )

        if max_images is not None:
            pairs = pairs[:max_images]

        preview_output = self._create_output_directory(output_parent)

        preview_paths: list[Path] = []
        skipped_count = 0

        for image_path, label_path in pairs:
            image = cv2.imread(str(image_path))

            if image is None:
                skipped_count += 1
                continue

            try:
                pose_objects = self._read_pose_label(
                    label_path=label_path,
                    keypoint_count=keypoint_count,
                    keypoint_dimensions=keypoint_dimensions,
                )
            except Exception:
                skipped_count += 1
                continue

            preview_image = self._draw_pose_annotations(
                image=image,
                pose_objects=pose_objects,
                class_names=class_names,
            )

            output_path = preview_output / image_path.name

            image_saved = cv2.imwrite(
                str(output_path),
                preview_image,
            )

            if not image_saved:
                skipped_count += 1
                continue

            preview_paths.append(output_path)

        return PosePreviewResult(
            output_directory=preview_output,
            preview_count=len(preview_paths),
            skipped_count=skipped_count,
            preview_image_paths=preview_paths,
        )

    @staticmethod
    def _validate_inputs(
        *,
        yaml_path: Path,
        images_path: Path,
        labels_path: Path,
        output_parent: Path,
        max_images: int | None,
    ) -> None:
        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"data.yaml bulunamadı: {yaml_path}"
            )

        if yaml_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                "data.yaml uzantısı .yaml veya .yml olmalıdır."
            )

        if not images_path.is_dir():
            raise FileNotFoundError(
                f"Images klasörü bulunamadı: {images_path}"
            )

        if not labels_path.is_dir():
            raise FileNotFoundError(
                f"Labels klasörü bulunamadı: {labels_path}"
            )

        if not output_parent.is_dir():
            raise FileNotFoundError(
                f"Çıktı klasörü bulunamadı: {output_parent}"
            )

        if max_images is not None and max_images <= 0:
            raise ValueError(
                "max_images değeri sıfırdan büyük olmalıdır."
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
                "data.yaml geçerli bir YAML sözlüğü değil."
            )

        return yaml_data

    @staticmethod
    def _extract_class_names(
        yaml_data: dict[str, Any],
    ) -> dict[int, str]:
        raw_names = yaml_data.get("names")

        if isinstance(raw_names, list):
            return {
                index: str(name)
                for index, name in enumerate(raw_names)
            }

        if isinstance(raw_names, dict):
            result: dict[int, str] = {}

            for class_id, class_name in raw_names.items():
                result[int(class_id)] = str(class_name)

            return result

        raise ValueError(
            "data.yaml içinde geçerli bir 'names' alanı bulunamadı."
        )

    @staticmethod
    def _extract_kpt_shape(
        yaml_data: dict[str, Any],
    ) -> tuple[int, int]:
        raw_shape = yaml_data.get("kpt_shape")

        if not isinstance(raw_shape, (list, tuple)):
            raise ValueError(
                "data.yaml içinde 'kpt_shape' bulunamadı."
            )

        if len(raw_shape) != 2:
            raise ValueError(
                "kpt_shape biçimi [keypoint_sayısı, boyut] olmalıdır."
            )

        keypoint_count = int(raw_shape[0])
        keypoint_dimensions = int(raw_shape[1])

        if keypoint_count <= 0:
            raise ValueError(
                "Keypoint sayısı sıfırdan büyük olmalıdır."
            )

        if keypoint_dimensions not in {2, 3}:
            raise ValueError(
                "Keypoint boyutu 2 veya 3 olmalıdır."
            )

        return keypoint_count, keypoint_dimensions

    def _match_images_and_labels(
        self,
        *,
        images_directory: Path,
        labels_directory: Path,
    ) -> list[tuple[Path, Path]]:
        image_files = sorted(
            path
            for path in images_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
        )

        label_files = sorted(
            path
            for path in labels_directory.rglob("*.txt")
            if path.is_file()
        )

        image_map = self._create_unique_stem_map(
            paths=image_files,
            file_type="görsel",
        )

        label_map = self._create_unique_stem_map(
            paths=label_files,
            file_type="label",
        )

        common_stems = sorted(
            set(image_map).intersection(label_map)
        )

        return [
            (image_map[stem], label_map[stem])
            for stem in common_stems
        ]

    @staticmethod
    def _create_unique_stem_map(
        *,
        paths: list[Path],
        file_type: str,
    ) -> dict[str, Path]:
        result: dict[str, Path] = {}

        for path in paths:
            normalized_stem = path.stem.lower()

            if normalized_stem in result:
                raise ValueError(
                    f"Aynı ada sahip birden fazla {file_type} bulundu: "
                    f"{normalized_stem}"
                )

            result[normalized_stem] = path

        return result

    @staticmethod
    def _read_pose_label(
        *,
        label_path: Path,
        keypoint_count: int,
        keypoint_dimensions: int,
    ) -> list[PosePreviewObject]:
        pose_objects: list[PosePreviewObject] = []

        expected_value_count = (
            5 + keypoint_count * keypoint_dimensions
        )

        with label_path.open("r", encoding="utf-8") as label_file:
            for line_number, raw_line in enumerate(
                label_file,
                start=1,
            ):
                line = raw_line.strip()

                if not line:
                    continue

                raw_values = line.split()

                if len(raw_values) != expected_value_count:
                    raise ValueError(
                        f"Geçersiz label satırı: {label_path}, "
                        f"satır {line_number}. Beklenen değer sayısı "
                        f"{expected_value_count}, bulunan {len(raw_values)}."
                    )

                values = [float(value) for value in raw_values]

                class_id_value = values[0]

                if not class_id_value.is_integer():
                    raise ValueError(
                        f"Class ID tam sayı olmalı: {label_path}, "
                        f"satır {line_number}."
                    )

                class_id = int(class_id_value)
                bbox = values[1:5]

                keypoint_values = values[5:]
                keypoints: list[list[float]] = []

                for keypoint_index in range(keypoint_count):
                    start_index = (
                        keypoint_index * keypoint_dimensions
                    )

                    point_values = keypoint_values[
                        start_index:
                        start_index + keypoint_dimensions
                    ]

                    if keypoint_dimensions == 2:
                        point_values.append(2.0)

                    keypoints.append(point_values)

                pose_objects.append(
                    PosePreviewObject(
                        class_id=class_id,
                        bbox=bbox,
                        keypoints=keypoints,
                    )
                )

        return pose_objects

    def _draw_pose_annotations(
        self,
        *,
        image,
        pose_objects: list[PosePreviewObject],
        class_names: dict[int, str],
    ):
        preview_image = image.copy()
        image_height, image_width = preview_image.shape[:2]

        for object_index, pose_object in enumerate(pose_objects, start=1):
            x_center, y_center, width, height = pose_object.bbox

            x1 = int((x_center - width / 2.0) * image_width)
            y1 = int((y_center - height / 2.0) * image_height)
            x2 = int((x_center + width / 2.0) * image_width)
            y2 = int((y_center + height / 2.0) * image_height)

            x1 = max(0, min(x1, image_width - 1))
            y1 = max(0, min(y1, image_height - 1))
            x2 = max(0, min(x2, image_width - 1))
            y2 = max(0, min(y2, image_height - 1))

            cv2.rectangle(
                preview_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2,
            )

            class_name = class_names.get(
                pose_object.class_id,
                f"class_{pose_object.class_id}",
            )

            label_text = f"{class_name} #{object_index}"

            cv2.putText(
                preview_image,
                label_text,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            for keypoint_index, keypoint in enumerate(
                pose_object.keypoints
            ):
                x_value, y_value, visibility = keypoint

                if visibility == 0:
                    continue

                px = int(x_value * image_width)
                py = int(y_value * image_height)

                if not (
                    0 <= px < image_width
                    and 0 <= py < image_height
                ):
                    continue

                if visibility == 1:
                    color = (0, 255, 255)  # sarı
                else:
                    color = (0, 0, 255)  # kırmızı

                cv2.circle(
                    preview_image,
                    (px, py),
                    4,
                    color,
                    -1,
                )

                cv2.putText(
                    preview_image,
                    str(keypoint_index),
                    (px + 5, py - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 0, 0),
                    1,
                    cv2.LINE_AA,
                )

        return preview_image

    @staticmethod
    def _create_output_directory(
        output_parent: Path,
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_directory = (
            output_parent / f"pose_preview_{timestamp}"
        )

        counter = 1

        while output_directory.exists():
            output_directory = (
                output_parent
                / f"pose_preview_{timestamp}_{counter}"
            )
            counter += 1

        output_directory.mkdir(parents=True)

        return output_directory