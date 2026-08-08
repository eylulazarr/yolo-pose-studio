from __future__ import annotations

import random
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
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
class PoseObject:
    """
    Bir YOLO Pose nesnesini temsil eder.

    bbox:
        [x_center, y_center, width, height]

    keypoints:
        [
            [x, y, visibility],
            [x, y, visibility],
            ...
        ]
    """

    class_id: int
    bbox: list[float]
    keypoints: list[list[float]]


@dataclass
class AugmentationSettings:
    """Kullanıcının seçtiği augmentation ayarları."""

    copies_per_image: int = 1

    horizontal_flip: bool = True
    flip_probability: float = 0.5

    brightness_contrast: bool = True
    brightness_limit: float = 0.20
    contrast_limit: float = 0.20

    rotation: bool = True
    rotation_limit: float = 10.0

    scale: bool = True
    scale_limit: float = 0.10

    translation: bool = True
    translation_limit: float = 0.05

    blur: bool = False
    blur_probability: float = 0.20

    noise: bool = False
    noise_probability: float = 0.20

    seed: int = 42


@dataclass
class AugmentationResult:
    """Augmentation işleminin toplu sonucudur."""

    output_directory: Path
    data_yaml_path: Path
    report_path: Path

    source_image_count: int
    generated_image_count: int
    skipped_image_count: int
    generated_label_count: int


class AugmentationService:
    """
    YOLO Pose datasetlerine augmentation uygular.

    Desteklenen işlemler:

    - Horizontal flip
    - Brightness / contrast
    - Rotation
    - Scale
    - Translation
    - Gaussian blur
    - Gaussian noise

    Görsel ile birlikte bounding box ve pose keypoint koordinatları
    da geometrik dönüşümlere göre güncellenir.
    """

    def augment_dataset(
        self,
        *,
        data_yaml_path: str,
        images_directory: str,
        labels_directory: str,
        output_directory: str,
        settings: AugmentationSettings,
    ) -> AugmentationResult:
        """Datasetin tamamına augmentation uygular."""

        yaml_path = Path(data_yaml_path).expanduser().resolve()
        images_path = Path(images_directory).expanduser().resolve()
        labels_path = Path(labels_directory).expanduser().resolve()
        output_parent = Path(output_directory).expanduser().resolve()

        self._validate_inputs(
            yaml_path=yaml_path,
            images_path=images_path,
            labels_path=labels_path,
            output_parent=output_parent,
            settings=settings,
        )

        yaml_data = self._read_yaml(yaml_path)

        keypoint_count, keypoint_dimensions = self._extract_kpt_shape(
            yaml_data
        )

        flip_indices = self._extract_flip_indices(
            yaml_data=yaml_data,
            keypoint_count=keypoint_count,
        )

        image_label_pairs = self._match_images_and_labels(
            images_directory=images_path,
            labels_directory=labels_path,
        )

        if not image_label_pairs:
            raise ValueError(
                "Augmentation yapılabilecek eşleşen görsel ve label "
                "dosyası bulunamadı."
            )

        random_generator = random.Random(settings.seed)
        numpy_generator = np.random.default_rng(settings.seed)

        augmentation_output = self._create_output_directory(
            output_parent
        )

        output_images = augmentation_output / "images"
        output_labels = augmentation_output / "labels"

        output_images.mkdir(parents=True, exist_ok=True)
        output_labels.mkdir(parents=True, exist_ok=True)

        generated_image_count = 0
        generated_label_count = 0
        skipped_image_count = 0
        operation_counts: dict[str, int] = {
            "horizontal_flip": 0,
            "brightness_contrast": 0,
            "rotation_scale_translation": 0,
            "blur": 0,
            "noise": 0,
        }

        for image_path, label_path in image_label_pairs:
            image = cv2.imread(str(image_path))

            if image is None:
                skipped_image_count += 1
                continue

            try:
                pose_objects = self._read_pose_label(
                    label_path=label_path,
                    keypoint_count=keypoint_count,
                    keypoint_dimensions=keypoint_dimensions,
                )

            except (ValueError, OSError, UnicodeDecodeError):
                skipped_image_count += 1
                continue

            # Orijinal dosyaları da çıktı datasetine kopyalıyoruz.
            original_image_target = output_images / image_path.name
            original_label_target = (
                output_labels / f"{image_path.stem}.txt"
            )

            shutil.copy2(image_path, original_image_target)
            shutil.copy2(label_path, original_label_target)

            generated_image_count += 1
            generated_label_count += 1

            for copy_index in range(1, settings.copies_per_image + 1):
                augmented_image = image.copy()

                augmented_objects = self._copy_pose_objects(
                    pose_objects
                )

                if (
                    settings.horizontal_flip
                    and random_generator.random()
                    < settings.flip_probability
                ):
                    augmented_image, augmented_objects = (
                        self._apply_horizontal_flip(
                            image=augmented_image,
                            pose_objects=augmented_objects,
                            flip_indices=flip_indices,
                        )
                    )

                    operation_counts["horizontal_flip"] += 1

                should_apply_affine = (
                    settings.rotation
                    or settings.scale
                    or settings.translation
                )

                if should_apply_affine:
                    angle = (
                        random_generator.uniform(
                            -settings.rotation_limit,
                            settings.rotation_limit,
                        )
                        if settings.rotation
                        else 0.0
                    )

                    scale_factor = (
                        random_generator.uniform(
                            1.0 - settings.scale_limit,
                            1.0 + settings.scale_limit,
                        )
                        if settings.scale
                        else 1.0
                    )

                    shift_x_ratio = (
                        random_generator.uniform(
                            -settings.translation_limit,
                            settings.translation_limit,
                        )
                        if settings.translation
                        else 0.0
                    )

                    shift_y_ratio = (
                        random_generator.uniform(
                            -settings.translation_limit,
                            settings.translation_limit,
                        )
                        if settings.translation
                        else 0.0
                    )

                    augmented_image, augmented_objects = (
                        self._apply_affine_transform(
                            image=augmented_image,
                            pose_objects=augmented_objects,
                            angle=angle,
                            scale_factor=scale_factor,
                            shift_x_ratio=shift_x_ratio,
                            shift_y_ratio=shift_y_ratio,
                        )
                    )

                    operation_counts[
                        "rotation_scale_translation"
                    ] += 1

                if settings.brightness_contrast:
                    augmented_image = (
                        self._apply_brightness_contrast(
                            image=augmented_image,
                            random_generator=random_generator,
                            brightness_limit=settings.brightness_limit,
                            contrast_limit=settings.contrast_limit,
                        )
                    )

                    operation_counts["brightness_contrast"] += 1

                if (
                    settings.blur
                    and random_generator.random()
                    < settings.blur_probability
                ):
                    augmented_image = self._apply_blur(
                        augmented_image
                    )

                    operation_counts["blur"] += 1

                if (
                    settings.noise
                    and random_generator.random()
                    < settings.noise_probability
                ):
                    augmented_image = self._apply_noise(
                        image=augmented_image,
                        numpy_generator=numpy_generator,
                    )

                    operation_counts["noise"] += 1

                valid_objects = self._remove_invalid_objects(
                    augmented_objects
                )

                if not valid_objects:
                    skipped_image_count += 1
                    continue

                output_stem = (
                    f"{image_path.stem}_aug_{copy_index:03d}"
                )

                output_image_path = (
                    output_images
                    / f"{output_stem}{image_path.suffix.lower()}"
                )

                output_label_path = (
                    output_labels / f"{output_stem}.txt"
                )

                image_saved = cv2.imwrite(
                    str(output_image_path),
                    augmented_image,
                )

                if not image_saved:
                    skipped_image_count += 1
                    continue

                self._write_pose_label(
                    label_path=output_label_path,
                    pose_objects=valid_objects,
                    keypoint_dimensions=keypoint_dimensions,
                )

                generated_image_count += 1
                generated_label_count += 1

        generated_yaml_path = self._write_output_yaml(
            original_yaml=yaml_data,
            output_directory=augmentation_output,
        )

        report_path = self._write_report(
            output_directory=augmentation_output,
            source_image_count=len(image_label_pairs),
            generated_image_count=generated_image_count,
            generated_label_count=generated_label_count,
            skipped_image_count=skipped_image_count,
            settings=settings,
            operation_counts=operation_counts,
        )

        return AugmentationResult(
            output_directory=augmentation_output,
            data_yaml_path=generated_yaml_path,
            report_path=report_path,
            source_image_count=len(image_label_pairs),
            generated_image_count=generated_image_count,
            skipped_image_count=skipped_image_count,
            generated_label_count=generated_label_count,
        )

    @staticmethod
    def _validate_inputs(
        *,
        yaml_path: Path,
        images_path: Path,
        labels_path: Path,
        output_parent: Path,
        settings: AugmentationSettings,
    ) -> None:
        """Dosya yollarını ve augmentation ayarlarını kontrol eder."""

        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"data.yaml dosyası bulunamadı: {yaml_path}"
            )

        if yaml_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                "Dataset yapılandırma dosyası YAML olmalıdır."
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

        if settings.copies_per_image <= 0:
            raise ValueError(
                "Her görsel için üretilecek kopya sayısı "
                "sıfırdan büyük olmalıdır."
            )

        if not 0.0 <= settings.flip_probability <= 1.0:
            raise ValueError(
                "Flip olasılığı 0 ile 1 arasında olmalıdır."
            )

        if not 0.0 <= settings.brightness_limit <= 1.0:
            raise ValueError(
                "Brightness sınırı 0 ile 1 arasında olmalıdır."
            )

        if not 0.0 <= settings.contrast_limit <= 1.0:
            raise ValueError(
                "Contrast sınırı 0 ile 1 arasında olmalıdır."
            )

        if settings.rotation_limit < 0:
            raise ValueError(
                "Rotation sınırı negatif olamaz."
            )

        if not 0.0 <= settings.scale_limit < 1.0:
            raise ValueError(
                "Scale sınırı 0 ile 1 arasında olmalıdır."
            )

        if not 0.0 <= settings.translation_limit < 1.0:
            raise ValueError(
                "Translation sınırı 0 ile 1 arasında olmalıdır."
            )

    @staticmethod
    def _read_yaml(yaml_path: Path) -> dict[str, Any]:
        """data.yaml dosyasını okur."""

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
    def _extract_kpt_shape(
        yaml_data: dict[str, Any],
    ) -> tuple[int, int]:
        """YAML içinden keypoint sayısını ve boyutunu alır."""

        kpt_shape = yaml_data.get("kpt_shape")

        if not isinstance(kpt_shape, (list, tuple)):
            raise ValueError(
                "data.yaml içinde kpt_shape bulunamadı."
            )

        if len(kpt_shape) != 2:
            raise ValueError(
                "kpt_shape biçimi [keypoint_sayısı, boyut] olmalıdır."
            )

        keypoint_count = int(kpt_shape[0])
        keypoint_dimensions = int(kpt_shape[1])

        if keypoint_count <= 0:
            raise ValueError(
                "Keypoint sayısı sıfırdan büyük olmalıdır."
            )

        if keypoint_dimensions not in {2, 3}:
            raise ValueError(
                "Keypoint boyutu 2 veya 3 olmalıdır."
            )

        return keypoint_count, keypoint_dimensions

    @staticmethod
    def _extract_flip_indices(
        *,
        yaml_data: dict[str, Any],
        keypoint_count: int,
    ) -> list[int]:
        """
        Horizontal flip sırasında sol-sağ keypointlerin nasıl
        değiştirileceğini alır.
        """

        raw_flip_indices = yaml_data.get("flip_idx")

        if raw_flip_indices is None:
            return list(range(keypoint_count))

        if not isinstance(raw_flip_indices, list):
            raise ValueError(
                "flip_idx alanı liste olmalıdır."
            )

        if len(raw_flip_indices) != keypoint_count:
            raise ValueError(
                "flip_idx uzunluğu keypoint sayısına eşit olmalıdır."
            )

        flip_indices = [
            int(index)
            for index in raw_flip_indices
        ]

        expected_indices = set(range(keypoint_count))

        if set(flip_indices) != expected_indices:
            raise ValueError(
                "flip_idx bütün keypoint indekslerini bir kez "
                "içermelidir."
            )

        return flip_indices

    def _match_images_and_labels(
        self,
        *,
        images_directory: Path,
        labels_directory: Path,
    ) -> list[tuple[Path, Path]]:
        """Görsel ve label dosyalarını dosya adına göre eşleştirir."""

        image_files = sorted(
            path
            for path in images_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_IMAGE_EXTENSIONS
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
            (
                image_map[stem],
                label_map[stem],
            )
            for stem in common_stems
        ]

    @staticmethod
    def _create_unique_stem_map(
        *,
        paths: list[Path],
        file_type: str,
    ) -> dict[str, Path]:
        """Dosyaları uzantısız isimlerine göre sözlüğe dönüştürür."""

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
    ) -> list[PoseObject]:
        """YOLO Pose label dosyasını nesne listesine dönüştürür."""

        pose_objects: list[PoseObject] = []

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
                        f"Geçersiz değer sayısı: {label_path}, "
                        f"satır {line_number}. "
                        f"Beklenen {expected_value_count}, "
                        f"bulunan {len(raw_values)}."
                    )

                try:
                    values = [
                        float(value)
                        for value in raw_values
                    ]

                except ValueError as error:
                    raise ValueError(
                        f"Sayısal olmayan değer bulundu: "
                        f"{label_path}, satır {line_number}."
                    ) from error

                class_id_value = values[0]

                if not class_id_value.is_integer():
                    raise ValueError(
                        f"Class ID tam sayı olmalıdır: "
                        f"{label_path}, satır {line_number}."
                    )

                class_id = int(class_id_value)
                bbox = values[1:5]

                keypoint_values = values[5:]
                keypoints: list[list[float]] = []

                for keypoint_index in range(keypoint_count):
                    start = (
                        keypoint_index * keypoint_dimensions
                    )

                    point_values = keypoint_values[
                        start:
                        start + keypoint_dimensions
                    ]

                    if keypoint_dimensions == 2:
                        point_values.append(2.0)

                    keypoints.append(point_values)

                pose_objects.append(
                    PoseObject(
                        class_id=class_id,
                        bbox=bbox,
                        keypoints=keypoints,
                    )
                )

        return pose_objects

    @staticmethod
    def _copy_pose_objects(
        pose_objects: list[PoseObject],
    ) -> list[PoseObject]:
        """Pose nesnelerinin bağımsız kopyasını oluşturur."""

        return [
            PoseObject(
                class_id=pose_object.class_id,
                bbox=pose_object.bbox.copy(),
                keypoints=[
                    keypoint.copy()
                    for keypoint in pose_object.keypoints
                ],
            )
            for pose_object in pose_objects
        ]

    @staticmethod
    def _apply_horizontal_flip(
        *,
        image: np.ndarray,
        pose_objects: list[PoseObject],
        flip_indices: list[int],
    ) -> tuple[np.ndarray, list[PoseObject]]:
        """Görsel, bbox ve keypointleri yatay olarak çevirir."""

        flipped_image = cv2.flip(image, 1)

        for pose_object in pose_objects:
            x_center, y_center, width, height = pose_object.bbox

            pose_object.bbox = [
                1.0 - x_center,
                y_center,
                width,
                height,
            ]

            flipped_keypoints: list[list[float]] = []

            for keypoint in pose_object.keypoints:
                x_value, y_value, visibility = keypoint

                if visibility == 0:
                    flipped_keypoints.append(
                        [0.0, 0.0, visibility]
                    )
                else:
                    flipped_keypoints.append(
                        [
                            1.0 - x_value,
                            y_value,
                            visibility,
                        ]
                    )

            pose_object.keypoints = [
                flipped_keypoints[index]
                for index in flip_indices
            ]

        return flipped_image, pose_objects

    def _apply_affine_transform(
        self,
        *,
        image: np.ndarray,
        pose_objects: list[PoseObject],
        angle: float,
        scale_factor: float,
        shift_x_ratio: float,
        shift_y_ratio: float,
    ) -> tuple[np.ndarray, list[PoseObject]]:
        """
        Görsel, bbox ve keypointlere rotation, scale ve translation
        dönüşümlerini birlikte uygular.
        """

        image_height, image_width = image.shape[:2]

        center = (
            image_width / 2.0,
            image_height / 2.0,
        )

        matrix = cv2.getRotationMatrix2D(
            center,
            angle,
            scale_factor,
        )

        matrix[0, 2] += shift_x_ratio * image_width
        matrix[1, 2] += shift_y_ratio * image_height

        transformed_image = cv2.warpAffine(
            image,
            matrix,
            (image_width, image_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(114, 114, 114),
        )

        transformed_objects: list[PoseObject] = []

        for pose_object in pose_objects:
            transformed_bbox = self._transform_bbox(
                bbox=pose_object.bbox,
                matrix=matrix,
                image_width=image_width,
                image_height=image_height,
            )

            transformed_keypoints = self._transform_keypoints(
                keypoints=pose_object.keypoints,
                matrix=matrix,
                image_width=image_width,
                image_height=image_height,
            )

            transformed_objects.append(
                PoseObject(
                    class_id=pose_object.class_id,
                    bbox=transformed_bbox,
                    keypoints=transformed_keypoints,
                )
            )

        return transformed_image, transformed_objects

    @staticmethod
    def _transform_bbox(
        *,
        bbox: list[float],
        matrix: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> list[float]:
        """YOLO bbox koordinatlarını affine dönüşüme göre günceller."""

        x_center, y_center, width, height = bbox

        x1 = (x_center - width / 2.0) * image_width
        y1 = (y_center - height / 2.0) * image_height
        x2 = (x_center + width / 2.0) * image_width
        y2 = (y_center + height / 2.0) * image_height

        corners = np.array(
            [
                [x1, y1, 1.0],
                [x2, y1, 1.0],
                [x2, y2, 1.0],
                [x1, y2, 1.0],
            ],
            dtype=np.float32,
        )

        transformed_corners = corners @ matrix.T

        transformed_x = transformed_corners[:, 0]
        transformed_y = transformed_corners[:, 1]

        new_x1 = float(np.clip(
            transformed_x.min(),
            0,
            image_width,
        ))

        new_y1 = float(np.clip(
            transformed_y.min(),
            0,
            image_height,
        ))

        new_x2 = float(np.clip(
            transformed_x.max(),
            0,
            image_width,
        ))

        new_y2 = float(np.clip(
            transformed_y.max(),
            0,
            image_height,
        ))

        new_width = max(0.0, new_x2 - new_x1)
        new_height = max(0.0, new_y2 - new_y1)

        new_x_center = (new_x1 + new_x2) / 2.0
        new_y_center = (new_y1 + new_y2) / 2.0

        return [
            new_x_center / image_width,
            new_y_center / image_height,
            new_width / image_width,
            new_height / image_height,
        ]

    @staticmethod
    def _transform_keypoints(
        *,
        keypoints: list[list[float]],
        matrix: np.ndarray,
        image_width: int,
        image_height: int,
    ) -> list[list[float]]:
        """Keypoint koordinatlarını affine dönüşüme göre günceller."""

        transformed_keypoints: list[list[float]] = []

        for keypoint in keypoints:
            x_value, y_value, visibility = keypoint

            if visibility == 0:
                transformed_keypoints.append(
                    [0.0, 0.0, 0.0]
                )
                continue

            pixel_point = np.array(
                [
                    x_value * image_width,
                    y_value * image_height,
                    1.0,
                ],
                dtype=np.float32,
            )

            transformed_point = matrix @ pixel_point

            transformed_x = float(transformed_point[0])
            transformed_y = float(transformed_point[1])

            is_inside = (
                0.0 <= transformed_x < image_width
                and 0.0 <= transformed_y < image_height
            )

            if not is_inside:
                transformed_keypoints.append(
                    [0.0, 0.0, 0.0]
                )
                continue

            transformed_keypoints.append(
                [
                    transformed_x / image_width,
                    transformed_y / image_height,
                    visibility,
                ]
            )

        return transformed_keypoints

    @staticmethod
    def _apply_brightness_contrast(
        *,
        image: np.ndarray,
        random_generator: random.Random,
        brightness_limit: float,
        contrast_limit: float,
    ) -> np.ndarray:
        """Rastgele parlaklık ve kontrast değişimi uygular."""

        contrast_factor = random_generator.uniform(
            1.0 - contrast_limit,
            1.0 + contrast_limit,
        )

        brightness_shift = random_generator.uniform(
            -brightness_limit * 255.0,
            brightness_limit * 255.0,
        )

        result = (
            image.astype(np.float32) * contrast_factor
            + brightness_shift
        )

        return np.clip(
            result,
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def _apply_blur(image: np.ndarray) -> np.ndarray:
        """Görsele hafif Gaussian blur uygular."""

        return cv2.GaussianBlur(
            image,
            (5, 5),
            sigmaX=0,
        )

    @staticmethod
    def _apply_noise(
        *,
        image: np.ndarray,
        numpy_generator: np.random.Generator,
    ) -> np.ndarray:
        """Görsele hafif Gaussian noise ekler."""

        noise = numpy_generator.normal(
            loc=0.0,
            scale=8.0,
            size=image.shape,
        )

        noisy_image = (
            image.astype(np.float32)
            + noise.astype(np.float32)
        )

        return np.clip(
            noisy_image,
            0,
            255,
        ).astype(np.uint8)

    @staticmethod
    def _remove_invalid_objects(
        pose_objects: list[PoseObject],
    ) -> list[PoseObject]:
        """Geçersiz veya görüntü dışına çıkan nesneleri kaldırır."""

        valid_objects: list[PoseObject] = []

        for pose_object in pose_objects:
            x_center, y_center, width, height = pose_object.bbox

            if width <= 0.001 or height <= 0.001:
                continue

            if not 0.0 <= x_center <= 1.0:
                continue

            if not 0.0 <= y_center <= 1.0:
                continue

            valid_objects.append(pose_object)

        return valid_objects

    @staticmethod
    def _write_pose_label(
        *,
        label_path: Path,
        pose_objects: list[PoseObject],
        keypoint_dimensions: int,
    ) -> None:
        """Pose nesnelerini YOLO Pose formatında kaydeder."""

        lines: list[str] = []

        for pose_object in pose_objects:
            values: list[str] = [
                str(pose_object.class_id),
                *[
                    f"{value:.6f}"
                    for value in pose_object.bbox
                ],
            ]

            for keypoint in pose_object.keypoints:
                x_value, y_value, visibility = keypoint

                values.append(f"{x_value:.6f}")
                values.append(f"{y_value:.6f}")

                if keypoint_dimensions == 3:
                    values.append(str(int(visibility)))

            lines.append(" ".join(values))

        label_path.write_text(
            "\n".join(lines) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _create_output_directory(
        output_parent: Path,
    ) -> Path:
        """Her işlem için ayrı augmentation klasörü oluşturur."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_directory = (
            output_parent / f"augmented_dataset_{timestamp}"
        )

        counter = 1

        while output_directory.exists():
            output_directory = (
                output_parent
                / f"augmented_dataset_{timestamp}_{counter}"
            )
            counter += 1

        output_directory.mkdir(parents=True)

        return output_directory

    @staticmethod
    def _write_output_yaml(
        *,
        original_yaml: dict[str, Any],
        output_directory: Path,
    ) -> Path:
        """Augmented dataset için yeni data.yaml oluşturur."""

        generated_yaml = original_yaml.copy()

        generated_yaml["path"] = str(output_directory)
        generated_yaml["train"] = "images"
        generated_yaml["val"] = "images"
        generated_yaml["test"] = "images"

        output_yaml_path = output_directory / "data.yaml"

        with output_yaml_path.open(
            "w",
            encoding="utf-8",
        ) as yaml_file:
            yaml.safe_dump(
                generated_yaml,
                yaml_file,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )

        return output_yaml_path

    @staticmethod
    def _write_report(
        *,
        output_directory: Path,
        source_image_count: int,
        generated_image_count: int,
        generated_label_count: int,
        skipped_image_count: int,
        settings: AugmentationSettings,
        operation_counts: dict[str, int],
    ) -> Path:
        """Augmentation işlem raporunu oluşturur."""

        report_path = output_directory / "augmentation_report.txt"

        lines = [
            "=" * 76,
            "YOLO POSE DATA AUGMENTATION RAPORU",
            "=" * 76,
            "",
            "GENEL SONUÇLAR",
            "-" * 76,
            f"Kaynak eşleşen görsel sayısı: {source_image_count}",
            f"Üretilen toplam görsel sayısı: {generated_image_count}",
            f"Üretilen toplam label sayısı: {generated_label_count}",
            f"Atlanan görsel sayısı: {skipped_image_count}",
            "",
            "AYARLAR",
            "-" * 76,
            (
                "Her görsel için augmentation kopyası: "
                f"{settings.copies_per_image}"
            ),
            f"Random seed: {settings.seed}",
            (
                "Horizontal flip: "
                f"{'Açık' if settings.horizontal_flip else 'Kapalı'}"
            ),
            (
                "Brightness/contrast: "
                f"{'Açık' if settings.brightness_contrast else 'Kapalı'}"
            ),
            (
                "Rotation: "
                f"{'Açık' if settings.rotation else 'Kapalı'}"
            ),
            (
                "Scale: "
                f"{'Açık' if settings.scale else 'Kapalı'}"
            ),
            (
                "Translation: "
                f"{'Açık' if settings.translation else 'Kapalı'}"
            ),
            (
                "Blur: "
                f"{'Açık' if settings.blur else 'Kapalı'}"
            ),
            (
                "Noise: "
                f"{'Açık' if settings.noise else 'Kapalı'}"
            ),
            "",
            "UYGULANAN İŞLEM SAYILARI",
            "-" * 76,
            (
                "Horizontal flip: "
                f"{operation_counts['horizontal_flip']}"
            ),
            (
                "Brightness/contrast: "
                f"{operation_counts['brightness_contrast']}"
            ),
            (
                "Rotation/scale/translation: "
                f"{operation_counts['rotation_scale_translation']}"
            ),
            f"Blur: {operation_counts['blur']}",
            f"Noise: {operation_counts['noise']}",
            "",
            "=" * 76,
            "RAPOR SONU",
            "=" * 76,
        ]

        report_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return report_path