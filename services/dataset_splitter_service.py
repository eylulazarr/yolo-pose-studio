from __future__ import annotations

import random
import shutil
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

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
class DatasetPair:
    """Bir görsel ve ona ait YOLO label dosyasını temsil eder."""

    image_path: Path
    label_path: Path


@dataclass
class KeypointVisibilityAnalysis:
    """
    Tek bir keypoint için görünürlük sayılarını tutar.

    YOLO Pose visibility değerleri:
    0 = etiketlenmemiş veya eksik
    1 = etiketli fakat görünmüyor
    2 = görünür
    """

    missing: int = 0
    hidden: int = 0
    visible: int = 0


@dataclass
class SectionAnalysis:
    """Train, validation veya test bölümünün analiz sonucudur."""

    image_count: int
    label_count: int
    matched_count: int
    object_count: int
    class_counts: Counter[int] = field(default_factory=Counter)
    keypoint_visibility: dict[int, KeypointVisibilityAnalysis] = field(
        default_factory=dict
    )


@dataclass
class SplitResult:
    """Dataset bölme işleminden arayüze dönen sonuç."""

    output_directory: Path
    data_yaml_path: Path
    analysis_report_path: Path

    train_count: int
    val_count: int
    test_count: int

    missing_label_images: list[Path]
    missing_image_labels: list[Path]


class DatasetSplitterService:
    """
    YOLO Pose datasetini train, validation ve test bölümlerine ayırır.

    Arayüz kodu içermez. Bütün dosya ve analiz işlemleri bu servis
    içerisinde gerçekleştirilir.
    """

    def split_dataset(
        self,
        *,
        data_yaml_path: str,
        images_directory: str,
        labels_directory: str,
        output_directory: str,
        train_ratio: int,
        val_ratio: int,
        test_ratio: int,
        seed: int,
    ) -> SplitResult:
        """
        Dataseti verilen oranlara göre böler ve analiz raporu üretir.
        """

        yaml_path = Path(data_yaml_path).expanduser().resolve()
        images_path = Path(images_directory).expanduser().resolve()
        labels_path = Path(labels_directory).expanduser().resolve()
        output_parent = Path(output_directory).expanduser().resolve()

        self._validate_inputs(
            yaml_path=yaml_path,
            images_path=images_path,
            labels_path=labels_path,
            output_parent=output_parent,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        yaml_data = self._read_yaml(yaml_path)

        class_names = self._extract_class_names(yaml_data)

        keypoint_count, keypoint_dimensions = self._extract_kpt_shape(
            yaml_data
        )

        keypoint_names = self._extract_keypoint_names(
            yaml_data=yaml_data,
            keypoint_count=keypoint_count,
        )

        pairs, missing_labels, missing_images = self._match_files(
            images_directory=images_path,
            labels_directory=labels_path,
        )

        if not pairs:
            raise ValueError(
                "Eşleşen hiçbir görsel ve label dosyası bulunamadı."
            )

        shuffled_pairs = pairs.copy()

        random_generator = random.Random(seed)
        random_generator.shuffle(shuffled_pairs)

        train_pairs, val_pairs, test_pairs = self._calculate_sections(
            pairs=shuffled_pairs,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        dataset_output = self._create_output_directory(output_parent)

        self._create_dataset_folders(dataset_output)

        self._copy_section(
            pairs=train_pairs,
            section_name="train",
            output_directory=dataset_output,
        )

        self._copy_section(
            pairs=val_pairs,
            section_name="val",
            output_directory=dataset_output,
        )

        self._copy_section(
            pairs=test_pairs,
            section_name="test",
            output_directory=dataset_output,
        )

        generated_yaml_path = self._write_output_yaml(
            original_yaml=yaml_data,
            output_directory=dataset_output,
        )

        train_analysis = self._analyze_section(
            pairs=train_pairs,
            keypoint_count=keypoint_count,
            keypoint_dimensions=keypoint_dimensions,
        )

        val_analysis = self._analyze_section(
            pairs=val_pairs,
            keypoint_count=keypoint_count,
            keypoint_dimensions=keypoint_dimensions,
        )

        test_analysis = self._analyze_section(
            pairs=test_pairs,
            keypoint_count=keypoint_count,
            keypoint_dimensions=keypoint_dimensions,
        )

        report_path = self._write_analysis_report(
            output_directory=dataset_output,
            class_names=class_names,
            keypoint_names=keypoint_names,
            keypoint_dimensions=keypoint_dimensions,
            train_analysis=train_analysis,
            val_analysis=val_analysis,
            test_analysis=test_analysis,
            missing_label_images=missing_labels,
            missing_image_labels=missing_images,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            seed=seed,
        )

        return SplitResult(
            output_directory=dataset_output,
            data_yaml_path=generated_yaml_path,
            analysis_report_path=report_path,
            train_count=len(train_pairs),
            val_count=len(val_pairs),
            test_count=len(test_pairs),
            missing_label_images=missing_labels,
            missing_image_labels=missing_images,
        )

    @staticmethod
    def _validate_inputs(
        *,
        yaml_path: Path,
        images_path: Path,
        labels_path: Path,
        output_parent: Path,
        train_ratio: int,
        val_ratio: int,
        test_ratio: int,
    ) -> None:
        """Dosya yollarını ve oranları kontrol eder."""

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

        ratio_total = train_ratio + val_ratio + test_ratio

        if ratio_total != 100:
            raise ValueError(
                "Train, validation ve test oranlarının toplamı "
                f"100 olmalıdır. Mevcut toplam: {ratio_total}"
            )

        if train_ratio <= 0:
            raise ValueError(
                "Train oranı sıfırdan büyük olmalıdır."
            )

        if val_ratio < 0:
            raise ValueError(
                "Validation oranı negatif olamaz."
            )

        if test_ratio < 0:
            raise ValueError(
                "Test oranı negatif olamaz."
            )

    @staticmethod
    def _read_yaml(yaml_path: Path) -> dict[str, Any]:
        """data.yaml dosyasını güvenli şekilde okur."""

        try:
            with yaml_path.open("r", encoding="utf-8") as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)

        except yaml.YAMLError as error:
            raise ValueError(
                f"data.yaml okunamadı: {error}"
            ) from error

        if not isinstance(yaml_data, dict):
            raise ValueError(
                "data.yaml içeriği geçerli bir YAML sözlüğü değildir."
            )

        if "names" not in yaml_data:
            raise ValueError(
                "data.yaml içerisinde 'names' alanı bulunamadı."
            )

        if "kpt_shape" not in yaml_data:
            raise ValueError(
                "data.yaml içerisinde 'kpt_shape' alanı bulunamadı."
            )

        return yaml_data

    @staticmethod
    def _extract_class_names(
        yaml_data: dict[str, Any],
    ) -> dict[int, str]:
        """Sınıf isimlerini standart sözlük yapısına çevirir."""

        raw_names = yaml_data.get("names")

        if isinstance(raw_names, list):
            return {
                index: str(class_name)
                for index, class_name in enumerate(raw_names)
            }

        if isinstance(raw_names, dict):
            class_names: dict[int, str] = {}

            for class_id, class_name in raw_names.items():
                try:
                    normalized_id = int(class_id)

                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"Geçersiz sınıf ID değeri: {class_id}"
                    ) from error

                class_names[normalized_id] = str(class_name)

            return class_names

        raise ValueError(
            "data.yaml içindeki 'names' alanı liste veya sözlük olmalıdır."
        )

    @staticmethod
    def _extract_kpt_shape(
        yaml_data: dict[str, Any],
    ) -> tuple[int, int]:
        """Keypoint sayısını ve keypoint boyutunu YAML dosyasından alır."""

        raw_shape = yaml_data.get("kpt_shape")

        if not isinstance(raw_shape, (list, tuple)):
            raise ValueError(
                "kpt_shape alanı liste olmalıdır."
            )

        if len(raw_shape) != 2:
            raise ValueError(
                "kpt_shape biçimi [keypoint_sayısı, boyut] olmalıdır."
            )

        try:
            keypoint_count = int(raw_shape[0])
            keypoint_dimensions = int(raw_shape[1])

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
                "Keypoint boyutu 2 veya 3 olmalıdır."
            )

        return keypoint_count, keypoint_dimensions

    @staticmethod
    def _extract_keypoint_names(
        *,
        yaml_data: dict[str, Any],
        keypoint_count: int,
    ) -> dict[int, str]:
        """
        data.yaml içerisinden keypoint isimlerini alır.

        keypoint_names alanı yoksa otomatik isim üretir:
        Keypoint 0, Keypoint 1...
        """

        raw_names = yaml_data.get("keypoint_names")

        if raw_names is None:
            raw_names = yaml_data.get("kpt_names")

        if isinstance(raw_names, list):
            result: dict[int, str] = {}

            for index in range(keypoint_count):
                if index < len(raw_names):
                    result[index] = str(raw_names[index])
                else:
                    result[index] = f"Keypoint {index}"

            return result

        if isinstance(raw_names, dict):
            result = {}

            for index in range(keypoint_count):
                name = raw_names.get(index)

                if name is None:
                    name = raw_names.get(str(index))

                result[index] = (
                    str(name)
                    if name is not None
                    else f"Keypoint {index}"
                )

            return result

        return {
            index: f"Keypoint {index}"
            for index in range(keypoint_count)
        }

    def _match_files(
        self,
        *,
        images_directory: Path,
        labels_directory: Path,
    ) -> tuple[list[DatasetPair], list[Path], list[Path]]:
        """Görsel ve label dosyalarını uzantısız isimlerine göre eşleştirir."""

        image_files = self._find_image_files(images_directory)

        label_files = sorted(
            path
            for path in labels_directory.rglob("*.txt")
            if path.is_file()
        )

        image_by_stem = self._create_unique_stem_map(
            paths=image_files,
            file_type="görsel",
        )

        label_by_stem = self._create_unique_stem_map(
            paths=label_files,
            file_type="label",
        )

        matched_stems = sorted(
            set(image_by_stem).intersection(label_by_stem)
        )

        pairs = [
            DatasetPair(
                image_path=image_by_stem[stem],
                label_path=label_by_stem[stem],
            )
            for stem in matched_stems
        ]

        missing_labels = [
            image_by_stem[stem]
            for stem in sorted(
                set(image_by_stem).difference(label_by_stem)
            )
        ]

        missing_images = [
            label_by_stem[stem]
            for stem in sorted(
                set(label_by_stem).difference(image_by_stem)
            )
        ]

        return pairs, missing_labels, missing_images

    @staticmethod
    def _find_image_files(
        images_directory: Path,
    ) -> list[Path]:
        """Desteklenen bütün görsel dosyalarını bulur."""

        return sorted(
            path
            for path in images_directory.rglob("*")
            if (
                path.is_file()
                and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
        )

    @staticmethod
    def _create_unique_stem_map(
        *,
        paths: list[Path],
        file_type: str,
    ) -> dict[str, Path]:
        """Dosyaları küçük harfli uzantısız isimlerine göre eşler."""

        result: dict[str, Path] = {}
        duplicate_stems: set[str] = set()

        for path in paths:
            normalized_stem = path.stem.lower()

            if normalized_stem in result:
                duplicate_stems.add(normalized_stem)
                continue

            result[normalized_stem] = path

        if duplicate_stems:
            duplicate_text = ", ".join(sorted(duplicate_stems))

            raise ValueError(
                f"Aynı dosya adına sahip birden fazla {file_type} "
                f"bulundu: {duplicate_text}"
            )

        return result

    @staticmethod
    def _calculate_sections(
        *,
        pairs: list[DatasetPair],
        train_ratio: int,
        val_ratio: int,
        test_ratio: int,
    ) -> tuple[
        list[DatasetPair],
        list[DatasetPair],
        list[DatasetPair],
    ]:
        """Dataset çiftlerini oranlara göre üç bölüme ayırır."""

        total_count = len(pairs)

        train_count = int(total_count * train_ratio / 100)
        val_count = int(total_count * val_ratio / 100)

        test_count = total_count - train_count - val_count

        if test_ratio == 0:
            train_count += test_count
            test_count = 0

        train_end = train_count
        val_end = train_count + val_count

        train_pairs = pairs[:train_end]
        val_pairs = pairs[train_end:val_end]
        test_pairs = pairs[val_end:]

        return train_pairs, val_pairs, test_pairs

    @staticmethod
    def _create_output_directory(
        output_parent: Path,
    ) -> Path:
        """Her split işlemi için ayrı çıktı klasörü oluşturur."""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        output_directory = (
            output_parent / f"split_dataset_{timestamp}"
        )

        counter = 1

        while output_directory.exists():
            output_directory = (
                output_parent
                / f"split_dataset_{timestamp}_{counter}"
            )
            counter += 1

        output_directory.mkdir(parents=True)

        return output_directory

    @staticmethod
    def _create_dataset_folders(
        output_directory: Path,
    ) -> None:
        """YOLO train, val ve test klasör yapısını oluşturur."""

        for section_name in ("train", "val", "test"):
            (
                output_directory
                / "images"
                / section_name
            ).mkdir(parents=True, exist_ok=True)

            (
                output_directory
                / "labels"
                / section_name
            ).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _copy_section(
        *,
        pairs: list[DatasetPair],
        section_name: str,
        output_directory: Path,
    ) -> None:
        """Bir bölüme ait görsel ve label dosyalarını kopyalar."""

        image_output = output_directory / "images" / section_name
        label_output = output_directory / "labels" / section_name

        for pair in pairs:
            target_image_path = image_output / pair.image_path.name
            target_label_path = label_output / f"{pair.image_path.stem}.txt"

            shutil.copy2(
                pair.image_path,
                target_image_path,
            )

            shutil.copy2(
                pair.label_path,
                target_label_path,
            )

    @staticmethod
    def _write_output_yaml(
        *,
        original_yaml: dict[str, Any],
        output_directory: Path,
    ) -> Path:
        """Yeni dataset yollarıyla data.yaml oluşturur."""

        generated_yaml = original_yaml.copy()

        generated_yaml["path"] = str(output_directory)
        generated_yaml["train"] = "images/train"
        generated_yaml["val"] = "images/val"
        generated_yaml["test"] = "images/test"

        output_yaml_path = output_directory / "data.yaml"

        with output_yaml_path.open("w", encoding="utf-8") as yaml_file:
            yaml.safe_dump(
                generated_yaml,
                yaml_file,
                allow_unicode=True,
                sort_keys=False,
                default_flow_style=False,
            )

        return output_yaml_path

    def _analyze_section(
        self,
        *,
        pairs: list[DatasetPair],
        keypoint_count: int,
        keypoint_dimensions: int,
    ) -> SectionAnalysis:
        """Bir dataset bölümünün bütün istatistiklerini hesaplar."""

        class_counts: Counter[int] = Counter()

        keypoint_visibility = {
            keypoint_index: KeypointVisibilityAnalysis()
            for keypoint_index in range(keypoint_count)
        }

        object_count = 0

        for pair in pairs:
            label_result = self._read_label_analysis(
                label_path=pair.label_path,
                keypoint_count=keypoint_count,
                keypoint_dimensions=keypoint_dimensions,
            )

            class_counts.update(label_result["class_counts"])
            object_count += label_result["object_count"]

            file_visibility = label_result["keypoint_visibility"]

            for keypoint_index in range(keypoint_count):
                target = keypoint_visibility[keypoint_index]
                source = file_visibility[keypoint_index]

                target.missing += source.missing
                target.hidden += source.hidden
                target.visible += source.visible

        return SectionAnalysis(
            image_count=len(pairs),
            label_count=len(pairs),
            matched_count=len(pairs),
            object_count=object_count,
            class_counts=class_counts,
            keypoint_visibility=keypoint_visibility,
        )

    @staticmethod
    def _read_label_analysis(
        *,
        label_path: Path,
        keypoint_count: int,
        keypoint_dimensions: int,
    ) -> dict[str, Any]:
        """Tek bir label dosyasındaki nesne, sınıf ve keypointleri analiz eder."""

        class_counts: Counter[int] = Counter()

        keypoint_visibility = {
            keypoint_index: KeypointVisibilityAnalysis()
            for keypoint_index in range(keypoint_count)
        }

        object_count = 0

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

                values = line.split()

                if len(values) != expected_value_count:
                    raise ValueError(
                        f"Geçersiz label değer sayısı: "
                        f"{label_path}, satır {line_number}. "
                        f"Beklenen {expected_value_count}, "
                        f"bulunan {len(values)}."
                    )

                try:
                    numeric_values = [
                        float(value)
                        for value in values
                    ]

                except ValueError as error:
                    raise ValueError(
                        f"Label içerisinde sayısal olmayan değer var: "
                        f"{label_path}, satır {line_number}."
                    ) from error

                class_id_value = numeric_values[0]

                if not class_id_value.is_integer():
                    raise ValueError(
                        f"Class ID tam sayı olmalıdır: "
                        f"{label_path}, satır {line_number}."
                    )

                class_id = int(class_id_value)

                class_counts[class_id] += 1
                object_count += 1

                keypoint_values = numeric_values[5:]

                for keypoint_index in range(keypoint_count):
                    start_index = (
                        keypoint_index * keypoint_dimensions
                    )

                    point_values = keypoint_values[
                        start_index:
                        start_index + keypoint_dimensions
                    ]

                    point_result = keypoint_visibility[
                        keypoint_index
                    ]

                    x_value = point_values[0]
                    y_value = point_values[1]

                    if keypoint_dimensions == 3:
                        visibility_value = int(point_values[2])

                        if visibility_value == 0:
                            point_result.missing += 1

                        elif visibility_value == 1:
                            point_result.hidden += 1

                        elif visibility_value == 2:
                            point_result.visible += 1

                        else:
                            raise ValueError(
                                f"Geçersiz visibility değeri: "
                                f"{label_path}, satır {line_number}, "
                                f"keypoint {keypoint_index}: "
                                f"{visibility_value}"
                            )

                    else:
                        if x_value == 0 and y_value == 0:
                            point_result.missing += 1
                        else:
                            point_result.visible += 1

        return {
            "class_counts": class_counts,
            "object_count": object_count,
            "keypoint_visibility": keypoint_visibility,
        }

    @staticmethod
    def _write_analysis_report(
        *,
        output_directory: Path,
        class_names: dict[int, str],
        keypoint_names: dict[int, str],
        keypoint_dimensions: int,
        train_analysis: SectionAnalysis,
        val_analysis: SectionAnalysis,
        test_analysis: SectionAnalysis,
        missing_label_images: list[Path],
        missing_image_labels: list[Path],
        train_ratio: int,
        val_ratio: int,
        test_ratio: int,
        seed: int,
    ) -> Path:
        """Detaylı analiz.txt dosyasını oluşturur."""

        report_path = output_directory / "analiz.txt"

        all_class_ids = sorted(
            set(class_names)
            | set(train_analysis.class_counts)
            | set(val_analysis.class_counts)
            | set(test_analysis.class_counts)
        )

        lines: list[str] = []

        lines.append("=" * 96)
        lines.append(
            "YOLO POSE VERİ SETİ DETAYLI ANALİZ RAPORU"
        )
        lines.append("=" * 96)
        lines.append("")

        lines.append("BÖLME AYARLARI")
        lines.append("-" * 96)
        lines.append(
            f"Train: %{train_ratio} | "
            f"Validation: %{val_ratio} | "
            f"Test: %{test_ratio}"
        )
        lines.append(f"Random seed: {seed}")
        lines.append("")

        lines.append("BÖLÜM ÖZETİ")
        lines.append("-" * 96)

        lines.append(
            f"{'Bölüm':<14}"
            f"{'Görsel':>10}"
            f"{'Etiket':>10}"
            f"{'Eşleşen':>12}"
            f"{'Nesne':>10}"
            f"{'EksikLbl':>12}"
            f"{'EksikImg':>12}"
        )

        lines.append("-" * 96)

        section_rows = [
            ("train", train_analysis),
            ("val", val_analysis),
            ("test", test_analysis),
        ]

        for section_name, section in section_rows:
            lines.append(
                f"{section_name:<14}"
                f"{section.image_count:>10}"
                f"{section.label_count:>10}"
                f"{section.matched_count:>12}"
                f"{section.object_count:>10}"
                f"{0:>12}"
                f"{0:>12}"
            )

        lines.append("")
        lines.append(
            "SINIF DAĞILIMI "
            "(Train / Val / Test yan yana)"
        )
        lines.append("-" * 96)

        lines.append(
            f"{'ID':>4}  "
            f"{'Sınıf':<32}"
            f"{'train':>12}"
            f"{'val':>12}"
            f"{'test':>12}"
            f"{'Toplam':>12}"
        )

        lines.append("-" * 96)

        for class_id in all_class_ids:
            class_name = class_names.get(
                class_id,
                f"Bilinmeyen sınıf {class_id}",
            )

            train_count = train_analysis.class_counts.get(
                class_id,
                0,
            )

            val_count = val_analysis.class_counts.get(
                class_id,
                0,
            )

            test_count = test_analysis.class_counts.get(
                class_id,
                0,
            )

            total_count = (
                train_count
                + val_count
                + test_count
            )

            lines.append(
                f"{class_id:>4}  "
                f"{class_name:<32}"
                f"{train_count:>12}"
                f"{val_count:>12}"
                f"{test_count:>12}"
                f"{total_count:>12}"
            )

        lines.append("")
        lines.append("KEYPOINT GÖRÜNÜRLÜK ANALİZİ")
        lines.append("-" * 96)

        if keypoint_dimensions == 3:
            lines.append(
                f"{'ID':>4}  "
                f"{'Keypoint':<28}"
                f"{'Train Gör.':>12}"
                f"{'Val Gör.':>12}"
                f"{'Test Gör.':>12}"
                f"{'Gizli':>10}"
                f"{'Eksik':>10}"
                f"{'Toplam':>10}"
            )

            lines.append("-" * 96)

            for keypoint_id in sorted(keypoint_names):
                keypoint_name = keypoint_names[keypoint_id]

                train_data = train_analysis.keypoint_visibility[
                    keypoint_id
                ]

                val_data = val_analysis.keypoint_visibility[
                    keypoint_id
                ]

                test_data = test_analysis.keypoint_visibility[
                    keypoint_id
                ]

                visible_total = (
                    train_data.visible
                    + val_data.visible
                    + test_data.visible
                )

                hidden_total = (
                    train_data.hidden
                    + val_data.hidden
                    + test_data.hidden
                )

                missing_total = (
                    train_data.missing
                    + val_data.missing
                    + test_data.missing
                )

                total = (
                    visible_total
                    + hidden_total
                    + missing_total
                )

                lines.append(
                    f"{keypoint_id:>4}  "
                    f"{keypoint_name:<28}"
                    f"{train_data.visible:>12}"
                    f"{val_data.visible:>12}"
                    f"{test_data.visible:>12}"
                    f"{hidden_total:>10}"
                    f"{missing_total:>10}"
                    f"{total:>10}"
                )

        else:
            lines.append(
                "Dataset keypoint boyutu 2 olduğu için visibility "
                "değeri bulunmamaktadır."
            )

            lines.append("")

            lines.append(
                f"{'ID':>4}  "
                f"{'Keypoint':<30}"
                f"{'Görünür':>14}"
                f"{'Eksik':>14}"
                f"{'Toplam':>14}"
            )

            lines.append("-" * 96)

            for keypoint_id in sorted(keypoint_names):
                keypoint_name = keypoint_names[keypoint_id]

                train_data = train_analysis.keypoint_visibility[
                    keypoint_id
                ]

                val_data = val_analysis.keypoint_visibility[
                    keypoint_id
                ]

                test_data = test_analysis.keypoint_visibility[
                    keypoint_id
                ]

                visible_total = (
                    train_data.visible
                    + val_data.visible
                    + test_data.visible
                )

                missing_total = (
                    train_data.missing
                    + val_data.missing
                    + test_data.missing
                )

                total = visible_total + missing_total

                lines.append(
                    f"{keypoint_id:>4}  "
                    f"{keypoint_name:<30}"
                    f"{visible_total:>14}"
                    f"{missing_total:>14}"
                    f"{total:>14}"
                )

        lines.append("")
        lines.append("EKSİK DOSYALAR")
        lines.append("-" * 96)

        lines.append(
            "Label dosyası olmayan görsel sayısı: "
            f"{len(missing_label_images)}"
        )

        lines.append(
            "Görsel dosyası olmayan label sayısı: "
            f"{len(missing_image_labels)}"
        )

        if missing_label_images:
            lines.append("")
            lines.append("Label dosyası bulunmayan görseller:")

            for image_path in missing_label_images:
                lines.append(f"- {image_path}")

        if missing_image_labels:
            lines.append("")
            lines.append("Görseli bulunmayan label dosyaları:")

            for label_path in missing_image_labels:
                lines.append(f"- {label_path}")

        lines.append("")
        lines.append("=" * 96)
        lines.append("RAPOR SONU")
        lines.append("=" * 96)

        report_path.write_text(
            "\n".join(lines),
            encoding="utf-8",
        )

        return report_path