from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from time import perf_counter
from typing import Any, Callable

import torch
import yaml
from ultralytics import YOLO


BUILD_ID = "2026-07-31-pose-recovery-v2-state-consistency"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

LogCallback = Callable[[str], None]
ProgressCallback = Callable[["TrainingProgress"], None]

RECOVERY_FILENAME = ".active_pose_training.json"


@dataclass(frozen=True)
class TrainingSettings:
    """Ultralytics YOLO Pose eğitim ayarları."""

    data_yaml_path: str
    model_path: str = "yolo11n-pose.pt"
    output_directory: str = "runs"
    run_name: str = "pose_training"

    epochs: int = 1
    image_size: int = 640
    batch_size: int = 2
    device: str = "auto"
    workers: int = 0

    patience: int = 50
    optimizer: str = "auto"
    seed: int = 42
    deterministic: bool = True
    pretrained: bool = True
    cache: bool = False
    amp: bool = False
    plots: bool = True
    save_period: int = 1
    exist_ok: bool = False

    # Detection + Pose loss ağırlıkları. Egzersiz asistanında insan kutusu ve
    # anatomik keypoint doğruluğu birlikte optimize edilir.
    box_loss_gain: float = 7.5
    cls_loss_gain: float = 0.5
    dfl_loss_gain: float = 1.5
    pose_loss_gain: float = 12.0
    keypoint_objectness_gain: float = 1.0
    rle_loss_gain: float = 1.0
    nominal_batch_size: int = 64

    # Optimizer ve learning-rate ayarları.
    initial_learning_rate: float = 0.01
    final_learning_rate_fraction: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    warmup_epochs: float = 3.0
    warmup_momentum: float = 0.8
    warmup_bias_learning_rate: float = 0.1
    cosine_learning_rate: bool = False
    close_mosaic: int = 10

    # Elektrik kesintisinden sonra son tamamlanan epoch'tan devam edebilmek
    # için aktif eğitim durumu JSON dosyasında tutulur.
    automatic_recovery: bool = True

    # Kesintiye uğrayan bir eğitimi optimizer, scheduler ve epoch durumu ile
    # birlikte kaldığı yerden devam ettirmek için kullanılır.
    resume: bool = False
    resume_checkpoint_path: str = ""


@dataclass(frozen=True)
class DatasetSplitSummary:
    """Bir dataset split yoluna ait temel dosya özeti."""

    configured_value: str
    resolved_path: Path | None
    image_count: int | None


@dataclass(frozen=True)
class TrainingValidationResult:
    """Eğitim öncesi doğrulama sonucu."""

    resolved_device: str
    yaml_data: dict[str, Any]
    train_summary: DatasetSplitSummary
    val_summary: DatasetSplitSummary
    test_summary: DatasetSplitSummary | None
    warnings: list[str] = field(default_factory=list)
    resume_mode: bool = False
    resume_checkpoint_path: Path | None = None
    keypoint_count: int = 0
    keypoint_dimensions: int = 3
    keypoint_names_defined: bool = False


@dataclass(frozen=True)
class TrainingRecoveryState:
    """Elektrik kesintisi sonrası yeniden yüklenebilen eğitim durumu."""

    state_path: Path
    status: str
    checkpoint_path: Path
    run_directory: Path | None
    last_completed_epoch: int
    total_epochs: int
    settings: dict[str, Any]
    updated_at: str


@dataclass(frozen=True)
class TrainingProgress:
    """Arayüze gönderilecek epoch ilerleme bilgisi."""

    epoch: int
    total_epochs: int
    percent: float
    metrics: dict[str, float] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class TrainingResult:
    """Tamamlanan eğitimden dönen çıktı yolları ve metrikler."""

    run_directory: Path
    weights_directory: Path
    best_model_path: Path | None
    last_model_path: Path | None
    results_csv_path: Path | None
    args_yaml_path: Path | None
    plot_paths: list[Path]
    metrics: dict[str, float]
    resolved_device: str
    elapsed_seconds: float
    stopped_early: bool
    resumed: bool = False
    resume_checkpoint_path: Path | None = None
    recovery_state_path: Path | None = None


class PoseTrainingService:
    """
    Ultralytics YOLO Pose model eğitimini yöneten servis.

    Servis senkron çalışır. PySide6 arayüzüne bağlanırken bir QThread
    veya worker thread içinde çağrılmalıdır; aksi hâlde eğitim süresince
    ana pencere donar.
    """

    def __init__(self) -> None:
        self._state_lock = Lock()
        self._stop_requested = False
        self._active_trainer: Any | None = None
        self._active_recovery_state_path: Path | None = None
        self._active_settings: TrainingSettings | None = None
        self._active_resume_checkpoint_path: Path | None = None

    def request_stop(self) -> bool:
        """Devam eden eğitimin epoch sonunda durmasını ister.

        Dönüş değeri, servis tarafından aktif bir trainer nesnesinin görülüp
        görülmediğini bildirir. Trainer henüz kurulmamış olsa bile bayrak
        korunur ve ilk tamamlanan epoch sonunda uygulanır.
        """

        with self._state_lock:
            self._stop_requested = True
            trainer_available = self._active_trainer is not None
        return trainer_available

    def reset_stop_request(self) -> None:
        """Yeni eğitim thread'i başlatılmadan önce durdurma isteğini sıfırlar."""

        with self._state_lock:
            self._stop_requested = False

    def is_stop_requested(self) -> bool:
        """Arayüzün güncel güvenli durdurma durumunu okumasını sağlar."""

        return self._is_stop_requested()

    def _set_active_trainer(self, trainer: Any | None) -> None:
        with self._state_lock:
            self._active_trainer = trainer

    def _get_active_trainer(self) -> Any | None:
        with self._state_lock:
            return self._active_trainer

    @staticmethod
    def recovery_state_path(output_directory: str) -> Path:
        # Merkezi pointer proje çalışma dizininde tutulur. Böylece kullanıcı
        # çıktı klasörünü proje dışında seçmiş olsa bile uygulama yeniden
        # açıldığında yarım eğitimi bulabilir.
        del output_directory
        return Path.cwd().resolve() / RECOVERY_FILENAME

    def find_recovery_state(
        self,
        search_directory: str,
    ) -> TrainingRecoveryState | None:
        """Aktif/yarım kalmış eğitimin last.pt checkpoint'ini bulur."""

        raw_root = search_directory.strip() or "runs"
        root = Path(raw_root).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root = root.resolve()

        candidate_paths: list[Path] = []
        central_state = Path.cwd().resolve() / RECOVERY_FILENAME
        if central_state.is_file():
            candidate_paths.append(central_state)
        direct_state = root / RECOVERY_FILENAME
        if direct_state.is_file() and direct_state not in candidate_paths:
            candidate_paths.append(direct_state)
        if root.is_dir():
            candidate_paths.extend(root.rglob(RECOVERY_FILENAME))

        states: list[TrainingRecoveryState] = []
        for state_path in candidate_paths:
            try:
                payload = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                continue

            if not isinstance(payload, dict):
                continue
            status = str(payload.get("status", "interrupted"))
            if status == "completed":
                continue

            raw_checkpoint = str(payload.get("last_checkpoint", "")).strip()
            if not raw_checkpoint:
                raw_checkpoint = str(payload.get("source_checkpoint", "")).strip()
            raw_run = str(payload.get("run_directory", "")).strip()
            checkpoint_path: Path | None = None
            if raw_checkpoint:
                checkpoint_path = Path(raw_checkpoint).expanduser()
                if not checkpoint_path.is_absolute():
                    checkpoint_path = state_path.parent / checkpoint_path
                checkpoint_path = checkpoint_path.resolve()

            run_directory: Path | None = None
            if raw_run:
                run_directory = Path(raw_run).expanduser()
                if not run_directory.is_absolute():
                    run_directory = state_path.parent / run_directory
                run_directory = run_directory.resolve()

            if (checkpoint_path is None or not checkpoint_path.is_file()) and run_directory:
                fallback = run_directory / "weights" / "last.pt"
                if fallback.is_file():
                    checkpoint_path = fallback.resolve()

            if checkpoint_path is None or not checkpoint_path.is_file():
                continue

            raw_settings = payload.get("settings", {})
            settings_payload = raw_settings if isinstance(raw_settings, dict) else {}
            states.append(
                TrainingRecoveryState(
                    state_path=state_path.resolve(),
                    status=status,
                    checkpoint_path=checkpoint_path,
                    run_directory=run_directory,
                    last_completed_epoch=int(payload.get("last_completed_epoch", 0) or 0),
                    total_epochs=int(payload.get("total_epochs", 0) or 0),
                    settings=settings_payload,
                    updated_at=str(payload.get("updated_at", "")),
                )
            )

        if not states:
            return None
        return max(
            states,
            key=lambda state: state.state_path.stat().st_mtime,
        )

    def _initialize_recovery_state(
        self,
        *,
        settings: TrainingSettings,
        checkpoint_path: Path | None,
    ) -> Path | None:
        if not settings.automatic_recovery:
            self._active_recovery_state_path = None
            self._active_settings = settings
            self._active_resume_checkpoint_path = (
                checkpoint_path.expanduser().resolve()
                if checkpoint_path is not None
                else None
            )
            return None

        state_path = self.recovery_state_path(settings.output_directory)
        expected_run = (
            checkpoint_path.parent.parent
            if checkpoint_path is not None
            else Path(settings.output_directory).expanduser().resolve() / settings.run_name
        )
        self._active_recovery_state_path = state_path
        self._active_settings = settings
        self._active_resume_checkpoint_path = (
            checkpoint_path.expanduser().resolve()
            if checkpoint_path is not None
            else None
        )
        self._write_recovery_state(
            status="running",
            run_directory=expected_run,
            checkpoint_path=checkpoint_path,
            last_completed_epoch=0,
            total_epochs=settings.epochs,
        )
        return state_path

    def _resolve_recovery_checkpoint(
        self,
        *,
        state_path: Path,
        run_directory: Path | None,
        checkpoint_path: Path | None,
    ) -> Path | None:
        """Callback geçici olarak None verse bile son geçerli checkpoint'i korur."""

        candidates: list[Path] = []
        if checkpoint_path is not None:
            candidates.append(checkpoint_path.expanduser())
        if run_directory is not None:
            candidates.append(run_directory.expanduser() / "weights" / "last.pt")
        if self._active_resume_checkpoint_path is not None:
            candidates.append(self._active_resume_checkpoint_path.expanduser())

        if state_path.is_file():
            try:
                existing = json.loads(state_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError, TypeError):
                existing = {}
            if isinstance(existing, dict):
                for key in ("last_checkpoint", "source_checkpoint"):
                    raw_value = str(existing.get(key, "")).strip()
                    if raw_value:
                        candidate = Path(raw_value).expanduser()
                        if not candidate.is_absolute():
                            candidate = state_path.parent / candidate
                        candidates.append(candidate)

        seen: set[Path] = set()
        for candidate in candidates:
            try:
                resolved = candidate.resolve()
            except OSError:
                continue
            if resolved in seen:
                continue
            seen.add(resolved)
            if resolved.is_file():
                return resolved
        return None

    def _write_recovery_state(
        self,
        *,
        status: str,
        run_directory: Path | None,
        checkpoint_path: Path | None,
        last_completed_epoch: int,
        total_epochs: int,
    ) -> None:
        state_path = self._active_recovery_state_path
        settings = self._active_settings
        if state_path is None or settings is None:
            return

        resolved_checkpoint = self._resolve_recovery_checkpoint(
            state_path=state_path,
            run_directory=run_directory,
            checkpoint_path=checkpoint_path,
        )
        resolved_source_checkpoint = self._active_resume_checkpoint_path
        resolved_run_directory = run_directory
        if resolved_run_directory is None and resolved_checkpoint is not None:
            parent = resolved_checkpoint.parent
            resolved_run_directory = (
                parent.parent if parent.name == "weights" else parent
            )

        effective_total_epochs = max(0, int(total_epochs))
        payload = {
            "schema_version": 2,
            "status": status,
            "task": "pose",
            "mode": "resume" if settings.resume else "new",
            "run_directory": (
                str(resolved_run_directory.expanduser().resolve())
                if resolved_run_directory is not None
                else ""
            ),
            "last_checkpoint": (
                str(resolved_checkpoint) if resolved_checkpoint is not None else ""
            ),
            "source_checkpoint": (
                str(resolved_source_checkpoint.expanduser().resolve())
                if resolved_source_checkpoint is not None
                else ""
            ),
            "last_completed_epoch": max(0, int(last_completed_epoch)),
            "total_epochs": effective_total_epochs,
            "requested_epochs": max(0, int(settings.epochs)),
            "effective_total_epochs": effective_total_epochs,
            "settings": asdict(settings),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        state_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = state_path.with_suffix(state_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(state_path)

    def _finish_recovery_state(
        self,
        *,
        completed: bool,
        stopped: bool,
        trainer: Any | None,
    ) -> None:
        state_path = self._active_recovery_state_path
        if state_path is None:
            return

        run_directory = self._trainer_run_directory(trainer)
        checkpoint_path = self._trainer_last_checkpoint(trainer)
        epoch = int(getattr(trainer, "epoch", -1)) + 1 if trainer is not None else 0
        total_epochs = self._trainer_total_epochs(
            trainer,
            fallback=self._active_settings.epochs if self._active_settings else 0,
        ) if trainer is not None else 0

        if completed and not stopped:
            try:
                state_path.unlink(missing_ok=True)
            except OSError:
                self._write_recovery_state(
                    status="completed",
                    run_directory=run_directory,
                    checkpoint_path=checkpoint_path,
                    last_completed_epoch=epoch,
                    total_epochs=total_epochs,
                )
            return

        self._write_recovery_state(
            status="stopped" if stopped else "interrupted",
            run_directory=run_directory,
            checkpoint_path=checkpoint_path,
            last_completed_epoch=epoch,
            total_epochs=total_epochs,
        )

    def validate_settings(
        self,
        settings: TrainingSettings,
    ) -> TrainingValidationResult:
        """Dosya yollarını, YAML içeriğini ve eğitim ayarlarını doğrular."""

        yaml_path = Path(settings.data_yaml_path).expanduser().resolve()

        if not yaml_path.is_file():
            raise FileNotFoundError(
                f"data.yaml bulunamadı: {yaml_path}"
            )

        if yaml_path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError(
                "Dataset yapılandırma dosyası .yaml veya .yml olmalıdır."
            )

        yaml_data = self._read_yaml(yaml_path)
        keypoint_count, keypoint_dimensions, keypoint_names_defined = (
            self._validate_yaml_content(yaml_data)
        )
        self._validate_numeric_settings(settings)

        resume_checkpoint_path: Path | None = None
        if settings.resume:
            resume_checkpoint_path = self._validate_resume_checkpoint(
                settings.resume_checkpoint_path
            )
        else:
            self._validate_run_name(settings.run_name)
            self._validate_model_reference(settings.model_path)

            output_directory = (
                Path(settings.output_directory).expanduser().resolve()
            )
            output_directory.mkdir(parents=True, exist_ok=True)

            if not output_directory.is_dir():
                raise NotADirectoryError(
                    f"Eğitim çıktı yolu klasör değil: {output_directory}"
                )

        resolved_device = self.resolve_device(settings.device)
        dataset_root = self._resolve_dataset_root(
            yaml_path=yaml_path,
            yaml_data=yaml_data,
        )

        train_summary = self._build_split_summary(
            configured_value=yaml_data["train"],
            dataset_root=dataset_root,
        )
        val_summary = self._build_split_summary(
            configured_value=yaml_data["val"],
            dataset_root=dataset_root,
        )

        test_summary: DatasetSplitSummary | None = None
        configured_test = yaml_data.get("test")
        if configured_test is not None and configured_test != "":
            test_summary = self._build_split_summary(
                configured_value=configured_test,
                dataset_root=dataset_root,
            )

        warnings: list[str] = []

        self._append_split_warnings(
            split_name="Train",
            summary=train_summary,
            warnings=warnings,
        )
        self._append_split_warnings(
            split_name="Validation",
            summary=val_summary,
            warnings=warnings,
        )

        if test_summary is not None:
            self._append_split_warnings(
                split_name="Test",
                summary=test_summary,
                warnings=warnings,
            )

        if (
            train_summary.resolved_path is not None
            and val_summary.resolved_path is not None
            and train_summary.resolved_path == val_summary.resolved_path
        ):
            warnings.append(
                "Train ve validation aynı yolu gösteriyor. Bu durum "
                "validation sonucunu yanıltıcı hâle getirebilir."
            )

        if settings.resume:
            warnings.append(
                "Devam modu açık. Model, optimizer, learning-rate scheduler, "
                "epoch ve önceki eğitim ayarları checkpoint içinden yüklenir. "
                "Yeni eğitim alanlarının çoğu bu modda kullanılmaz."
            )
            if (
                resume_checkpoint_path is not None
                and resume_checkpoint_path.name != "last.pt"
            ):
                warnings.append(
                    "Seçilen checkpoint last.pt değil. Resume için last.pt "
                    "tercih edilir; epoch*.pt dosyaları yedek olarak kullanılabilir."
                )

        if not keypoint_names_defined:
            warnings.append(
                "kpt_names tanımlı değil. Egzersiz analiz motoru omuz, kalça, "
                "diz ve ayak bileği indekslerini anatomik adlarla eşleştiremez."
            )

        if settings.automatic_recovery and settings.save_period != 1:
            warnings.append(
                "Otomatik recovery açık olduğu için save_period eğitim sırasında 1 "
                "olarak uygulanacaktır."
            )

        if resolved_device == "mps":
            warnings.append(
                "Apple MPS seçildi. Pose eğitiminde beklenmeyen bir MPS "
                "hatası oluşursa aynı ayarlarla CPU denenebilir."
            )

        return TrainingValidationResult(
            resolved_device=resolved_device,
            yaml_data=yaml_data,
            train_summary=train_summary,
            val_summary=val_summary,
            test_summary=test_summary,
            warnings=warnings,
            resume_mode=settings.resume,
            resume_checkpoint_path=resume_checkpoint_path,
            keypoint_count=keypoint_count,
            keypoint_dimensions=keypoint_dimensions,
            keypoint_names_defined=keypoint_names_defined,
        )

    def train(
        self,
        settings: TrainingSettings,
        *,
        log_callback: LogCallback | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> TrainingResult:
        """YOLO Pose eğitimini başlatır ve çıktı yollarını döndürür.

        Durdurma bayrağı burada sıfırlanmaz. Arayüz, worker thread'i
        başlatmadan hemen önce ``reset_stop_request`` çağırır. Böylece
        Başlat düğmesinden hemen sonra gelen bir durdurma isteği yarış
        koşulu nedeniyle kaybolmaz.
        """

        validation = self.validate_settings(settings)

        self._emit_log(log_callback, "Eğitim ayarları doğrulandı.")
        self._emit_log(
            log_callback,
            f"Cihaz: {validation.resolved_device}",
        )
        if settings.resume:
            self._emit_log(
                log_callback,
                f"Devam checkpoint'i: {validation.resume_checkpoint_path}",
            )
            self._emit_log(
                log_callback,
                "Kesintiye uğrayan eğitim son tamamlanan epoch'tan devam edecek.",
            )
        else:
            self._emit_log(
                log_callback,
                f"Model: {settings.model_path}",
            )
        self._emit_log(
            log_callback,
            f"data.yaml: {Path(settings.data_yaml_path).expanduser().resolve()}",
        )

        if validation.train_summary.image_count is not None:
            self._emit_log(
                log_callback,
                f"Train görsel sayısı: "
                f"{validation.train_summary.image_count}",
            )

        if validation.val_summary.image_count is not None:
            self._emit_log(
                log_callback,
                f"Validation görsel sayısı: "
                f"{validation.val_summary.image_count}",
            )

        for warning in validation.warnings:
            self._emit_log(log_callback, f"UYARI: {warning}")

        model_reference = (
            str(validation.resume_checkpoint_path)
            if settings.resume and validation.resume_checkpoint_path is not None
            else settings.model_path
        )
        recovery_state_path = self._initialize_recovery_state(
            settings=settings,
            checkpoint_path=validation.resume_checkpoint_path,
        )

        model = YOLO(model_reference)
        model_task = str(getattr(model, "task", "pose") or "pose").lower()
        if model_task != "pose":
            raise ValueError(
                f"Seçilen model pose görevi için değil: task={model_task}. "
                "Egzersiz asistanında -pose.pt veya pose model YAML kullanın."
            )

        self._register_callbacks(
            model=model,
            settings=settings,
            log_callback=log_callback,
            progress_callback=progress_callback,
        )

        output_directory = (
            Path(settings.output_directory).expanduser().resolve()
        )

        if settings.resume:
            # Resume sırasında checkpoint içindeki özgün eğitim ayarları,
            # optimizer/scheduler durumu ve hedef epoch korunur. Cihaz seçimi
            # gerektiğinde farklı donanıma geçebilmek için açıkça verilir.
            train_arguments: dict[str, Any] = {
                "resume": True,
                "device": validation.resolved_device,
            }
            self._emit_log(
                log_callback,
                "Checkpoint durumu yükleniyor ve eğitim devam ettiriliyor.",
            )
        else:
            train_arguments = {
                "data": str(
                    Path(settings.data_yaml_path).expanduser().resolve()
                ),
                "epochs": settings.epochs,
                "imgsz": settings.image_size,
                "batch": settings.batch_size,
                "device": validation.resolved_device,
                "workers": settings.workers,
                "project": str(output_directory),
                "name": settings.run_name,
                "patience": settings.patience,
                "optimizer": settings.optimizer,
                "seed": settings.seed,
                "deterministic": settings.deterministic,
                "pretrained": settings.pretrained,
                "cache": settings.cache,
                "amp": settings.amp,
                "plots": settings.plots,
                "save_period": 1 if settings.automatic_recovery else settings.save_period,
                "exist_ok": settings.exist_ok,
                "save": True,
                "val": True,
                "verbose": True,
                "box": settings.box_loss_gain,
                "cls": settings.cls_loss_gain,
                "dfl": settings.dfl_loss_gain,
                "pose": settings.pose_loss_gain,
                "kobj": settings.keypoint_objectness_gain,
                "rle": settings.rle_loss_gain,
                "nbs": settings.nominal_batch_size,
                "lr0": settings.initial_learning_rate,
                "lrf": settings.final_learning_rate_fraction,
                "momentum": settings.momentum,
                "weight_decay": settings.weight_decay,
                "warmup_epochs": settings.warmup_epochs,
                "warmup_momentum": settings.warmup_momentum,
                "warmup_bias_lr": settings.warmup_bias_learning_rate,
                "cos_lr": settings.cosine_learning_rate,
                "close_mosaic": settings.close_mosaic,
            }

            self._emit_log(
                log_callback,
                (
                    "Eğitim başlatılıyor: "
                    f"epochs={settings.epochs}, "
                    f"imgsz={settings.image_size}, "
                    f"batch={settings.batch_size}, "
                    f"box={settings.box_loss_gain}, dfl={settings.dfl_loss_gain}, "
                    f"pose={settings.pose_loss_gain}, "
                    f"kobj={settings.keypoint_objectness_gain}"
                ),
            )

        started_at = perf_counter()

        try:
            model.train(**train_arguments)
        except Exception as error:
            trainer_on_error = getattr(model, "trainer", None)
            self._finish_recovery_state(
                completed=False,
                stopped=self._is_stop_requested(),
                trainer=trainer_on_error,
            )
            self._emit_log(
                log_callback,
                f"EĞİTİM HATASI: {error}",
            )
            raise RuntimeError(
                f"YOLO Pose eğitimi tamamlanamadı: {error}"
            ) from error
        finally:
            # Eğitim normal bittiğinde veya hata aldığında eski trainer
            # referansının sonraki eğitime taşınmasını engeller.
            self._set_active_trainer(None)

        elapsed_seconds = perf_counter() - started_at
        trainer = getattr(model, "trainer", None)

        if trainer is None:
            raise RuntimeError(
                "Eğitim tamamlandı ancak Ultralytics trainer sonucu "
                "alınamadı."
            )

        self._finish_recovery_state(
            completed=not self._is_stop_requested(),
            stopped=self._is_stop_requested(),
            trainer=trainer,
        )

        result = self._build_training_result(
            trainer=trainer,
            resolved_device=validation.resolved_device,
            elapsed_seconds=elapsed_seconds,
            resumed=settings.resume,
            resume_checkpoint_path=validation.resume_checkpoint_path,
            recovery_state_path=recovery_state_path,
        )

        self._emit_log(
            log_callback,
            f"Eğitim klasörü: {result.run_directory}",
        )

        if result.best_model_path is not None:
            self._emit_log(
                log_callback,
                f"best.pt: {result.best_model_path}",
            )

        if result.last_model_path is not None:
            self._emit_log(
                log_callback,
                f"last.pt: {result.last_model_path}",
            )

        self._emit_log(
            log_callback,
            f"Toplam süre: {elapsed_seconds:.2f} saniye",
        )

        return result

    @staticmethod
    def resolve_device(device: str) -> str:
        """auto/cpu/mps/cuda cihaz seçimini doğrular ve çözümler."""

        normalized = device.strip().lower()

        if not normalized:
            raise ValueError("Device alanı boş bırakılamaz.")

        if normalized == "auto":
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "0"
            return "cpu"

        if normalized == "mps":
            if not torch.backends.mps.is_available():
                raise RuntimeError(
                    "MPS seçildi ancak bu ortamda kullanılamıyor."
                )
            return "mps"

        if normalized in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA seçildi ancak kullanılabilir NVIDIA GPU yok."
                )
            return "0"

        if normalized == "cpu":
            return "cpu"

        if normalized.isdigit():
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA cihaz numarası seçildi ancak CUDA kullanılamıyor."
                )
            return normalized

        raise ValueError(
            "Device değeri auto, mps, cpu, cuda veya CUDA cihaz "
            "numarası olmalıdır."
        )

    def _register_callbacks(
        self,
        *,
        model: YOLO,
        settings: TrainingSettings,
        log_callback: LogCallback | None,
        progress_callback: ProgressCallback | None,
    ) -> None:
        """Ultralytics callbacklerini servis callbacklerine bağlar."""

        def on_train_start(trainer: Any) -> None:
            self._set_active_trainer(trainer)
            self._emit_log(
                log_callback,
                f"Ultralytics eğitim döngüsü başladı: {trainer.save_dir}",
            )
            effective_total_epochs = self._trainer_total_epochs(
                trainer, fallback=settings.epochs
            )
            if settings.resume:
                self._emit_log(
                    log_callback,
                    "Resume hedefi checkpoint'ten okundu: "
                    f"toplam epoch={effective_total_epochs}. "
                    f"Arayüzdeki epoch={settings.epochs} bu modda uygulanmaz.",
                )
            self._write_recovery_state(
                status="running",
                run_directory=Path(trainer.save_dir).expanduser().resolve(),
                checkpoint_path=self._trainer_last_checkpoint(trainer),
                last_completed_epoch=max(0, int(getattr(trainer, "epoch", -1))),
                total_epochs=effective_total_epochs,
            )
            if self._is_stop_requested():
                self._emit_log(
                    log_callback,
                    "Durdurma isteği eğitim başlamadan alındı. İlk tamamlanan "
                    "epoch ve checkpoint sonrasında eğitim duracak.",
                )

        def on_train_epoch_start(trainer: Any) -> None:
            current_epoch = int(getattr(trainer, "epoch", 0)) + 1
            total_epochs = self._trainer_total_epochs(
                trainer,
                fallback=settings.epochs,
            )
            self._emit_log(
                log_callback,
                f"Epoch {current_epoch}/{total_epochs} başladı.",
            )

        def on_train_epoch_end(trainer: Any) -> None:
            """Stop isteğini epoch eğitim döngüsü biter bitmez uygular.

            Ultralytics bu callback sonrasında validation ve save_model akışını
            çalıştırdığı için last.pt korunur ve sonraki epoch başlamaz.
            """

            if not self._is_stop_requested():
                return

            trainer.stop = True
            self._emit_log(
                log_callback,
                "Epoch sonu durdurma uygulandı. Validation ve last.pt kaydı "
                "tamamlandıktan sonra eğitim döngüsü sona erecek.",
            )

        def on_fit_epoch_end(trainer: Any) -> None:
            total_epochs = self._trainer_total_epochs(
                trainer,
                fallback=settings.epochs,
            )
            raw_epoch = int(getattr(trainer, "epoch", 0))
            current_epoch = max(1, min(total_epochs, raw_epoch + 1))
            metrics = self._collect_trainer_metrics(trainer)
            percent = min(
                100.0,
                (current_epoch / max(total_epochs, 1)) * 100.0,
            )

            message = f"Epoch {current_epoch}/{total_epochs} tamamlandı."
            metric_text = self._format_metric_summary(metrics)
            if metric_text:
                message = f"{message} {metric_text}"

            self._emit_log(log_callback, message)
            self._write_recovery_state(
                status="running",
                run_directory=self._trainer_run_directory(trainer),
                checkpoint_path=self._trainer_last_checkpoint(trainer),
                last_completed_epoch=current_epoch,
                total_epochs=total_epochs,
            )

            if progress_callback is not None:
                progress_callback(
                    TrainingProgress(
                        epoch=current_epoch,
                        total_epochs=total_epochs,
                        percent=percent,
                        metrics=metrics,
                        message=message,
                    )
                )

            if self._is_stop_requested():
                # on_fit_epoch_end, Ultralytics tarafından validation ve
                # checkpoint kaydından sonra çağrılır. Burada trainer.stop=True
                # yapılınca ana eğitim döngüsü bir sonraki epoch'a geçmez.
                trainer.stop = True
                self._emit_log(
                    log_callback,
                    "Durdurma isteği uygulandı: checkpoint kaydedildi ve "
                    "eğitim bir sonraki epoch başlamadan duracak.",
                )

        def on_model_save(trainer: Any) -> None:
            last_path = self._existing_path(
                getattr(trainer, "last", None)
            )
            if last_path is not None:
                self._emit_log(
                    log_callback,
                    f"Checkpoint kaydedildi: {last_path}",
                )
                self._write_recovery_state(
                    status="running",
                    run_directory=self._trainer_run_directory(trainer),
                    checkpoint_path=last_path,
                    last_completed_epoch=int(getattr(trainer, "epoch", -1)) + 1,
                    total_epochs=self._trainer_total_epochs(
                        trainer, fallback=settings.epochs
                    ),
                )

            if self._is_stop_requested():
                # Bu callback checkpoint yazıldıktan sonra çalışır. Stop
                # bayrağını burada da ayarlamak, farklı Ultralytics callback
                # sıralamalarında yeni epoch başlamasını engeller.
                trainer.stop = True
                self._emit_log(
                    log_callback,
                    "last.pt kaydı tamamlandı; güvenli durdurma kesinleştirildi.",
                )

        def on_train_end(trainer: Any) -> None:
            self._set_active_trainer(None)
            if self._is_stop_requested():
                self._emit_log(
                    log_callback,
                    "Eğitim durdurma isteğiyle sona erdi.",
                )
            else:
                self._emit_log(
                    log_callback,
                    "Ultralytics eğitim döngüsü tamamlandı.",
                )

        model.add_callback("on_train_start", on_train_start)
        model.add_callback("on_train_epoch_start", on_train_epoch_start)
        model.add_callback("on_train_epoch_end", on_train_epoch_end)
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
        model.add_callback("on_model_save", on_model_save)
        model.add_callback("on_train_end", on_train_end)

    def _build_training_result(
        self,
        *,
        trainer: Any,
        resolved_device: str,
        elapsed_seconds: float,
        resumed: bool,
        resume_checkpoint_path: Path | None,
        recovery_state_path: Path | None = None,
    ) -> TrainingResult:
        run_directory = Path(trainer.save_dir).expanduser().resolve()
        weights_directory = run_directory / "weights"

        best_model_path = self._existing_path(
            getattr(trainer, "best", weights_directory / "best.pt")
        )
        last_model_path = self._existing_path(
            getattr(trainer, "last", weights_directory / "last.pt")
        )
        results_csv_path = self._existing_path(
            run_directory / "results.csv"
        )
        args_yaml_path = self._existing_path(
            run_directory / "args.yaml"
        )

        plot_paths = sorted(
            path
            for path in run_directory.rglob("*")
            if path.is_file()
            and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        )

        metrics = self._collect_trainer_metrics(trainer)

        return TrainingResult(
            run_directory=run_directory,
            weights_directory=weights_directory,
            best_model_path=best_model_path,
            last_model_path=last_model_path,
            results_csv_path=results_csv_path,
            args_yaml_path=args_yaml_path,
            plot_paths=plot_paths,
            metrics=metrics,
            resolved_device=resolved_device,
            elapsed_seconds=elapsed_seconds,
            stopped_early=self._is_stop_requested(),
            resumed=resumed,
            resume_checkpoint_path=resume_checkpoint_path,
            recovery_state_path=recovery_state_path,
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
    def _validate_yaml_content(
        yaml_data: dict[str, Any],
    ) -> tuple[int, int, bool]:
        required_fields = {"train", "val", "names", "kpt_shape"}
        missing_fields = sorted(required_fields.difference(yaml_data))

        if missing_fields:
            raise ValueError(
                "data.yaml içinde eksik alanlar var: "
                + ", ".join(missing_fields)
            )

        names = yaml_data["names"]
        if not isinstance(names, (list, dict)) or not names:
            raise ValueError(
                "data.yaml içindeki names alanı boş olmayan list veya "
                "dictionary olmalıdır."
            )

        kpt_shape = yaml_data["kpt_shape"]
        if not isinstance(kpt_shape, (list, tuple)) or len(kpt_shape) != 2:
            raise ValueError(
                "kpt_shape biçimi [keypoint_sayısı, boyut] olmalıdır."
            )

        try:
            keypoint_count = int(kpt_shape[0])
            keypoint_dimensions = int(kpt_shape[1])
        except (TypeError, ValueError) as error:
            raise ValueError("kpt_shape sayısal değerler içermelidir.") from error

        if keypoint_count <= 0:
            raise ValueError(
                "kpt_shape içindeki keypoint sayısı sıfırdan büyük olmalıdır."
            )

        if keypoint_dimensions not in {2, 3}:
            raise ValueError(
                "kpt_shape boyutu yalnızca 2 veya 3 olabilir."
            )

        flip_idx = yaml_data.get("flip_idx")
        if flip_idx is not None:
            if not isinstance(flip_idx, list):
                raise ValueError("flip_idx bir liste olmalıdır.")
            if len(flip_idx) != keypoint_count:
                raise ValueError(
                    "flip_idx eleman sayısı kpt_shape keypoint sayısıyla "
                    "aynı olmalıdır."
                )
            try:
                normalized_flip = [int(value) for value in flip_idx]
            except (TypeError, ValueError) as error:
                raise ValueError("flip_idx yalnızca tam sayı içermelidir.") from error
            if any(index < 0 or index >= keypoint_count for index in normalized_flip):
                raise ValueError(
                    "flip_idx değerleri 0 ile keypoint_sayısı-1 arasında olmalıdır."
                )

        kpt_names = yaml_data.get("kpt_names")
        keypoint_names_defined = kpt_names is not None
        if kpt_names is not None:
            name_lists: list[Any]
            if isinstance(kpt_names, dict):
                name_lists = list(kpt_names.values())
            elif isinstance(kpt_names, list):
                # Tek sınıflı veri setlerinde doğrudan listeyi de kabul et.
                name_lists = [kpt_names]
            else:
                raise ValueError("kpt_names liste veya dictionary olmalıdır.")
            for class_names in name_lists:
                if not isinstance(class_names, list):
                    raise ValueError("Her kpt_names sınıf değeri liste olmalıdır.")
                if len(class_names) != keypoint_count:
                    raise ValueError(
                        "kpt_names içindeki anatomik ad sayısı kpt_shape "
                        "keypoint sayısıyla aynı olmalıdır."
                    )
                if any(not str(name).strip() for name in class_names):
                    raise ValueError("kpt_names boş anatomik ad içeremez.")

        oks_sigmas = yaml_data.get("kpt_oks_sigmas")
        if oks_sigmas is not None:
            if not isinstance(oks_sigmas, list) or len(oks_sigmas) != keypoint_count:
                raise ValueError(
                    "kpt_oks_sigmas liste olmalı ve keypoint sayısı kadar değer "
                    "içermelidir."
                )
            try:
                normalized_sigmas = [float(value) for value in oks_sigmas]
            except (TypeError, ValueError) as error:
                raise ValueError("kpt_oks_sigmas sayısal olmalıdır.") from error
            if any(value <= 0 for value in normalized_sigmas):
                raise ValueError("kpt_oks_sigmas değerleri pozitif olmalıdır.")

        return keypoint_count, keypoint_dimensions, keypoint_names_defined

    @staticmethod
    def _validate_numeric_settings(settings: TrainingSettings) -> None:
        if settings.epochs <= 0:
            raise ValueError("Epoch değeri sıfırdan büyük olmalıdır.")

        if settings.image_size <= 0:
            raise ValueError("Image size sıfırdan büyük olmalıdır.")

        if settings.batch_size == 0 or settings.batch_size < -1:
            raise ValueError(
                "Batch size -1 veya sıfırdan büyük bir sayı olmalıdır."
            )

        if settings.workers < 0:
            raise ValueError("Workers negatif olamaz.")

        if settings.patience < 0:
            raise ValueError("Patience negatif olamaz.")

        if settings.save_period == 0 or settings.save_period < -1:
            raise ValueError(
                "save_period -1 veya sıfırdan büyük bir sayı olmalıdır."
            )

        positive_losses = {
            "box": settings.box_loss_gain,
            "cls": settings.cls_loss_gain,
            "dfl": settings.dfl_loss_gain,
            "pose": settings.pose_loss_gain,
            "kobj": settings.keypoint_objectness_gain,
            "rle": settings.rle_loss_gain,
        }
        for name, value in positive_losses.items():
            if value < 0:
                raise ValueError(f"{name} loss gain negatif olamaz.")

        if settings.nominal_batch_size <= 0:
            raise ValueError("Nominal batch size sıfırdan büyük olmalıdır.")
        if settings.initial_learning_rate <= 0:
            raise ValueError("lr0 sıfırdan büyük olmalıdır.")
        if not 0 < settings.final_learning_rate_fraction <= 1:
            raise ValueError("lrf 0 ile 1 arasında olmalıdır.")
        if settings.momentum < 0:
            raise ValueError("Momentum negatif olamaz.")
        if settings.weight_decay < 0:
            raise ValueError("Weight decay negatif olamaz.")
        if settings.warmup_epochs < 0:
            raise ValueError("Warmup epoch negatif olamaz.")
        if settings.warmup_momentum < 0:
            raise ValueError("Warmup momentum negatif olamaz.")
        if settings.warmup_bias_learning_rate < 0:
            raise ValueError("Warmup bias LR negatif olamaz.")
        if settings.close_mosaic < 0:
            raise ValueError("close_mosaic negatif olamaz.")

    @staticmethod
    def _validate_resume_checkpoint(raw_path: str) -> Path:
        cleaned_path = raw_path.strip()
        if not cleaned_path:
            raise ValueError("Devam etmek için checkpoint dosyası seçilmedi.")

        checkpoint_path = Path(cleaned_path).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Checkpoint dosyası bulunamadı: {checkpoint_path}"
            )

        if checkpoint_path.suffix.lower() != ".pt":
            raise ValueError("Resume checkpoint'i bir .pt dosyası olmalıdır.")

        if checkpoint_path.name == "best.pt":
            raise ValueError(
                "best.pt resume için kullanılmamalıdır. Optimizer ve epoch "
                "durumunu koruyan last.pt veya epoch*.pt seçin."
            )

        return checkpoint_path

    @staticmethod
    def find_latest_checkpoint(search_directory: str) -> Path | None:
        """Bir yol altında en yeni last.pt/epoch*.pt checkpoint'ini bulur.

        Verilen yol doğrudan bir checkpoint dosyasıysa onu da kabul eder.
        Klasör verilirse bütün alt klasörler taranır.
        """

        raw_path = search_directory.strip()
        if not raw_path:
            return None

        root = Path(raw_path).expanduser()
        if not root.is_absolute():
            root = Path.cwd() / root
        root = root.resolve()

        if root.is_file():
            if (
                root.suffix.lower() == ".pt"
                and (
                    root.name == "last.pt"
                    or root.stem.startswith("epoch")
                )
            ):
                return root
            return None

        if not root.is_dir():
            return None

        candidates: list[Path] = []

        for path in root.rglob("*.pt"):
            if not path.is_file():
                continue
            if path.name == "last.pt" or path.stem.startswith("epoch"):
                candidates.append(path.resolve())

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda path: (
                path.stat().st_mtime,
                path.name == "last.pt",
            ),
        )

    @staticmethod
    def _validate_run_name(run_name: str) -> None:
        cleaned_name = run_name.strip()

        if not cleaned_name:
            raise ValueError("Eğitim çalışma adı boş bırakılamaz.")

        if "/" in cleaned_name or "\\" in cleaned_name:
            raise ValueError(
                "Eğitim çalışma adı klasör ayırıcı içeremez."
            )

        if cleaned_name in {".", ".."}:
            raise ValueError("Geçersiz eğitim çalışma adı.")

    @staticmethod
    def _validate_model_reference(model_path: str) -> None:
        cleaned_model = model_path.strip()

        if not cleaned_model:
            raise ValueError("Model alanı boş bırakılamaz.")

        candidate = Path(cleaned_model).expanduser()

        if candidate.exists():
            if not candidate.is_file():
                raise ValueError(
                    f"Seçilen model yolu dosya değil: {candidate}"
                )
            return

        has_directory_component = candidate.parent != Path(".")
        if has_directory_component:
            raise FileNotFoundError(
                f"Model dosyası bulunamadı: {candidate}"
            )

        if candidate.suffix.lower() not in {".pt", ".yaml", ".yml"}:
            raise ValueError(
                "Model bir .pt, .yaml veya .yml dosyası olmalıdır."
            )

        if candidate.suffix.lower() == ".pt" and "-pose" not in candidate.stem:
            raise ValueError(
                "İndirilecek resmi ağırlık bir pose modeli olmalıdır. "
                "Örnek: yolo11n-pose.pt"
            )

    @staticmethod
    def _resolve_dataset_root(
        *,
        yaml_path: Path,
        yaml_data: dict[str, Any],
    ) -> Path:
        configured_root = yaml_data.get("path")

        if configured_root is None or configured_root == "":
            return yaml_path.parent.resolve()

        root_path = Path(str(configured_root)).expanduser()

        if not root_path.is_absolute():
            root_path = yaml_path.parent / root_path

        return root_path.resolve()

    def _build_split_summary(
        self,
        *,
        configured_value: Any,
        dataset_root: Path,
    ) -> DatasetSplitSummary:
        if not isinstance(configured_value, (str, Path)):
            return DatasetSplitSummary(
                configured_value=str(configured_value),
                resolved_path=None,
                image_count=None,
            )

        configured_text = str(configured_value)
        configured_path = Path(configured_text).expanduser()

        if not configured_path.is_absolute():
            configured_path = dataset_root / configured_path

        resolved_path = configured_path.resolve()
        image_count = self._count_images(resolved_path)

        return DatasetSplitSummary(
            configured_value=configured_text,
            resolved_path=resolved_path,
            image_count=image_count,
        )

    @staticmethod
    def _count_images(path: Path) -> int | None:
        if path.is_dir():
            return sum(
                1
                for file_path in path.rglob("*")
                if file_path.is_file()
                and file_path.suffix.lower() in IMAGE_EXTENSIONS
            )

        if path.is_file() and path.suffix.lower() == ".txt":
            try:
                return sum(
                    1
                    for line in path.read_text(
                        encoding="utf-8"
                    ).splitlines()
                    if line.strip()
                )
            except OSError:
                return None

        return None

    @staticmethod
    def _append_split_warnings(
        *,
        split_name: str,
        summary: DatasetSplitSummary,
        warnings: list[str],
    ) -> None:
        if summary.resolved_path is None:
            warnings.append(
                f"{split_name} yolu liste veya özel yapı kullanıyor; "
                "dosya sayısı önceden doğrulanamadı."
            )
            return

        if not summary.resolved_path.exists():
            warnings.append(
                f"{split_name} yolu bulunamadı: {summary.resolved_path}"
            )
            return

        if summary.image_count == 0:
            warnings.append(
                f"{split_name} yolunda desteklenen görsel bulunamadı: "
                f"{summary.resolved_path}"
            )

    @staticmethod
    def _trainer_total_epochs(trainer: Any, *, fallback: int) -> int:
        trainer_epochs = getattr(trainer, "epochs", None)
        if trainer_epochs is not None:
            return int(trainer_epochs)

        trainer_args = getattr(trainer, "args", None)
        args_epochs = getattr(trainer_args, "epochs", None)
        if args_epochs is not None:
            return int(args_epochs)

        return fallback

    @staticmethod
    def _trainer_run_directory(trainer: Any | None) -> Path | None:
        if trainer is None:
            return None
        raw_save_dir = getattr(trainer, "save_dir", None)
        if raw_save_dir in {None, ""}:
            return None
        return Path(raw_save_dir).expanduser().resolve()

    @staticmethod
    def _trainer_last_checkpoint(trainer: Any | None) -> Path | None:
        if trainer is None:
            return None
        raw_last = getattr(trainer, "last", None)
        if raw_last not in {None, ""}:
            candidate = Path(raw_last).expanduser().resolve()
            if candidate.is_file():
                return candidate
        run_directory = PoseTrainingService._trainer_run_directory(trainer)
        if run_directory is not None:
            candidate = run_directory / "weights" / "last.pt"
            if candidate.is_file():
                return candidate.resolve()
        return None

    def _collect_trainer_metrics(self, trainer: Any) -> dict[str, float]:
        metrics = self._convert_metrics(getattr(trainer, "metrics", {}))

        validator = getattr(trainer, "validator", None)
        validator_metrics = getattr(validator, "metrics", None)
        results_dict = getattr(validator_metrics, "results_dict", None)
        metrics.update(self._convert_metrics(results_dict))

        label_loss_items = getattr(trainer, "label_loss_items", None)
        total_loss = getattr(trainer, "tloss", None)
        if callable(label_loss_items) and total_loss is not None:
            try:
                metrics.update(
                    self._convert_metrics(
                        label_loss_items(total_loss, prefix="train")
                    )
                )
            except Exception:
                pass

        csv_path = getattr(trainer, "csv", None)
        if csv_path not in {None, ""}:
            metrics.update(self._read_latest_results_row(Path(csv_path)))
        else:
            run_directory = self._trainer_run_directory(trainer)
            if run_directory is not None:
                metrics.update(
                    self._read_latest_results_row(run_directory / "results.csv")
                )
        return metrics

    @staticmethod
    def _read_latest_results_row(csv_path: Path) -> dict[str, float]:
        if not csv_path.is_file():
            return {}
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                rows = list(csv.DictReader(csv_file))
        except (OSError, csv.Error):
            return {}
        if not rows:
            return {}
        converted: dict[str, float] = {}
        for raw_name, raw_value in rows[-1].items():
            if raw_name is None or raw_value in {None, ""}:
                continue
            try:
                converted[str(raw_name).strip()] = float(str(raw_value).strip())
            except (TypeError, ValueError):
                continue
        return converted

    @staticmethod
    def _convert_metrics(raw_metrics: Any) -> dict[str, float]:
        if not isinstance(raw_metrics, dict):
            return {}

        converted: dict[str, float] = {}

        for metric_name, metric_value in raw_metrics.items():
            try:
                if hasattr(metric_value, "item"):
                    metric_value = metric_value.item()
                converted[str(metric_name)] = float(metric_value)
            except (TypeError, ValueError, RuntimeError):
                continue

        return converted

    @staticmethod
    def _format_metric_summary(metrics: dict[str, float]) -> str:
        preferred_keys = (
            "train/box_loss",
            "train/pose_loss",
            "train/kobj_loss",
            "train/cls_loss",
            "train/dfl_loss",
            "val/box_loss",
            "val/pose_loss",
            "val/kobj_loss",
            "val/cls_loss",
            "val/dfl_loss",
            "metrics/mAP50(P)",
            "metrics/mAP50-95(P)",
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
            "fitness",
        )

        parts: list[str] = []

        for key in preferred_keys:
            if key in metrics:
                short_name = key.replace("metrics/", "")
                parts.append(f"{short_name}={metrics[key]:.4f}")

        if not parts:
            for key, value in list(metrics.items())[:4]:
                parts.append(f"{key}={value:.4f}")

        return " | ".join(parts)

    @staticmethod
    def _existing_path(raw_path: Any) -> Path | None:
        if raw_path in {None, ""}:
            return None

        path = Path(raw_path).expanduser().resolve()
        return path if path.exists() else None

    def _is_stop_requested(self) -> bool:
        with self._state_lock:
            return self._stop_requested

    @staticmethod
    def _emit_log(
        callback: LogCallback | None,
        message: str,
    ) -> None:
        if callback is not None:
            callback(message)
