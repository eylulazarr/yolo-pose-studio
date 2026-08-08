from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter, sleep, strftime
from typing import Any, Callable

import cv2
import sys
import numpy as np
import torch
from ultralytics import YOLO


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}

BUILD_ID = "2026-07-30-model-test-v9-macos-camera"


VIDEO_EXTENSIONS = {
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".m4v",
    ".webm",
}

# Ultralytics/COCO 17 keypoint sırası için iskelet bağlantıları.
# Özel bir modelde keypoint sayısı 17 değilse servis güvenli biçimde
# ardışık noktaları bağlar.
COCO_POSE_EDGES = (
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
)

LogCallback = Callable[[str], None]
ProgressCallback = Callable[["InferenceProgress"], None]


@dataclass(frozen=True)
class InferenceSettings:
    """YOLO Pose inference ayarları."""

    model_path: str
    output_directory: str
    confidence: float = 0.25
    iou: float = 0.70
    image_size: int = 640
    device: str = "auto"
    keypoint_confidence: float = 0.25
    line_width: int = 2
    point_radius: int = 4
    show_boxes: bool = True
    show_labels: bool = True
    show_confidence: bool = True
    show_keypoints: bool = True
    show_skeleton: bool = True
    save_output: bool = True


@dataclass(frozen=True)
class InferenceProgress:
    """Arayüze gönderilen canlı inference ilerlemesi."""

    processed: int
    total: int | None
    percent: float | None
    detections: int
    inference_ms: float
    message: str
    preview_frame: np.ndarray | None = field(default=None, repr=False)


@dataclass(frozen=True)
class InferenceRunResult:
    """Bir inference çalışmasının toplu sonucu."""

    mode: str
    output_directory: Path
    output_paths: list[Path]
    processed_count: int
    total_detections: int
    average_inference_ms: float
    elapsed_seconds: float
    stopped: bool
    resolved_device: str
    last_preview_frame: np.ndarray | None = field(default=None, repr=False)


class _WebcamFrameReader:
    """Webcam'i tek bir daemon thread içinde açar, okur ve kapatır.

    macOS AVFoundation tarafında ``VideoCapture`` nesnesini bir thread'de açıp
    başka bir thread'de okumak güvenilir değildir. Bu sınıf kamera açma,
    ısınma, kare okuma ve normal kapatma işlemlerinin tamamını aynı thread'de
    gerçekleştirir. Stop sırasında bloklanmış bir ``read`` çağrısını çözmek
    için yalnızca acil ``release`` işlemi yardımcı bir daemon thread'e verilir.
    """

    _WARMUP_SECONDS = 6.0
    _WARMUP_RETRY_DELAY = 0.08
    _MAX_RUNTIME_READ_FAILURES = 30

    def __init__(self, camera_index: int) -> None:
        self.camera_index = int(camera_index)
        self._stop_event = Event()
        self._frame_ready = Event()
        self._startup_finished = Event()
        self._state_lock = Lock()
        self._latest_frame: np.ndarray | None = None
        self._sequence = 0
        self._failed = False
        self._error_message = ""
        self._backend_name = ""
        self._fps = 25.0
        self._capture: cv2.VideoCapture | None = None
        self._release_started = False
        self._thread = Thread(
            target=self._read_loop,
            name=f"webcam-reader-{self.camera_index}",
            daemon=True,
        )

    def start(self) -> None:
        self._thread.start()

    @staticmethod
    def _is_valid_frame(success: bool, frame: np.ndarray | None) -> bool:
        return bool(
            success
            and frame is not None
            and isinstance(frame, np.ndarray)
            and frame.size > 0
        )

    @staticmethod
    def _backend_candidates() -> list[tuple[int | None, str]]:
        candidates: list[tuple[int | None, str]] = []
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            candidates.append((int(cv2.CAP_AVFOUNDATION), "AVFoundation"))
        candidates.append((None, "Otomatik backend"))
        return candidates

    @staticmethod
    def _release_capture(capture: cv2.VideoCapture | None) -> None:
        if capture is None:
            return
        try:
            capture.release()
        except Exception:
            pass

    def _set_failure(self, message: str) -> None:
        with self._state_lock:
            self._failed = True
            self._error_message = message
        self._startup_finished.set()
        self._frame_ready.set()

    def _store_first_frame(
        self,
        *,
        capture: cv2.VideoCapture,
        frame: np.ndarray,
        backend_name: str,
    ) -> None:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or not np.isfinite(fps):
            fps = 25.0

        with self._state_lock:
            self._capture = capture
            self._backend_name = backend_name
            self._fps = fps
            self._latest_frame = frame.copy()
            self._sequence = 1
            self._failed = False
            self._error_message = ""

        self._startup_finished.set()
        self._frame_ready.set()

    def _open_and_warm_up(self) -> cv2.VideoCapture | None:
        errors: list[str] = []

        for backend, backend_name in self._backend_candidates():
            if self._stop_event.is_set():
                return None

            try:
                capture = (
                    cv2.VideoCapture(self.camera_index, backend)
                    if backend is not None
                    else cv2.VideoCapture(self.camera_index)
                )
            except Exception as error:
                errors.append(f"{backend_name}: açılamadı ({error})")
                continue

            if not capture.isOpened():
                errors.append(f"{backend_name}: kamera açılamadı")
                self._release_capture(capture)
                continue

            # macOS kameraları açıldıktan sonra birkaç başarısız/boş kare
            # döndürebilir. İlk başarısız okumada vazgeçmek yerine kameraya
            # ısınması için süre tanınır.
            warmup_deadline = perf_counter() + self._WARMUP_SECONDS
            read_attempts = 0
            last_error = "geçerli kare alınamadı"

            while (
                not self._stop_event.is_set()
                and perf_counter() < warmup_deadline
            ):
                read_attempts += 1
                try:
                    success, frame = capture.read()
                except Exception as error:
                    success, frame = False, None
                    last_error = str(error)

                if self._is_valid_frame(success, frame):
                    self._store_first_frame(
                        capture=capture,
                        frame=frame,
                        backend_name=backend_name,
                    )
                    return capture

                sleep(self._WARMUP_RETRY_DELAY)

            errors.append(
                f"{backend_name}: {read_attempts} denemede kare alınamadı "
                f"({last_error})"
            )
            self._release_capture(capture)

        if self._stop_event.is_set():
            return None

        detail = "; ".join(errors) if errors else "uygun backend bulunamadı"
        self._set_failure(
            f"Kamera {self.camera_index} açılamadı veya geçerli kare üretmedi. "
            f"Denenen backendler: {detail}. Kamera iznini ve kamerayı kullanan "
            "diğer uygulamaları kontrol edin."
        )
        return None

    def _read_loop(self) -> None:
        capture: cv2.VideoCapture | None = None
        try:
            capture = self._open_and_warm_up()
            if capture is None:
                self._startup_finished.set()
                self._frame_ready.set()
                return

            consecutive_failures = 0

            while not self._stop_event.is_set():
                try:
                    success, frame = capture.read()
                except Exception:
                    success, frame = False, None

                if not self._is_valid_frame(success, frame):
                    consecutive_failures += 1
                    if consecutive_failures < self._MAX_RUNTIME_READ_FAILURES:
                        sleep(0.03)
                        continue

                    if not self._stop_event.is_set():
                        self._set_failure(
                            f"Kamera {self.camera_index} akışı kesildi; art arda "
                            f"{consecutive_failures} kare okunamadı."
                        )
                    return

                consecutive_failures = 0
                with self._state_lock:
                    self._latest_frame = frame.copy()
                    self._sequence += 1
                self._frame_ready.set()
        finally:
            # Normal akışta kamerayı onu açan/okuyan reader thread kapatır.
            self._release_capture(capture)
            with self._state_lock:
                if self._capture is capture:
                    self._capture = None
            self._startup_finished.set()
            self._frame_ready.set()

    def wait_until_started(self, timeout: float = 0.10) -> bool:
        return self._startup_finished.wait(timeout)

    def get_latest(
        self,
        *,
        last_sequence: int,
        timeout: float = 0.10,
    ) -> tuple[np.ndarray | None, int]:
        self._frame_ready.wait(timeout)
        self._frame_ready.clear()
        with self._state_lock:
            if self._sequence == last_sequence or self._latest_frame is None:
                return None, last_sequence
            return self._latest_frame.copy(), self._sequence

    def has_failed(self) -> bool:
        with self._state_lock:
            return self._failed

    def error_message(self) -> str:
        with self._state_lock:
            return self._error_message

    def backend_name(self) -> str:
        with self._state_lock:
            return self._backend_name

    def fps(self) -> float:
        with self._state_lock:
            return self._fps

    def request_stop(self) -> None:
        self._stop_event.set()
        self._startup_finished.set()
        self._frame_ready.set()

        with self._state_lock:
            if self._release_started:
                return
            self._release_started = True
            capture = self._capture

        # capture.read() sürücü içinde bloklanırsa stop event tek başına yeterli
        # olmayabilir. Acil release UI thread'ini bekletmeden uygulanır.
        if capture is not None:
            Thread(
                target=self._release_capture,
                args=(capture,),
                name=f"webcam-release-{self.camera_index}",
                daemon=True,
            ).start()


class PoseInferenceService:
    """Fotoğraf, klasör, video ve webcam üzerinde YOLO Pose inference servisi."""

    def __init__(self) -> None:
        self._stop_event = Event()
        self._state_lock = Lock()
        self._model: YOLO | None = None
        self._model_reference: str | None = None
        self._active_capture: cv2.VideoCapture | None = None
        self._active_webcam_reader: _WebcamFrameReader | None = None
        self._stop_cleanup_started = Event()

    # ------------------------------------------------------------------
    # Genel durum yönetimi
    # ------------------------------------------------------------------

    def request_stop(self) -> None:
        """Stop isteğini UI thread'ini hiçbir kilitte bekletmeden gönderir."""

        # Event.set() anlıktır ve lock gerektirmez. Asıl hata önceki sürümde
        # bunun hemen ardından UI thread'inde _state_lock beklenmesiydi. Model
        # yüklenirken aynı lock tutulduğu için buton donmuş gibi görünüyordu.
        self._stop_event.set()

        if self._stop_cleanup_started.is_set():
            return
        self._stop_cleanup_started.set()
        Thread(target=self._interrupt_active_io, daemon=True).start()

    def _interrupt_active_io(self) -> None:
        """Aktif kamera/video kaynağını arka planda keser."""

        with self._state_lock:
            reader = self._active_webcam_reader
            capture = self._active_capture

        if reader is not None:
            reader.request_stop()
        elif capture is not None:
            self._safe_release_capture(capture)

    def reset_stop_request(self) -> None:
        self._stop_event.clear()
        self._stop_cleanup_started.clear()

    def is_stop_requested(self) -> bool:
        return self._stop_event.is_set()

    # ------------------------------------------------------------------
    # Doğrulama
    # ------------------------------------------------------------------

    def validate_settings(self, settings: InferenceSettings) -> str:
        model_reference = settings.model_path.strip()
        if not model_reference:
            raise ValueError("Model seçilmedi.")

        model_candidate = Path(model_reference).expanduser()
        if model_candidate.parent != Path(".") or model_candidate.is_absolute():
            if not model_candidate.is_file():
                raise FileNotFoundError(f"Model dosyası bulunamadı: {model_candidate}")
            if model_candidate.suffix.lower() != ".pt":
                raise ValueError("Inference modeli .pt dosyası olmalıdır.")
        elif model_candidate.suffix.lower() != ".pt":
            raise ValueError("Model alanı bir .pt dosyası veya resmi model adı olmalıdır.")

        if not 0.0 <= settings.confidence <= 1.0:
            raise ValueError("Confidence 0 ile 1 arasında olmalıdır.")
        if not 0.0 <= settings.iou <= 1.0:
            raise ValueError("IoU 0 ile 1 arasında olmalıdır.")
        if not 0.0 <= settings.keypoint_confidence <= 1.0:
            raise ValueError("Keypoint confidence 0 ile 1 arasında olmalıdır.")
        if settings.image_size <= 0:
            raise ValueError("Görüntü boyutu sıfırdan büyük olmalıdır.")
        if settings.line_width <= 0:
            raise ValueError("Çizgi kalınlığı sıfırdan büyük olmalıdır.")
        if settings.point_radius <= 0:
            raise ValueError("Nokta yarıçapı sıfırdan büyük olmalıdır.")

        output_directory = Path(settings.output_directory).expanduser()
        output_directory.mkdir(parents=True, exist_ok=True)
        if not output_directory.is_dir():
            raise NotADirectoryError(
                f"Inference çıktı yolu klasör değil: {output_directory}"
            )

        return self.resolve_device(settings.device)

    @staticmethod
    def resolve_device(device: str) -> str:
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
                raise RuntimeError("MPS seçildi ancak bu ortamda kullanılamıyor.")
            return "mps"

        if normalized in {"cuda", "gpu"}:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA seçildi ancak kullanılabilir NVIDIA GPU yok.")
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
            "Device değeri auto, mps, cpu, cuda veya CUDA cihaz numarası olmalıdır."
        )

    @staticmethod
    def validate_source(mode: str, source_path: str, camera_index: int = 0) -> None:
        if mode == "webcam":
            if camera_index < 0:
                raise ValueError("Webcam numarası negatif olamaz.")
            return

        cleaned_source = source_path.strip()
        if not cleaned_source:
            raise ValueError("Test kaynağı seçilmedi.")

        source = Path(cleaned_source).expanduser()
        if mode == "image":
            if not source.is_file():
                raise FileNotFoundError(f"Görsel bulunamadı: {source}")
            if source.suffix.lower() not in IMAGE_EXTENSIONS:
                raise ValueError("Seçilen dosya desteklenen bir görsel değil.")
            return

        if mode == "directory":
            if not source.is_dir():
                raise NotADirectoryError(f"Görsel klasörü bulunamadı: {source}")
            if not any(
                path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
                for path in source.rglob("*")
            ):
                raise ValueError("Seçilen klasörde desteklenen görsel bulunamadı.")
            return

        if mode == "video":
            if not source.is_file():
                raise FileNotFoundError(f"Video bulunamadı: {source}")
            if source.suffix.lower() not in VIDEO_EXTENSIONS:
                raise ValueError("Seçilen dosya desteklenen bir video değil.")
            return

        raise ValueError(f"Bilinmeyen inference modu: {mode}")

    # ------------------------------------------------------------------
    # Ana çalışma noktası
    # ------------------------------------------------------------------

    def run(
        self,
        *,
        mode: str,
        settings: InferenceSettings,
        source_path: str = "",
        camera_index: int = 0,
        log_callback: LogCallback | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> InferenceRunResult:
        resolved_device = self.validate_settings(settings)
        self.validate_source(mode, source_path, camera_index)

        self._emit_log(log_callback, f"PoseInferenceService build: {BUILD_ID}")
        self._emit_log(log_callback, "Inference ayarları doğrulandı.")
        self._emit_log(log_callback, f"Cihaz: {resolved_device}")
        self._emit_log(log_callback, f"Model: {settings.model_path}")
        if mode == "webcam":
            self._emit_log(log_callback, f"İstenen webcam numarası: {camera_index}")

        # Model yüklenirken kullanıcı durdurduysa kamerayı açmaya geçme.
        if self.is_stop_requested():
            raise RuntimeError("Model testi kullanıcı tarafından başlatma sırasında durduruldu.")

        model = self._get_model(settings.model_path, log_callback)

        # Model yükleme uzun sürebilir. Kullanıcı bu sırada durdurduysa kamera veya
        # video kaynağı açılmadan temiz bir "stopped" sonucu döndürülür.
        if self.is_stop_requested():
            return self._build_cancelled_result(
                mode=mode,
                settings=settings,
                resolved_device=resolved_device,
            )

        if mode == "image":
            return self._run_image(
                model=model,
                source_path=source_path,
                settings=settings,
                resolved_device=resolved_device,
                log_callback=log_callback,
                progress_callback=progress_callback,
            )

        if mode == "directory":
            return self._run_directory(
                model=model,
                source_path=source_path,
                settings=settings,
                resolved_device=resolved_device,
                log_callback=log_callback,
                progress_callback=progress_callback,
            )

        if mode == "video":
            return self._run_video(
                model=model,
                source_path=source_path,
                settings=settings,
                resolved_device=resolved_device,
                log_callback=log_callback,
                progress_callback=progress_callback,
            )

        if mode == "webcam":
            return self._run_webcam(
                model=model,
                camera_index=camera_index,
                settings=settings,
                resolved_device=resolved_device,
                log_callback=log_callback,
                progress_callback=progress_callback,
            )

        raise ValueError(f"Desteklenmeyen inference modu: {mode}")

    # ------------------------------------------------------------------
    # Modlar
    # ------------------------------------------------------------------

    def _run_image(
        self,
        *,
        model: YOLO,
        source_path: str,
        settings: InferenceSettings,
        resolved_device: str,
        log_callback: LogCallback | None,
        progress_callback: ProgressCallback | None,
    ) -> InferenceRunResult:
        source = Path(source_path).expanduser().resolve()
        frame = cv2.imread(str(source))
        if frame is None:
            raise ValueError(f"Görsel OpenCV ile okunamadı: {source}")

        output_directory = self._prepare_output_directory(
            settings.output_directory,
            "image_test",
        )
        started_at = perf_counter()

        annotated, detections, inference_ms = self._predict_and_draw(
            model=model,
            frame=frame,
            settings=settings,
            resolved_device=resolved_device,
        )

        output_paths: list[Path] = []
        if settings.save_output:
            output_path = output_directory / f"{source.stem}_pose_pred{source.suffix}"
            self._write_image(output_path, annotated)
            output_paths.append(output_path)
            self._emit_log(log_callback, f"Sonuç kaydedildi: {output_path}")

        elapsed_seconds = perf_counter() - started_at
        message = (
            f"Görsel tamamlandı — tespit: {detections}, "
            f"inference: {inference_ms:.2f} ms"
        )
        self._emit_log(log_callback, message)
        self._emit_progress(
            progress_callback,
            processed=1,
            total=1,
            detections=detections,
            inference_ms=inference_ms,
            message=message,
            preview_frame=annotated,
        )

        return InferenceRunResult(
            mode="image",
            output_directory=output_directory,
            output_paths=output_paths,
            processed_count=1,
            total_detections=detections,
            average_inference_ms=inference_ms,
            elapsed_seconds=elapsed_seconds,
            stopped=False,
            resolved_device=resolved_device,
            last_preview_frame=annotated,
        )

    def _run_directory(
        self,
        *,
        model: YOLO,
        source_path: str,
        settings: InferenceSettings,
        resolved_device: str,
        log_callback: LogCallback | None,
        progress_callback: ProgressCallback | None,
    ) -> InferenceRunResult:
        source_directory = Path(source_path).expanduser().resolve()
        image_paths = sorted(
            path
            for path in source_directory.rglob("*")
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        output_directory = self._prepare_output_directory(
            settings.output_directory,
            "folder_test",
        )

        started_at = perf_counter()
        output_paths: list[Path] = []
        inference_times: list[float] = []
        total_detections = 0
        processed_count = 0
        last_preview: np.ndarray | None = None

        for index, image_path in enumerate(image_paths, start=1):
            if self.is_stop_requested():
                self._emit_log(log_callback, "Klasör testi kullanıcı isteğiyle durduruldu.")
                break

            frame = cv2.imread(str(image_path))
            if frame is None:
                self._emit_log(log_callback, f"ATLANDI: okunamadı: {image_path}")
                continue

            annotated, detections, inference_ms = self._predict_and_draw(
                model=model,
                frame=frame,
                settings=settings,
                resolved_device=resolved_device,
            )

            processed_count += 1
            total_detections += detections
            inference_times.append(inference_ms)
            last_preview = annotated

            if settings.save_output:
                relative_parent = image_path.parent.relative_to(source_directory)
                destination_directory = output_directory / relative_parent
                destination_directory.mkdir(parents=True, exist_ok=True)
                output_path = (
                    destination_directory
                    / f"{image_path.stem}_pose_pred{image_path.suffix}"
                )
                self._write_image(output_path, annotated)
                output_paths.append(output_path)

            message = (
                f"{index}/{len(image_paths)} — {image_path.name} — "
                f"tespit: {detections}, {inference_ms:.2f} ms"
            )
            self._emit_log(log_callback, message)
            self._emit_progress(
                progress_callback,
                processed=index,
                total=len(image_paths),
                detections=total_detections,
                inference_ms=inference_ms,
                message=message,
                preview_frame=annotated,
            )

        elapsed_seconds = perf_counter() - started_at
        average_inference_ms = (
            sum(inference_times) / len(inference_times) if inference_times else 0.0
        )

        return InferenceRunResult(
            mode="directory",
            output_directory=output_directory,
            output_paths=output_paths,
            processed_count=processed_count,
            total_detections=total_detections,
            average_inference_ms=average_inference_ms,
            elapsed_seconds=elapsed_seconds,
            stopped=self.is_stop_requested(),
            resolved_device=resolved_device,
            last_preview_frame=last_preview,
        )

    def _run_video(
        self,
        *,
        model: YOLO,
        source_path: str,
        settings: InferenceSettings,
        resolved_device: str,
        log_callback: LogCallback | None,
        progress_callback: ProgressCallback | None,
    ) -> InferenceRunResult:
        source = Path(source_path).expanduser().resolve()
        output_directory = self._prepare_output_directory(
            settings.output_directory,
            "video_test",
        )
        output_path = output_directory / f"{source.stem}_pose_pred.mp4"

        capture = cv2.VideoCapture(str(source))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Video açılamadı: {source}")
        self._set_active_capture(capture)

        total_frames_raw = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_frames = total_frames_raw if total_frames_raw > 0 else None
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if fps <= 0 or not np.isfinite(fps):
            fps = 25.0
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer: cv2.VideoWriter | None = None
        if settings.save_output:
            writer = self._create_video_writer(output_path, fps, width, height)

        started_at = perf_counter()
        inference_times: list[float] = []
        processed_count = 0
        total_detections = 0
        last_preview: np.ndarray | None = None

        try:
            while capture.isOpened():
                if self.is_stop_requested():
                    self._emit_log(log_callback, "Video testi durdurma isteği alındı.")
                    break

                success, frame = capture.read()
                if not success:
                    break

                annotated, detections, inference_ms = self._predict_and_draw(
                    model=model,
                    frame=frame,
                    settings=settings,
                    resolved_device=resolved_device,
                )

                processed_count += 1
                total_detections += detections
                inference_times.append(inference_ms)
                last_preview = annotated

                if writer is not None:
                    writer.write(annotated)

                # UI yükünü azaltmak için yaklaşık her 3 karede bir preview gönder.
                preview = annotated if processed_count == 1 or processed_count % 3 == 0 else None
                message = self._video_progress_message(
                    processed_count,
                    total_frames,
                    detections,
                    inference_ms,
                )
                self._emit_progress(
                    progress_callback,
                    processed=processed_count,
                    total=total_frames,
                    detections=total_detections,
                    inference_ms=inference_ms,
                    message=message,
                    preview_frame=preview,
                )

                if processed_count == 1 or processed_count % 30 == 0:
                    self._emit_log(log_callback, message)
        finally:
            try:
                capture.release()
            finally:
                self._clear_active_capture(capture)
            if writer is not None:
                writer.release()

        elapsed_seconds = perf_counter() - started_at
        average_inference_ms = (
            sum(inference_times) / len(inference_times) if inference_times else 0.0
        )
        output_paths = [output_path] if settings.save_output and output_path.is_file() else []

        if output_paths:
            self._emit_log(log_callback, f"İşlenmiş video kaydedildi: {output_path}")

        return InferenceRunResult(
            mode="video",
            output_directory=output_directory,
            output_paths=output_paths,
            processed_count=processed_count,
            total_detections=total_detections,
            average_inference_ms=average_inference_ms,
            elapsed_seconds=elapsed_seconds,
            stopped=self.is_stop_requested(),
            resolved_device=resolved_device,
            last_preview_frame=last_preview,
        )

    def _run_webcam(
        self,
        *,
        model: YOLO,
        camera_index: int,
        settings: InferenceSettings,
        resolved_device: str,
        log_callback: LogCallback | None,
        progress_callback: ProgressCallback | None,
    ) -> InferenceRunResult:
        output_directory = self._prepare_output_directory(
            settings.output_directory,
            "webcam_test",
        )
        output_path = output_directory / f"webcam_pose_{strftime('%Y%m%d_%H%M%S')}.mp4"

        reader = _WebcamFrameReader(camera_index)
        self._set_active_webcam_reader(reader)
        reader.start()

        writer: cv2.VideoWriter | None = None
        started_at = perf_counter()
        startup_deadline = started_at + 15.0
        inference_times: list[float] = []
        processed_count = 0
        total_detections = 0
        last_preview: np.ndarray | None = None
        last_ui_emit_at = 0.0
        last_sequence = 0
        ui_emit_interval_seconds = 0.75
        startup_logged = False

        self._emit_log(
            log_callback,
            f"Kamera {camera_index} hazırlanıyor. İlk kare bekleniyor...",
        )

        try:
            while not self.is_stop_requested():
                frame, last_sequence = reader.get_latest(
                    last_sequence=last_sequence,
                    timeout=0.10,
                )

                if self.is_stop_requested():
                    break

                if frame is None:
                    if reader.has_failed():
                        raise RuntimeError(
                            reader.error_message()
                            or f"Kamera {camera_index} geçerli kare üretemedi."
                        )
                    if processed_count == 0 and perf_counter() > startup_deadline:
                        raise RuntimeError(
                            f"Kamera {camera_index} ilk kareyi 15 saniye içinde "
                            "vermedi. Kamera iznini ve kamerayı kullanan diğer "
                            "uygulamaları kontrol edin."
                        )
                    continue

                if not startup_logged:
                    backend_name = reader.backend_name() or "bilinmeyen backend"
                    self._emit_log(
                        log_callback,
                        f"Webcam başladı. Kamera: {camera_index}, backend: "
                        f"{backend_name}. Durdur butonuyla kapatabilirsiniz.",
                    )
                    startup_logged = True

                annotated, detections, inference_ms = self._predict_and_draw(
                    model=model,
                    frame=frame,
                    settings=settings,
                    resolved_device=resolved_device,
                )

                if self.is_stop_requested():
                    break

                processed_count += 1
                total_detections += detections
                inference_times.append(inference_ms)
                last_preview = annotated

                if settings.save_output and writer is None:
                    height, width = annotated.shape[:2]
                    try:
                        writer = self._create_video_writer(
                            output_path,
                            reader.fps(),
                            width,
                            height,
                        )
                    except RuntimeError as error:
                        self._emit_log(
                            log_callback,
                            "UYARI: Webcam kaydı açılamadı; canlı test kayıtsız "
                            f"devam ediyor: {error}",
                        )

                if writer is not None:
                    writer.write(annotated)

                effective_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
                message = (
                    f"Webcam kare {processed_count} — tespit: {detections}, "
                    f"{inference_ms:.2f} ms, yaklaşık FPS: {effective_fps:.1f}"
                )
                now = perf_counter()
                if processed_count == 1 or now - last_ui_emit_at >= ui_emit_interval_seconds:
                    self._emit_progress(
                        progress_callback,
                        processed=processed_count,
                        total=None,
                        detections=total_detections,
                        inference_ms=inference_ms,
                        message=message,
                        preview_frame=self._prepare_preview_frame(annotated),
                    )
                    last_ui_emit_at = now

                if processed_count == 1 or processed_count % 60 == 0:
                    self._emit_log(log_callback, message)
        finally:
            reader.request_stop()
            self._clear_active_webcam_reader(reader)
            if writer is not None:
                writer.release()

        elapsed_seconds = perf_counter() - started_at
        average_inference_ms = (
            sum(inference_times) / len(inference_times) if inference_times else 0.0
        )
        output_paths = (
            [output_path]
            if settings.save_output and output_path.is_file()
            else []
        )

        if output_paths:
            self._emit_log(log_callback, f"Webcam kaydı oluşturuldu: {output_path}")

        return InferenceRunResult(
            mode="webcam",
            output_directory=output_directory,
            output_paths=output_paths,
            processed_count=processed_count,
            total_detections=total_detections,
            average_inference_ms=average_inference_ms,
            elapsed_seconds=elapsed_seconds,
            stopped=self.is_stop_requested(),
            resolved_device=resolved_device,
            last_preview_frame=last_preview,
        )

    @staticmethod
    def _prepare_preview_frame(frame: np.ndarray) -> np.ndarray:
        """UI kuyruğunu doldurmamak için canlı preview karesini küçültüp kopyalar."""

        height, width = frame.shape[:2]
        max_width = 960
        max_height = 540
        scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
        if scale < 1.0:
            frame = cv2.resize(
                frame,
                (max(1, int(width * scale)), max(1, int(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        return np.ascontiguousarray(frame).copy()

    def _set_active_capture(self, capture: cv2.VideoCapture) -> None:
        with self._state_lock:
            self._active_capture = capture

    def _clear_active_capture(self, capture: cv2.VideoCapture) -> None:
        with self._state_lock:
            if self._active_capture is capture:
                self._active_capture = None

    def _set_active_webcam_reader(self, reader: _WebcamFrameReader) -> None:
        with self._state_lock:
            self._active_webcam_reader = reader

    def _clear_active_webcam_reader(self, reader: _WebcamFrameReader) -> None:
        with self._state_lock:
            if self._active_webcam_reader is reader:
                self._active_webcam_reader = None

    @staticmethod
    def _safe_release_capture(capture: cv2.VideoCapture) -> None:
        try:
            capture.release()
        except Exception:
            pass

    def _create_webcam_capture(self, camera_index: int) -> cv2.VideoCapture | None:
        backends: list[int | None] = []
        if sys.platform == "darwin" and hasattr(cv2, "CAP_AVFOUNDATION"):
            backends.append(int(cv2.CAP_AVFOUNDATION))
        backends.append(None)

        for backend in backends:
            capture = (
                cv2.VideoCapture(camera_index, backend)
                if backend is not None
                else cv2.VideoCapture(camera_index)
            )
            if capture.isOpened():
                try:
                    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                except Exception:
                    pass
                return capture
            self._safe_release_capture(capture)
        return None

    def _open_webcam_capture(self, camera_index: int) -> cv2.VideoCapture | None:
        """Seçilen kamerayı açar; kare okuma ayrı reader thread'inde yapılır."""

        if camera_index < 0:
            raise ValueError("Webcam numarası negatif olamaz.")
        if self.is_stop_requested():
            return None

        capture = self._create_webcam_capture(camera_index)
        if capture is not None:
            return capture

        raise RuntimeError(
            f"Webcam açılamadı (kamera numarası: {camera_index}). "
            "macOS Sistem Ayarları > Gizlilik ve Güvenlik > Kamera bölümünde "
            "Terminal/Python iznini kontrol edin."
        )

    # ------------------------------------------------------------------
    # Model ve çizim yardımcıları
    # ------------------------------------------------------------------

    def _get_model(
        self,
        model_reference: str,
        log_callback: LogCallback | None,
    ) -> YOLO:
        normalized_reference = self._normalized_model_reference(model_reference)

        # Cache kontrolü kısa bir lock içinde yapılır. YOLO(...) yüklemesi lock
        # dışında çalışır; böylece kullanıcı model yüklenirken bile stop'a
        # bastığında UI thread'i anında döner.
        with self._state_lock:
            cached_model = (
                self._model
                if self._model is not None
                and self._model_reference == normalized_reference
                else None
            )

        if cached_model is not None:
            self._emit_log(log_callback, "Önceden yüklenen model kullanılıyor.")
            return cached_model

        self._emit_log(log_callback, "Model belleğe yükleniyor...")
        loaded_model = YOLO(normalized_reference)

        with self._state_lock:
            if (
                self._model is not None
                and self._model_reference == normalized_reference
            ):
                model = self._model
            else:
                self._model = loaded_model
                self._model_reference = normalized_reference
                model = loaded_model

        self._emit_log(log_callback, "Model başarıyla yüklendi.")
        return model

    @staticmethod
    def _normalized_model_reference(model_reference: str) -> str:
        candidate = Path(model_reference).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        return model_reference.strip()

    def _predict_and_draw(
        self,
        *,
        model: YOLO,
        frame: np.ndarray,
        settings: InferenceSettings,
        resolved_device: str,
    ) -> tuple[np.ndarray, int, float]:
        results = model.predict(
            source=frame,
            conf=settings.confidence,
            iou=settings.iou,
            imgsz=settings.image_size,
            device=resolved_device,
            verbose=False,
            save=False,
        )

        if not results:
            return frame.copy(), 0, 0.0

        result = results[0]
        inference_ms = self._extract_inference_ms(result)
        annotated = self._draw_result(
            frame=frame,
            result=result,
            model=model,
            settings=settings,
        )
        detections = self._detection_count(result)
        return annotated, detections, inference_ms

    def _draw_result(
        self,
        *,
        frame: np.ndarray,
        result: Any,
        model: YOLO,
        settings: InferenceSettings,
    ) -> np.ndarray:
        annotated = frame.copy()
        boxes = getattr(result, "boxes", None)
        keypoints = getattr(result, "keypoints", None)
        names = getattr(result, "names", None) or getattr(model, "names", {})

        if boxes is not None and len(boxes) > 0:
            xyxy = self._to_numpy(getattr(boxes, "xyxy", None))
            confidences = self._to_numpy(getattr(boxes, "conf", None))
            classes = self._to_numpy(getattr(boxes, "cls", None))

            if xyxy is not None:
                for index, coordinates in enumerate(xyxy):
                    x1, y1, x2, y2 = (int(round(value)) for value in coordinates[:4])
                    class_id = (
                        int(classes[index])
                        if classes is not None and index < len(classes)
                        else 0
                    )
                    confidence = (
                        float(confidences[index])
                        if confidences is not None and index < len(confidences)
                        else 0.0
                    )
                    color = self._class_color(class_id)

                    if settings.show_boxes:
                        cv2.rectangle(
                            annotated,
                            (x1, y1),
                            (x2, y2),
                            color,
                            settings.line_width,
                            cv2.LINE_AA,
                        )

                    label_parts: list[str] = []
                    if settings.show_labels:
                        label_parts.append(self._class_name(names, class_id))
                    if settings.show_confidence:
                        label_parts.append(f"{confidence:.2f}")

                    if label_parts and settings.show_boxes:
                        self._draw_label(
                            annotated,
                            x=x1,
                            y=y1,
                            text=" ".join(label_parts),
                            color=color,
                            line_width=settings.line_width,
                        )

        if settings.show_keypoints and keypoints is not None:
            keypoint_data = self._to_numpy(getattr(keypoints, "data", None))
            if keypoint_data is not None:
                for person_index, person_keypoints in enumerate(keypoint_data):
                    color = self._class_color(person_index)
                    visible: dict[int, tuple[int, int]] = {}

                    for keypoint_index, values in enumerate(person_keypoints):
                        if len(values) < 2:
                            continue
                        x = int(round(float(values[0])))
                        y = int(round(float(values[1])))
                        score = float(values[2]) if len(values) >= 3 else 1.0
                        if score < settings.keypoint_confidence or x <= 0 or y <= 0:
                            continue

                        visible[keypoint_index] = (x, y)
                        cv2.circle(
                            annotated,
                            (x, y),
                            settings.point_radius,
                            color,
                            -1,
                            cv2.LINE_AA,
                        )

                    if settings.show_skeleton:
                        edges = self._skeleton_edges(len(person_keypoints))
                        for first, second in edges:
                            if first not in visible or second not in visible:
                                continue
                            cv2.line(
                                annotated,
                                visible[first],
                                visible[second],
                                color,
                                settings.line_width,
                                cv2.LINE_AA,
                            )

        return annotated

    @staticmethod
    def _skeleton_edges(keypoint_count: int) -> tuple[tuple[int, int], ...]:
        if keypoint_count == 17:
            return COCO_POSE_EDGES
        if keypoint_count <= 1:
            return ()
        return tuple((index, index + 1) for index in range(keypoint_count - 1))

    @staticmethod
    def _draw_label(
        image: np.ndarray,
        *,
        x: int,
        y: int,
        text: str,
        color: tuple[int, int, int],
        line_width: int,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = max(0.45, line_width * 0.22)
        thickness = max(1, line_width - 1)
        (text_width, text_height), baseline = cv2.getTextSize(
            text,
            font,
            font_scale,
            thickness,
        )
        top = max(0, y - text_height - baseline - 8)
        cv2.rectangle(
            image,
            (x, top),
            (x + text_width + 8, y),
            color,
            -1,
        )
        cv2.putText(
            image,
            text,
            (x + 4, max(text_height + 2, y - baseline - 4)),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _class_color(class_id: int) -> tuple[int, int, int]:
        palette = (
            (56, 180, 75),
            (255, 127, 14),
            (31, 119, 180),
            (214, 39, 40),
            (148, 103, 189),
            (23, 190, 207),
            (227, 119, 194),
            (188, 189, 34),
        )
        return palette[class_id % len(palette)]

    @staticmethod
    def _class_name(names: Any, class_id: int) -> str:
        if isinstance(names, dict):
            return str(names.get(class_id, class_id))
        if isinstance(names, (list, tuple)) and 0 <= class_id < len(names):
            return str(names[class_id])
        return str(class_id)

    @staticmethod
    def _to_numpy(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            if hasattr(value, "detach"):
                value = value.detach()
            if hasattr(value, "cpu"):
                value = value.cpu()
            if hasattr(value, "numpy"):
                value = value.numpy()
            return np.asarray(value)
        except (TypeError, ValueError, RuntimeError):
            return None

    @staticmethod
    def _detection_count(result: Any) -> int:
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return 0
        try:
            return len(boxes)
        except TypeError:
            return 0

    @staticmethod
    def _extract_inference_ms(result: Any) -> float:
        speed = getattr(result, "speed", None)
        if isinstance(speed, dict):
            try:
                return float(speed.get("inference", 0.0))
            except (TypeError, ValueError):
                return 0.0
        return 0.0

    # ------------------------------------------------------------------
    # Dosya ve callback yardımcıları
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_output_directory(raw_output: str, prefix: str) -> Path:
        root = Path(raw_output).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        run_directory = root / f"{prefix}_{strftime('%Y%m%d_%H%M%S')}"
        counter = 2
        original = run_directory
        while run_directory.exists():
            run_directory = Path(f"{original}_{counter}")
            counter += 1
        run_directory.mkdir(parents=True, exist_ok=False)
        return run_directory

    @staticmethod
    def _write_image(output_path: Path, image: np.ndarray) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(output_path), image)
        if not success:
            raise OSError(f"Görsel kaydedilemedi: {output_path}")

    @staticmethod
    def _create_video_writer(
        output_path: Path,
        fps: float,
        width: int,
        height: int,
    ) -> cv2.VideoWriter:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Video writer oluşturulamadı: {output_path}")
        return writer

    @staticmethod
    def _video_progress_message(
        processed: int,
        total: int | None,
        detections: int,
        inference_ms: float,
    ) -> str:
        total_text = str(total) if total is not None else "?"
        effective_fps = 1000.0 / inference_ms if inference_ms > 0 else 0.0
        return (
            f"Video kare {processed}/{total_text} — tespit: {detections}, "
            f"{inference_ms:.2f} ms, yaklaşık FPS: {effective_fps:.1f}"
        )

    @staticmethod
    def _emit_log(callback: LogCallback | None, message: str) -> None:
        if callback is not None:
            callback(message)

    @staticmethod
    def _emit_progress(
        callback: ProgressCallback | None,
        *,
        processed: int,
        total: int | None,
        detections: int,
        inference_ms: float,
        message: str,
        preview_frame: np.ndarray | None,
    ) -> None:
        if callback is None:
            return

        percent = None
        if total is not None and total > 0:
            percent = min(100.0, (processed / total) * 100.0)

        callback(
            InferenceProgress(
                processed=processed,
                total=total,
                percent=percent,
                detections=detections,
                inference_ms=inference_ms,
                message=message,
                preview_frame=preview_frame,
            )
        )
