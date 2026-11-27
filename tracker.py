"""Low-latency webcam hand tracker built on OpenCV and MediaPipe.

Supports both the modern MediaPipe Tasks HandLandmarker API (MediaPipe >= 0.10.30,
including Python 3.13-3.15 wheels) and the legacy ``mp.solutions.hands`` API.
Returns normalized 3D landmarks (x, y, z) labeled by physical handedness
(``Left`` and ``Right``) aligned with the returned frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any, Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import mediapipe as mp
import numpy as np

HandLabel = Literal["Left", "Right"]

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
FALLBACK_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


@dataclass(frozen=True, slots=True)
class TrackedHand:
    """Normalized 3D landmarks for a single detected hand."""

    handedness: HandLabel
    score: float
    landmarks: np.ndarray  # Shape: (21, 3), float32 normalized (x, y, z)

    @property
    def wrist(self) -> np.ndarray:
        """Return the normalized (x, y, z) coordinates of the wrist (landmark 0)."""
        return self.landmarks[0]

    def to_pixel_xy(self, frame_width: int, frame_height: int) -> np.ndarray:
        """Convert normalized (x, y) landmarks into integer pixel coordinates."""
        scale = np.array([float(frame_width), float(frame_height)], dtype=np.float32)
        pts = self.landmarks[:, :2] * scale
        return np.round(pts).astype(np.int32)


@dataclass(frozen=True, slots=True)
class DetectionPacket:
    """Latest frame and tracked hands published by the async tracker worker."""

    seq: int
    frame: np.ndarray
    hands: dict[HandLabel, TrackedHand]
    timestamp: float


def _download_model(url: str, destination: Path, timeout: float = 30.0) -> None:
    """Download the MediaPipe hand_landmarker.task asset atomically."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    part_path = destination.with_name(f"{destination.name}.part")
    request = Request(url, headers={"User-Agent": "FH4-Hand-Controller/1.0"})

    try:
        with urlopen(request, timeout=timeout) as response:
            content_type = (response.headers.get("Content-Type") or "").lower()
            if "text/html" in content_type:
                raise RuntimeError(f"Unexpected HTML response downloading model from {url}")
            downloaded = 0
            with part_path.open("wb") as handle:
                while True:
                    chunk = response.read(64 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
        if downloaded < 100_000:
            raise RuntimeError(
                f"Downloaded model from {url} is unexpectedly small ({downloaded} bytes)."
            )
        os.replace(part_path, destination)
    except Exception:
        if part_path.exists():
            part_path.unlink(missing_ok=True)
        raise


def ensure_hand_landmarker_model(model_path: Path) -> Path:
    """Ensure ``hand_landmarker.task`` exists locally, downloading it if absent."""
    resolved = Path(model_path)
    if resolved.is_file() and resolved.stat().st_size > 100_000:
        return resolved

    last_error: Exception | None = None
    for url in (DEFAULT_MODEL_URL, FALLBACK_MODEL_URL):
        try:
            print(f"[Tracker] Downloading MediaPipe HandLandmarker model to {resolved}...")
            _download_model(url, resolved)
            return resolved
        except (HTTPError, URLError, OSError, RuntimeError) as exc:
            last_error = exc

    raise RuntimeError(
        f"Could not find or download MediaPipe hand_landmarker.task at '{resolved}'. "
        f"Download it from {DEFAULT_MODEL_URL} and save it there. Last error: {last_error}"
    )


def _normalize_label(raw_label: str, swap_handedness: bool = False) -> HandLabel | None:
    """Normalize MediaPipe handedness labels to 'Left' or 'Right'."""
    cleaned = (raw_label or "").strip().lower()
    if cleaned not in {"left", "right"}:
        return None
    if swap_handedness:
        return "Right" if cleaned == "left" else "Left"
    return "Left" if cleaned == "left" else "Right"


class HandTracker:
    """Low-latency OpenCV + MediaPipe hand landmark tracker."""

    def __init__(
        self,
        camera_index: int = 0,
        frame_width: int = 640,
        frame_height: int = 480,
        target_fps: int = 60,
        mirror: bool = True,
        swap_handedness: bool = False,
        max_num_hands: int = 2,
        model_complexity: int = 0,
        min_detection_confidence: float = 0.55,
        min_presence_confidence: float = 0.55,
        min_tracking_confidence: float = 0.55,
        delegate: str = "auto",
        model_path: str | Path = "models/hand_landmarker.task",
        base_dir: str | Path | None = None,
        open_camera: bool = True,
    ) -> None:
        self.camera_index = int(camera_index)
        self.frame_width = int(frame_width)
        self.frame_height = int(frame_height)
        self.target_fps = int(target_fps)
        self.mirror = bool(mirror)
        self.swap_handedness = bool(swap_handedness)
        self.max_num_hands = int(max_num_hands)
        self.model_complexity = int(model_complexity)
        self.min_detection_confidence = float(min_detection_confidence)
        self.min_presence_confidence = float(min_presence_confidence)
        self.min_tracking_confidence = float(min_tracking_confidence)
        self.delegate = str(delegate)

        raw_model_path = Path(model_path)
        if not raw_model_path.is_absolute() and base_dir is not None:
            raw_model_path = Path(base_dir) / raw_model_path
        self.model_path = raw_model_path

        self._cap: cv2.VideoCapture | None = None
        self._last_timestamp_ms = -1
        self._backend: str = ""
        self._solutions_hands: Any = None
        self._landmarker: Any = None

        # Async capture + inference pipeline state
        self._stop_event = threading.Event()
        self._raw_cv = threading.Condition()
        self._raw_seq = 0
        self._raw_frame: np.ndarray | None = None
        self._raw_ts = 0.0
        self._result_lock = threading.Lock()
        self._latest_packet: DetectionPacket | None = None
        self._capture_thread: threading.Thread | None = None
        self._infer_thread: threading.Thread | None = None

        self._init_mediapipe()
        if open_camera:
            self.open()

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        base_dir: str | Path | None = None,
        camera_override: int | None = None,
        open_camera: bool = True,
    ) -> HandTracker:
        """Instantiate a HandTracker from parsed ``config.json`` values."""
        cam = config.get("camera", {})
        trk = config.get("tracker", {})
        return cls(
            camera_index=cam.get("index", 0) if camera_override is None else camera_override,
            frame_width=cam.get("frame_width", 640),
            frame_height=cam.get("frame_height", 480),
            target_fps=cam.get("target_fps", 60),
            mirror=cam.get("mirror", True),
            swap_handedness=cam.get("swap_handedness", False),
            max_num_hands=trk.get("max_num_hands", 2),
            model_complexity=trk.get("model_complexity", 0),
            min_detection_confidence=trk.get("min_detection_confidence", 0.55),
            min_presence_confidence=trk.get("min_presence_confidence", 0.55),
            min_tracking_confidence=trk.get("min_tracking_confidence", 0.55),
            delegate=trk.get("delegate", "auto"),
            model_path=trk.get("model_path", "models/hand_landmarker.task"),
            base_dir=base_dir,
            open_camera=open_camera,
        )

    def _init_mediapipe(self) -> None:
        """Initialize MediaPipe Solutions Hands if present, else Tasks HandLandmarker."""
        solutions = getattr(mp, "solutions", None)
        if solutions is not None and hasattr(solutions, "hands"):
            self._solutions_hands = solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=self.max_num_hands,
                model_complexity=self.model_complexity,
                min_detection_confidence=self.min_detection_confidence,
                min_tracking_confidence=self.min_tracking_confidence,
            )
            self._backend = f"solutions.hands(complexity={self.model_complexity})"
            return

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        verified_model = ensure_hand_landmarker_model(self.model_path)
        delegate_norm = self.delegate.lower().strip()
        if delegate_norm == "gpu":
            candidates = [("GPU", mp_python.BaseOptions.Delegate.GPU)]
        elif delegate_norm == "cpu":
            candidates = [("CPU", mp_python.BaseOptions.Delegate.CPU)]
        else:
            # Windows pip wheels lack OpenGL GPU delegate; avoid failing probe delay.
            candidates = (
                [("CPU", mp_python.BaseOptions.Delegate.CPU)]
                if sys.platform.startswith("win")
                else [
                    ("GPU", mp_python.BaseOptions.Delegate.GPU),
                    ("CPU", mp_python.BaseOptions.Delegate.CPU),
                ]
            )

        last_exc: Exception | None = None
        for label, delegate_enum in candidates:
            try:
                base_options = mp_python.BaseOptions(
                    model_asset_path=str(verified_model),
                    delegate=delegate_enum,
                )
                options = mp_vision.HandLandmarkerOptions(
                    base_options=base_options,
                    running_mode=mp_vision.RunningMode.VIDEO,
                    num_hands=self.max_num_hands,
                    min_hand_detection_confidence=self.min_detection_confidence,
                    min_hand_presence_confidence=self.min_presence_confidence,
                    min_tracking_confidence=self.min_tracking_confidence,
                )
                self._landmarker = mp_vision.HandLandmarker.create_from_options(options)
                self._backend = f"tasks.HandLandmarker({label})"
                return
            except Exception as exc:
                last_exc = exc
                if delegate_norm not in {"auto", ""}:
                    raise

        raise RuntimeError(f"Failed to initialize MediaPipe HandLandmarker: {last_exc}")

    @property
    def backend_name(self) -> str:
        """Return a human-readable description of the active MediaPipe backend."""
        return self._backend

    def open(self) -> None:
        """Open the configured webcam with low-latency capture settings."""
        if self._cap is not None and self._cap.isOpened():
            return

        if sys.platform.startswith("win"):
            cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                self._configure_capture(cap)
                self._cap = cap
                return
            cap.release()

        cap = cv2.VideoCapture(self.camera_index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Unable to open webcam at index {self.camera_index}. "
                "Check camera permissions, connection, or update 'camera.index' in config.json."
            )
        self._configure_capture(cap)
        self._cap = cap

    def _configure_capture(self, cap: cv2.VideoCapture) -> None:
        try:
            fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
            cap.set(cv2.CAP_PROP_FOURCC, fourcc_fn(*"MJPG"))
        except Exception:
            pass
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read_frame(self) -> np.ndarray | None:
        """Capture a single BGR frame from the webcam."""
        if self._cap is None or not self._cap.isOpened():
            raise RuntimeError("Webcam is not opened.")
        ok, frame = self._cap.read()
        if not ok or frame is None or frame.size == 0:
            return None
        return frame

    def detect(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> tuple[np.ndarray, dict[HandLabel, TrackedHand]]:
        """Process a raw BGR frame and return (output_frame, hands_by_label).

        When ``mirror`` is True, the returned ``output_frame`` and landmark ``x``
        coordinates are horizontally mirrored for intuitive selfie viewing while
        preserving true physical ``Left`` / ``Right`` handedness labels.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return frame_bgr, {}

        if self._solutions_hands is not None:
            return self._detect_solutions(frame_bgr)
        return self._detect_tasks(frame_bgr, timestamp_ms=timestamp_ms)

    def _detect_solutions(
        self, frame_bgr: np.ndarray
    ) -> tuple[np.ndarray, dict[HandLabel, TrackedHand]]:
        # Legacy mp.solutions.hands expects selfie-mirrored input to label physical hands.
        out_frame = cv2.flip(frame_bgr, 1) if self.mirror else frame_bgr
        rgb = np.ascontiguousarray(cv2.cvtColor(out_frame, cv2.COLOR_BGR2RGB))
        rgb.flags.writeable = False
        result = self._solutions_hands.process(rgb)

        hands_by_label: dict[HandLabel, TrackedHand] = {}
        multi_lms = getattr(result, "multi_hand_landmarks", None) or []
        multi_handedness = getattr(result, "multi_handedness", None) or []

        for idx, hand_lms in enumerate(multi_lms):
            raw_label = "Unknown"
            score = 1.0
            if idx < len(multi_handedness) and multi_handedness[idx].classification:
                cls_info = multi_handedness[idx].classification[0]
                raw_label = getattr(cls_info, "label", "Unknown")
                score = float(getattr(cls_info, "score", 1.0))

            label = _normalize_label(raw_label, swap_handedness=self.swap_handedness)
            if label is None:
                continue

            coords = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_lms.landmark],
                dtype=np.float32,
            )
            self._store_best_hand(hands_by_label, TrackedHand(label, score, coords))

        return out_frame, hands_by_label

    def _detect_tasks(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: int | None = None,
    ) -> tuple[np.ndarray, dict[HandLabel, TrackedHand]]:
        # Tasks HandLandmarker evaluates subject handedness on unflipped frames;
        # mirror display frame + landmark x afterwards when mirror=True.
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        now_ms = int(time.perf_counter() * 1000.0) if timestamp_ms is None else int(timestamp_ms)
        ts_ms = max(now_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = ts_ms

        result = self._landmarker.detect_for_video(mp_image, ts_ms)
        out_frame = cv2.flip(frame_bgr, 1) if self.mirror else frame_bgr

        hands_by_label: dict[HandLabel, TrackedHand] = {}
        for idx, hand_lms in enumerate(result.hand_landmarks):
            raw_label = "Unknown"
            score = 1.0
            if idx < len(result.handedness) and result.handedness[idx]:
                cat = result.handedness[idx][0]
                raw_label = cat.category_name or "Unknown"
                score = float(cat.score if cat.score is not None else 1.0)

            label = _normalize_label(raw_label, swap_handedness=self.swap_handedness)
            if label is None:
                continue

            coords = np.array(
                [
                    [
                        1.0 - float(lm.x) if self.mirror else float(lm.x),
                        float(lm.y),
                        float(lm.z if lm.z is not None else 0.0),
                    ]
                    for lm in hand_lms
                ],
                dtype=np.float32,
            )
            self._store_best_hand(hands_by_label, TrackedHand(label, score, coords))

        return out_frame, hands_by_label

    @staticmethod
    def _store_best_hand(
        bucket: dict[HandLabel, TrackedHand], candidate: TrackedHand
    ) -> None:
        existing = bucket.get(candidate.handedness)
        if existing is None or candidate.score > existing.score:
            bucket[candidate.handedness] = candidate

    def read_and_detect(self) -> tuple[np.ndarray | None, dict[HandLabel, TrackedHand]]:
        """Capture one frame from the webcam and run hand landmark detection."""
        frame = self.read_frame()
        if frame is None:
            return None, {}
        return self.detect(frame)

    def start_async(self) -> None:
        """Start background webcam capture and MediaPipe inference threads."""
        if self._capture_thread is not None and self._capture_thread.is_alive():
            return
        if self._cap is None or not self._cap.isOpened():
            self.open()
        self._stop_event.clear()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="WebcamCaptureThread",
            daemon=True,
        )
        self._infer_thread = threading.Thread(
            target=self._infer_loop,
            name="MediaPipeInferThread",
            daemon=True,
        )
        self._capture_thread.start()
        self._infer_thread.start()

    def _capture_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                frame = self.read_frame()
            except Exception:
                break
            if frame is None:
                time.sleep(0.002)
                continue
            ts = time.perf_counter()
            with self._raw_cv:
                self._raw_frame = frame
                self._raw_ts = ts
                self._raw_seq += 1
                self._raw_cv.notify_all()

    def _infer_loop(self) -> None:
        last_seq = 0
        while not self._stop_event.is_set():
            with self._raw_cv:
                self._raw_cv.wait_for(
                    lambda: self._raw_seq != last_seq or self._stop_event.is_set(),
                    timeout=0.05,
                )
                if self._stop_event.is_set():
                    break
                if self._raw_frame is None or self._raw_seq == last_seq:
                    continue
                seq = self._raw_seq
                frame = self._raw_frame
                cap_ts = self._raw_ts
                last_seq = seq

            out_frame, hands = self.detect(frame, timestamp_ms=int(cap_ts * 1000.0))
            packet = DetectionPacket(
                seq=seq,
                frame=out_frame,
                hands=hands,
                timestamp=time.perf_counter(),
            )
            with self._result_lock:
                self._latest_packet = packet

    def get_latest(self) -> DetectionPacket | None:
        """Return the latest completed hand-detection packet without blocking."""
        with self._result_lock:
            return self._latest_packet

    def stop_async(self) -> None:
        """Stop background capture and inference threads cleanly."""
        self._stop_event.set()
        with self._raw_cv:
            self._raw_cv.notify_all()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=1.0)
            self._capture_thread = None
        if self._infer_thread is not None:
            self._infer_thread.join(timeout=1.0)
            self._infer_thread = None

    def release(self) -> None:
        """Stop async threads and release camera capture and MediaPipe resources."""
        self.stop_async()
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._solutions_hands is not None:
            self._solutions_hands.close()
            self._solutions_hands = None
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *_: object) -> None:
        self.release()
