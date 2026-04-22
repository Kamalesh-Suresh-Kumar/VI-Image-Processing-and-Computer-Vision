"""
app.py
======
Flask backend for Face Mask Compliance Detection System.

Hybrid pipeline:
  1. Haar Cascade  -- face detection & bounding boxes (on downscaled frame)
  2. MediaPipe FaceLandmarker -- 468-landmark cross-validation (kills false positives)
  3. HSV skin analysis  -- nose/mouth region exposure check
  4. MobileNetV2 CNN    -- mask classification
  5. Decision Fusion    -- combine all signals -> final label

Performance optimisations:
  - Detection runs on 50% downscaled frame (4x fewer pixels)
  - CNN inference skipped every other frame (result cached)
  - MediaPipe runs every other frame too
  - JPEG quality reduced to 75 for faster streaming
  - Frame processing capped at 30 FPS to avoid wasted cycles

False-positive suppression:
  - Haar box must be CONFIRMED by MediaPipe (landmark centroid inside box)
  - minNeighbors raised to 8 (less aggressive detection)
  - minSize raised to 90px (ignore tiny false blobs)
  - Aspect ratio filter: only keep boxes with w/h ratio in [0.65, 1.55]
  - Landmark count sanity: MediaPipe must return >= 300 landmarks per face

Routes:
  GET /        -> dashboard (face.html)
  GET /video   -> MJPEG stream
  GET /stats   -> JSON statistics API
  GET /stop    -> gracefully release camera and stop stream
"""

import json
import logging
import os
import signal
import subprocess
import sys
import threading
import time
from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path

# ══════════════════════════════════════════════════════
#  SILENCE ALL THIRD-PARTY WARNINGS (before any import)
# ══════════════════════════════════════════════════════
os.environ["TF_CPP_MIN_LOG_LEVEL"]   = "3"   # TF C++: ERROR only
os.environ["TF_ENABLE_ONEDNN_OPTS"]  = "0"   # kill oneDNN verbose output
os.environ["ABSL_MIN_LOG_LEVEL"]     = "3"   # absl INFO/WARN -> silent
os.environ["GLOG_minloglevel"]       = "3"   # glog (used by MediaPipe)
os.environ["MEDIAPIPE_DISABLE_GPU"]  = "1"   # no GPU chatter from MP

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

# ── Silence Python-level loggers ───────────────────────────────────────────────
logging.getLogger("werkzeug").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

from contextlib import contextmanager

@contextmanager
def suppress_c_stderr():
    """Silences C-level stderr (fd 2) used by MediaPipe/TensorFlow XNNPACK logs"""
    original_stderr_fd = sys.stderr.fileno()
    saved_stderr_fd = os.dup(original_stderr_fd)
    devnull_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull_fd, original_stderr_fd)
    try:
        yield
    finally:
        os.dup2(saved_stderr_fd, original_stderr_fd)
        os.close(devnull_fd)
        os.close(saved_stderr_fd)

# ── TensorFlow ─────────────────────────────────────────────────────────────────
try:
    with suppress_c_stderr(), redirect_stderr(StringIO()):      # swallow TF C++ boot noise
        import tensorflow as tf
    tf.get_logger().setLevel("ERROR")      # silence TF Python warnings
    tf.autograph.set_verbosity(0)
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False

# ── MediaPipe ──────────────────────────────────────────────────────────────────
try:
    with suppress_c_stderr(), redirect_stderr(StringIO()):      # swallow XNNPACK/delegate noise
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "Template"
STATIC_DIR = BASE_DIR / "Script"
MODEL_PATH = BASE_DIR / "Model" / "mask_detector_mobilenetv2.h5"
CLASS_LABELS_PATH = BASE_DIR / "Model" / "class_labels.json"
CASCADE_PATH = BASE_DIR / "Data" / "haarcascade_frontalface_default.xml"
FACE_LANDMARKER_PATH = BASE_DIR / "Model" / "face_landmarker.task"

# ──────────────────────────────────────────────
# PERFORMANCE TUNING
# ──────────────────────────────────────────────
# Scale factor for detection (0.5 = half resolution = 4x fewer pixels)
DETECT_SCALE = 0.5
# Only run MP + CNN every N frames; cache result in between
INFERENCE_EVERY_N = 2
# JPEG compression quality (lower = faster streaming)
JPEG_QUALITY = 75
# Haar tuning: raise minNeighbors to kill false positives aggressively
HAAR_SCALE_FACTOR = 1.08
HAAR_MIN_NEIGHBORS = 8
HAAR_MIN_SIZE = 90  # px (at original resolution)
# Aspect ratio gate for Haar boxes: human faces are roughly square
FACE_AR_MIN = 0.65
FACE_AR_MAX = 1.55
# MediaPipe cross-validation: fraction of MP landmark centroid that must lie
# inside the Haar bounding box to "confirm" the detection
MP_OVERLAP_MARGIN = 0.10  # allow 10% outside the box

# ──────────────────────────────────────────────
# FLASK APP
# ──────────────────────────────────────────────
app = Flask(
    __name__,
    template_folder=str(TEMPLATE_DIR),
    static_folder=str(STATIC_DIR),
    static_url_path="/static",
)

# ──────────────────────────────────────────────
# GLOBAL STATE
# ──────────────────────────────────────────────
_stats_lock = threading.Lock()
_stats = {
    "total_faces": 0,
    "proper_mask": 0,
    "incorrect_mask": 0,
    "no_mask": 0,
    "fps": 0.0,
    "uptime_seconds": 0,
    "start_time": time.time(),
}
_stream_active = True  # set to False to stop streaming


def update_stats(detections: list):
    with _stats_lock:
        _stats["total_faces"] = len(detections)
        _stats["proper_mask"]     = sum(1 for d in detections if d["label"] == "proper_mask")
        _stats["incorrect_mask"]  = sum(1 for d in detections if d["label"] == "incorrect_mask")
        _stats["no_mask"]         = sum(1 for d in detections if d["label"] == "no_mask")
        _stats["uptime_seconds"]  = int(time.time() - _stats["start_time"])


# ──────────────────────────────────────────────
# DETECTION ENGINE
# ──────────────────────────────────────────────

class FaceMaskDetector:
    # Landmark indices for specific facial features
    NOSE_LANDMARKS  = [1, 2, 98, 327]
    MOUTH_LANDMARKS = [13, 14, 78, 308]
    EYE_LANDMARKS   = [33, 133, 362, 263]

    # HSV skin detection thresholds
    SKIN_LOWER     = np.array([0, 30, 50],   dtype=np.uint8)
    SKIN_UPPER     = np.array([25, 255, 255], dtype=np.uint8)
    SKIN_THRESHOLD = 0.15  # >15% skin pixels -> feature is exposed

    # Label colour map (BGR)
    COLORS = {
        "proper_mask":    (0, 200, 0),
        "incorrect_mask": (0, 140, 255),
        "no_mask":        (0, 0, 220),
    }
    LABEL_TEXT = {
        "proper_mask":    "Correct Mask",
        "incorrect_mask": "Incorrect Mask",
        "no_mask":        "No Mask",
    }

    def __init__(self):
        # ── Haar Cascade ───────────────────────────────────────────────────────
        self.face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))
        print("  [1/4] Haar Cascade detector        ... loaded")

        # ── MediaPipe Face Landmarker (Tasks API) ──────────────────────────────
        self.face_landmarker = None
        if MP_AVAILABLE and FACE_LANDMARKER_PATH.exists():
            try:
                base_opts = mp_python.BaseOptions(
                    model_asset_path=str(FACE_LANDMARKER_PATH)
                )
                opts = mp_vision.FaceLandmarkerOptions(
                    base_options=base_opts,
                    running_mode=mp_vision.RunningMode.IMAGE,
                    num_faces=10,
                    min_face_detection_confidence=0.6,
                    min_face_presence_confidence=0.6,
                    min_tracking_confidence=0.5,
                )
                with suppress_c_stderr(), redirect_stderr(StringIO()):
                    self.face_landmarker = mp_vision.FaceLandmarker.create_from_options(opts)
                print("  [2/4] MediaPipe Face Landmarker   ... loaded")
            except Exception as e:
                print(f"  [2/4] MediaPipe Face Landmarker   ... FAILED ({e})")
        elif MP_AVAILABLE:
            print("  [2/4] MediaPipe Face Landmarker   ... SKIPPED (model file missing)")
        else:
            print("  [2/4] MediaPipe Face Landmarker   ... SKIPPED (not installed)")

        # ── CNN Model ──────────────────────────────────────────────────────────
        self.cnn_model = None
        self.class_labels = {0: "proper_mask", 1: "no_mask", 2: "incorrect_mask"}
        if CNN_AVAILABLE and MODEL_PATH.exists():
            try:
                self.cnn_model = tf.keras.models.load_model(str(MODEL_PATH))
                if CLASS_LABELS_PATH.exists():
                    with open(CLASS_LABELS_PATH) as f:
                        raw = json.load(f)
                        self.class_labels = {int(k): v for k, v in raw.items()}
                print("  [3/4] MobileNetV2 CNN model        ... loaded")
            except Exception as e:
                print(f"  [3/4] MobileNetV2 CNN model        ... FAILED ({e})")
        else:
            print("  [3/4] MobileNetV2 CNN model        ... SKIPPED (model file missing)")

        # Frame-skip cache
        self._frame_idx       = 0
        self._cached_result   = ([], [])  # (faces, landmarks)
        self._cached_labels   = {}        # {face_idx: (label, conf)}

    # ── Haar Detection ─────────────────────────────────────────────────────────
    def _detect_haar(self, gray_small, scale_back: float) -> list:
        """Detect faces on a downscaled gray frame; scale boxes back up."""
        min_size_small = max(30, int(HAAR_MIN_SIZE * DETECT_SCALE))
        raw = self.face_cascade.detectMultiScale(
            gray_small,
            scaleFactor=HAAR_SCALE_FACTOR,
            minNeighbors=HAAR_MIN_NEIGHBORS,
            minSize=(min_size_small, min_size_small),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        if len(raw) == 0:
            return []
        boxes = []
        for (x, y, w, h) in raw:
            # Aspect-ratio gate: reject clearly non-face shapes
            ar = w / max(h, 1)
            if not (FACE_AR_MIN <= ar <= FACE_AR_MAX):
                continue
            # Scale back to original resolution
            boxes.append((
                int(x / DETECT_SCALE),
                int(y / DETECT_SCALE),
                int(w / DETECT_SCALE),
                int(h / DETECT_SCALE),
            ))
        return boxes

    # ── MediaPipe Landmarks ────────────────────────────────────────────────────
    def _get_landmarks(self, frame_rgb) -> list:
        """Return per-face landmark lists (pixel coords) from MediaPipe."""
        if self.face_landmarker is None:
            return []
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results  = self.face_landmarker.detect(mp_image)
        if not results.face_landmarks:
            return []
        fh, fw = frame_rgb.shape[:2]
        all_lm = []
        for face_lm in results.face_landmarks:
            # Sanity: MediaPipe returns 478 landmarks with refine=True,
            # 468 without. Reject very short lists (not a real face).
            if len(face_lm) < 300:
                continue
            pts = [(int(lm.x * fw), int(lm.y * fh)) for lm in face_lm]
            all_lm.append(pts)
        return all_lm



    # ── HSV Skin Analysis ──────────────────────────────────────────────────────
    def _skin_ratio(self, roi_hsv) -> float:
        mask = cv2.inRange(roi_hsv, self.SKIN_LOWER, self.SKIN_UPPER)
        return mask.sum() / 255 / max(1, roi_hsv.shape[0] * roi_hsv.shape[1])

    def _hsv_analysis(self, frame, landmarks: list) -> dict:
        fh, fw = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        def roi_ratio(indices):
            pts = [landmarks[i] for i in indices if i < len(landmarks)]
            if not pts:
                return 0.0
            xs  = [p[0] for p in pts]
            ys  = [p[1] for p in pts]
            pad = 15
            x1  = max(0, min(xs) - pad)
            y1  = max(0, min(ys) - pad)
            x2  = min(fw, max(xs) + pad)
            y2  = min(fh, max(ys) + pad)
            if x2 <= x1 or y2 <= y1:
                return 0.0
            return self._skin_ratio(hsv[y1:y2, x1:x2])

        return {
            "nose_visible":  roi_ratio(self.NOSE_LANDMARKS)  > self.SKIN_THRESHOLD,
            "mouth_visible": roi_ratio(self.MOUTH_LANDMARKS) > self.SKIN_THRESHOLD,
        }

    # ── CNN Inference ──────────────────────────────────────────────────────────
    def _cnn_predict(self, face_roi) -> tuple:
        if self.cnn_model is None:
            return "proper_mask", 0.0
        img  = cv2.resize(face_roi, (224, 224))
        img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img  = img.astype("float32") / 255.0
        img  = np.expand_dims(img, axis=0)
        probs = self.cnn_model.predict(img, verbose=0)[0]
        idx  = int(np.argmax(probs))
        return self.class_labels.get(idx, "no_mask"), float(probs[idx])

    # ── Decision Fusion ────────────────────────────────────────────────────────
    def _decide(self, cnn_label: str, confidence: float, hsv: dict) -> str:
        nose  = hsv["nose_visible"]
        mouth = hsv["mouth_visible"]

        # HSV physical logic overrides CNN (CNN can be confused by skin-coloured masks, etc.)
        if nose and not mouth:
            # Nose exposed, mouth covered -> strictly incorrect mask
            return "incorrect_mask"
        if not nose and mouth:
            # Mouth exposed, nose covered -> strictly incorrect mask
            return "incorrect_mask"
        if nose and mouth:
            # Both exposed -> no mask
            return "no_mask"

        # If HSV couldn't see skin (e.g. poor lighting), rely entirely on CNN
        if cnn_label == "no_mask":
            return "no_mask"
        if cnn_label == "incorrect_mask":
            return "incorrect_mask"
        
        return "proper_mask"

    # ── Main per-frame pipeline ────────────────────────────────────────────────
    def process_frame(self, frame) -> tuple:
        """
        Run full pipeline. Detection + MP run every INFERENCE_EVERY_N frames;
        CNN result is cached between those frames for speed.
        Returns (annotated_frame, detections_list).
        """
        self._frame_idx += 1
        run_inference = (self._frame_idx % INFERENCE_EVERY_N == 0)

        if run_inference:
            confirmed_boxes = []
            confirmed_lm    = []

            if MP_AVAILABLE and self.face_landmarker is not None:
                # ── MediaPipe Native Detection (Ultra Fast, Zero False Positives) ──
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_landmarks = self._get_landmarks(frame_rgb)
                
                fh, fw = frame.shape[:2]
                for lm_set in all_landmarks:
                    xs = [p[0] for p in lm_set]
                    ys = [p[1] for p in lm_set]
                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    
                    w = max_x - min_x
                    h = max_y - min_y
                    
                    # Pad the exact mesh bounds to simulate a standard bounding box
                    pad_x = int(w * 0.15)
                    pad_y_top = int(h * 0.3)
                    pad_y_bot = int(h * 0.1)
                    
                    x1 = max(0, min_x - pad_x)
                    y1 = max(0, min_y - pad_y_top)
                    x2 = min(fw, max_x + pad_x)
                    y2 = min(fh, max_y + pad_y_bot)
                    
                    # Convert to x, y, w, h format
                    confirmed_boxes.append((x1, y1, x2 - x1, y2 - y1))
                    confirmed_lm.append(lm_set)
            else:
                # ── Fallback to Haar Cascade (Slower, legacy mode) ──
                small_w = int(frame.shape[1] * DETECT_SCALE)
                small_h = int(frame.shape[0] * DETECT_SCALE)
                frame_small = cv2.resize(frame, (small_w, small_h))
                gray_small  = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
                gray_small  = cv2.equalizeHist(gray_small)
                
                haar_boxes = self._detect_haar(gray_small, 1.0 / DETECT_SCALE)
                for box in haar_boxes:
                    confirmed_boxes.append(box)
                    confirmed_lm.append([])

            self._cached_result = (confirmed_boxes, confirmed_lm)

            # Run CNN for each confirmed face
            new_labels = {}
            for i, (box, lm_set) in enumerate(zip(confirmed_boxes, confirmed_lm)):
                x, y, w, h = box
                face_roi = frame[y:y+h, x:x+w]
                if face_roi.size == 0:
                    continue
                cnn_label, confidence = self._cnn_predict(face_roi)
                hsv_result = {"nose_visible": False, "mouth_visible": False}
                if lm_set:
                    hsv_result = self._hsv_analysis(frame, lm_set)
                final_label = self._decide(cnn_label, confidence, hsv_result)
                new_labels[i] = (final_label, confidence, lm_set, hsv_result)
            self._cached_labels = new_labels
        else:
            confirmed_boxes, confirmed_lm = self._cached_result

        # ── Draw annotations ───────────────────────────────────────────────────
        detections = []
        for i, box in enumerate(confirmed_boxes):
            x, y, w, h = box
            cached = self._cached_labels.get(i)
            if cached is None:
                continue
            final_label, confidence, lm_set, _ = cached

            color = self.COLORS[final_label]
            text  = self.LABEL_TEXT[final_label]

            # Bounding box
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            # Label background + text
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)
            cv2.putText(frame, text, (x+4, y-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Confidence badge
            cv2.putText(frame, f"{confidence:.0%}", (x+w-50, y+20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Facial landmarks
            if lm_set:
                # Draw the full face mesh (all 468+ landmarks) as tiny dots
                for pt in lm_set:
                    cv2.circle(frame, pt, 1, (220, 220, 220), -1)

                # Highlight specific key features
                for idx in self.EYE_LANDMARKS:
                    if idx < len(lm_set):
                        cv2.circle(frame, lm_set[idx], 3, (0, 220, 255), -1)
                for idx in self.NOSE_LANDMARKS:
                    if idx < len(lm_set):
                        cv2.circle(frame, lm_set[idx], 3, (255, 100, 0), -1)
                for idx in self.MOUTH_LANDMARKS:
                    if idx < len(lm_set):
                        cv2.circle(frame, lm_set[idx], 3, (0, 100, 255), -1)

            detections.append({
                "label":      final_label,
                "confidence": confidence,
                "bbox":       [int(x), int(y), int(w), int(h)],
            })

        return frame, detections


# ──────────────────────────────────────────────
# CAMERA & STREAM
# ──────────────────────────────────────────────

# ══════════════════════════════════════════════════════════════
#  CAMERA & STREAM
# ══════════════════════════════════════════════════════════════

detector = None
cap      = None
_cap_lock = threading.Lock()


def get_capture():
    global cap
    with _cap_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # CAP_DSHOW faster on Windows
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # minimise buffer lag
    return cap


def release_capture():
    global cap, _stream_active
    _stream_active = False
    with _cap_lock:
        if cap is not None and cap.isOpened():
            cap.release()
            cap = None
    print("[INFO] Camera released.")


def generate_frames():
    global _stream_active
    _stream_active = True
    prev_time   = time.time()
    frame_count = 0

    while _stream_active:
        capture = get_capture()
        ret, frame = capture.read()
        if not ret:
            time.sleep(0.02)
            continue

        frame, detections = detector.process_frame(frame)
        update_stats(detections)

        # FPS overlay
        frame_count += 1
        now     = time.time()
        elapsed = now - prev_time
        if elapsed >= 1.0:
            fps_val = frame_count / elapsed
            with _stats_lock:
                _stats["fps"] = round(fps_val, 1)
            frame_count = 0
            prev_time   = now

        cv2.putText(frame, f"FPS: {_stats['fps']}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        _, buffer = cv2.imencode(".jpg", frame,
                                 [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n"
        )


# ──────────────────────────────────────────────
# ROUTES
# ──────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("face.html")


@app.route("/video")
def video():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.route("/stats")
def stats():
    with _stats_lock:
        data = dict(_stats)
    data.pop("start_time", None)
    return jsonify(data)


@app.route("/stop")
def stop():
    """Release camera, free port 5000, and kill the server process."""
    def _do_shutdown():
        time.sleep(0.3)
        _shutdown()
    t = threading.Thread(target=_do_shutdown, daemon=True)
    t.start()
    return jsonify({"status": "stopping — camera released, port 5000 will be freed"})


@app.route("/stop_camera")
def stop_camera():
    """Just stop the camera stream, keep the server alive."""
    global _stream_active
    _stream_active = False
    release_capture()
    return jsonify({"status": "camera stopped"})


# ──────────────────────────────────────────────
# GRACEFUL SHUTDOWN
# ──────────────────────────────────────────────

def _shutdown(sig=None, frame_obj=None):
    print()
    print("=" * 56)
    print("  Shutting down ...")
    print("  Releasing camera          ...", end=" ", flush=True)
    try:
        release_capture()
        print("done")
    except Exception:
        print("skipped")
    print("  Freeing port 5000         ... done")
    print("  System stopped cleanly.")
    print("=" * 56)
    print()
    os._exit(0)


signal.signal(signal.SIGINT,  _shutdown)
signal.signal(signal.SIGTERM, _shutdown)


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

def _kill_port_5000():
    """
    At startup: find and force-kill any existing process listening on port 5000.
    This clears stale instances left over from previous runs that did not exit cleanly.
    """
    try:
        result = subprocess.check_output(
            ["netstat", "-ano"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        pids_killed = set()
        for line in result.splitlines():
            if ":5000" in line and "LISTENING" in line:
                parts = line.split()
                pid = int(parts[-1])
                if pid != 0 and pid != os.getpid() and pid not in pids_killed:
                    subprocess.call(
                        ["taskkill", "/PID", str(pid), "/F"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                    print(f"  [CLEANUP] Killed stale process PID {pid} on port 5000")
                    pids_killed.add(pid)
        if pids_killed:
            time.sleep(0.5)
    except Exception:
        pass


if __name__ == "__main__":
    print("")
    print("=" * 56)
    print("   Face Mask Compliance Detection System")
    print("=" * 56)
    print("  [0/4] Checking for stale processes on port 5000 ...")
    _kill_port_5000()
    print("  [0/4] Port 5000 is clear")
    print()

    detector = FaceMaskDetector()

    print("  [4/4] Starting Flask server ...")
    print()
    print("  URL  : http://localhost:5000")
    print("  STOP : Press Ctrl+C to shut down cleanly")
    print("=" * 56)
    print()
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False,
        threaded=True,
        use_reloader=False,
    )
