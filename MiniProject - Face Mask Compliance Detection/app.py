"""
app.py
======
Flask backend for Face Mask Compliance Detection System.

Hybrid pipeline:
  1. Haar Cascade  — face detection & bounding boxes
  2. MediaPipe Face Mesh — 468-landmark facial feature extraction
  3. HSV skin analysis  — nose/mouth region exposure check
  4. MobileNetV2 CNN    — mask classification
  5. Decision Fusion    — combine all signals → final label

Routes:
  GET /        → dashboard (face.html)
  GET /video   → MJPEG stream
  GET /stats   → JSON statistics API
"""

import json
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template

# ── Optional TensorFlow import (graceful degradation if not available) ─────────
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    import tensorflow as tf
    CNN_AVAILABLE = True
except ImportError:
    CNN_AVAILABLE = False
    print("[WARN] TensorFlow not installed — CNN classifier disabled.")

# ── Optional MediaPipe import ──────────────────────────────────────────────────
try:
    import mediapipe as mp
    MP_AVAILABLE = True
except ImportError:
    MP_AVAILABLE = False
    print("[WARN] MediaPipe not installed — landmark detection disabled.")

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
TEMPLATE_DIR = BASE_DIR / "Template"
STATIC_DIR = BASE_DIR / "Script"
MODEL_PATH = BASE_DIR / "Model" / "mask_detector_mobilenetv2.h5"
CLASS_LABELS_PATH = BASE_DIR / "Model" / "class_labels.json"
CASCADE_PATH = BASE_DIR / "Data" / "haarcascade_frontalface_default.xml"

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


def update_stats(detections: list[dict]):
    with _stats_lock:
        _stats["total_faces"] = len(detections)
        _stats["proper_mask"] = sum(1 for d in detections if d["label"] == "proper_mask")
        _stats["incorrect_mask"] = sum(1 for d in detections if d["label"] == "incorrect_mask")
        _stats["no_mask"] = sum(1 for d in detections if d["label"] == "no_mask")
        _stats["uptime_seconds"] = int(time.time() - _stats["start_time"])


# ──────────────────────────────────────────────
# DETECTION ENGINE
# ──────────────────────────────────────────────

class FaceMaskDetector:
    # Landmark indices for specific features
    NOSE_LANDMARKS = [1, 2, 98, 327]
    MOUTH_LANDMARKS = [13, 14, 78, 308]
    EYE_LANDMARKS = [33, 133, 362, 263]

    # HSV skin detection thresholds
    SKIN_LOWER = np.array([0, 40, 60], dtype=np.uint8)
    SKIN_UPPER = np.array([25, 255, 255], dtype=np.uint8)
    SKIN_THRESHOLD = 0.40  # 40% skin pixels → feature visible

    # Label colour map (BGR)
    COLORS = {
        "proper_mask": (0, 200, 0),        # green
        "incorrect_mask": (0, 140, 255),   # orange
        "no_mask": (0, 0, 220),            # red
    }

    LABEL_TEXT = {
        "proper_mask": "Correct Mask",
        "incorrect_mask": "Incorrect Mask",
        "no_mask": "No Mask",
    }

    def __init__(self):
        # Haar Cascade
        self.face_cascade = cv2.CascadeClassifier(str(CASCADE_PATH))

        # MediaPipe Face Mesh
        self.mp_face_mesh = None
        self.face_mesh = None
        if MP_AVAILABLE:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=10,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

        # CNN Model
        self.cnn_model = None
        self.class_labels = {0: "proper_mask", 1: "no_mask", 2: "incorrect_mask"}
        if CNN_AVAILABLE and MODEL_PATH.exists():
            try:
                self.cnn_model = tf.keras.models.load_model(str(MODEL_PATH))
                if CLASS_LABELS_PATH.exists():
                    with open(CLASS_LABELS_PATH) as f:
                        raw = json.load(f)
                        self.class_labels = {int(k): v for k, v in raw.items()}
                print("[✓] CNN model loaded.")
            except Exception as e:
                print(f"[WARN] Failed to load CNN model: {e}")
        else:
            print("[WARN] CNN model not found — using HSV-only classification.")

    def detect_faces(self, frame_gray):
        """Haar cascade face detection."""
        faces = self.face_cascade.detectMultiScale(
            frame_gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
        )
        return faces if len(faces) > 0 else []

    def get_landmarks(self, frame_rgb) -> list[list[tuple]]:
        """Return per-face landmark coordinate lists using MediaPipe."""
        if self.face_mesh is None:
            return []
        results = self.face_mesh.process(frame_rgb)
        if not results.multi_face_landmarks:
            return []
        h, w = frame_rgb.shape[:2]
        all_landmarks = []
        for face_lm in results.multi_face_landmarks:
            pts = [(int(lm.x * w), int(lm.y * h)) for lm in face_lm.landmark]
            all_landmarks.append(pts)
        return all_landmarks

    def skin_ratio(self, roi_hsv) -> float:
        """Calculate proportion of skin-coloured pixels in ROI."""
        mask = cv2.inRange(roi_hsv, self.SKIN_LOWER, self.SKIN_UPPER)
        return mask.sum() / 255 / max(1, roi_hsv.shape[0] * roi_hsv.shape[1])

    def hsv_analysis(self, frame, landmarks: list[tuple]) -> dict:
        """Determine if nose/mouth are visible via HSV skin ratio."""
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        def roi_ratio(indices):
            pts = [landmarks[i] for i in indices if i < len(landmarks)]
            if not pts:
                return 0.0
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            pad = 15
            x1 = max(0, min(xs) - pad)
            y1 = max(0, min(ys) - pad)
            x2 = min(w, max(xs) + pad)
            y2 = min(h, max(ys) + pad)
            if x2 <= x1 or y2 <= y1:
                return 0.0
            roi = hsv[y1:y2, x1:x2]
            return self.skin_ratio(roi)

        nose_ratio = roi_ratio(self.NOSE_LANDMARKS)
        mouth_ratio = roi_ratio(self.MOUTH_LANDMARKS)
        return {
            "nose_visible": nose_ratio > self.SKIN_THRESHOLD,
            "mouth_visible": mouth_ratio > self.SKIN_THRESHOLD,
        }

    def cnn_predict(self, face_roi) -> tuple[str, float]:
        """Run CNN inference on a face ROI (BGR crop)."""
        if self.cnn_model is None:
            return "proper_mask", 0.0
        img = cv2.resize(face_roi, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype("float32") / 255.0
        img = np.expand_dims(img, axis=0)
        probs = self.cnn_model.predict(img, verbose=0)[0]
        idx = int(np.argmax(probs))
        return self.class_labels.get(idx, "no_mask"), float(probs[idx])

    def decide(self, cnn_label: str, confidence: float, hsv: dict) -> str:
        """Fusion layer: combine CNN + HSV signals."""
        nose = hsv["nose_visible"]
        mouth = hsv["mouth_visible"]

        if confidence < 0.5 and nose and mouth:
            return "no_mask"
        if cnn_label == "no_mask":
            return "no_mask"
        if cnn_label == "incorrect_mask":
            return "incorrect_mask"
        # cnn_label == "proper_mask"
        if nose or mouth:
            return "incorrect_mask"
        return "proper_mask"

    def process_frame(self, frame) -> tuple[np.ndarray, list[dict]]:
        """Run the full pipeline on a single frame."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = self.detect_faces(frame_gray)
        all_landmarks = self.get_landmarks(frame_rgb)

        detections = []

        for i, (x, y, w, h) in enumerate(faces):
            face_roi = frame[y:y+h, x:x+w]
            if face_roi.size == 0:
                continue

            # CNN prediction
            cnn_label, confidence = self.cnn_predict(face_roi)

            # Find closest landmark set for this face
            lm_set = None
            if all_landmarks:
                cx, cy = x + w // 2, y + h // 2
                best = min(
                    range(len(all_landmarks)),
                    key=lambda j: abs(all_landmarks[j][1][0] - cx) + abs(all_landmarks[j][1][1] - cy)
                    if len(all_landmarks[j]) > 1 else 9999,
                )
                lm_set = all_landmarks[best]

            hsv_result = {"nose_visible": False, "mouth_visible": False}
            if lm_set:
                hsv_result = self.hsv_analysis(frame, lm_set)

            final_label = self.decide(cnn_label, confidence, hsv_result)
            color = self.COLORS[final_label]
            text = self.LABEL_TEXT[final_label]

            # Draw bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(frame, (x, y - th - 10), (x + tw + 8, y), color, -1)
            cv2.putText(
                frame, text, (x + 4, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
            )

            # Confidence badge
            conf_text = f"{confidence:.0%}"
            cv2.putText(
                frame, conf_text, (x + w - 50, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
            )

            # Draw facial landmarks
            if lm_set:
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
                "label": final_label,
                "confidence": confidence,
                "bbox": [int(x), int(y), int(w), int(h)],
            })

        return frame, detections


# ──────────────────────────────────────────────
# VIDEO STREAM GENERATOR
# ──────────────────────────────────────────────

detector = FaceMaskDetector()
cap = None
_cap_lock = threading.Lock()


def get_capture():
    global cap
    with _cap_lock:
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(0)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    return cap


def generate_frames():
    prev_time = time.time()
    frame_count = 0

    while True:
        capture = get_capture()
        ret, frame = capture.read()
        if not ret:
            time.sleep(0.05)
            continue

        frame, detections = detector.process_frame(frame)
        update_stats(detections)

        # FPS overlay
        frame_count += 1
        now = time.time()
        elapsed = now - prev_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            with _stats_lock:
                _stats["fps"] = round(fps, 1)
            frame_count = 0
            prev_time = now

        fps_text = f"FPS: {_stats['fps']}"
        cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
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


# ──────────────────────────────────────────────
# ENTRY POINT
# ──────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Face Mask Compliance Detection System")
    print("  http://localhost:5000")
    print("=" * 60)
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
