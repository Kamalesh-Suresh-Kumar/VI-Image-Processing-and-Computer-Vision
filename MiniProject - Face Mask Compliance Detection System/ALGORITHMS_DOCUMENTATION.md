# Algorithms Documentation — Face Mask Compliance Detection System

## Table of Contents

1. [Haar Cascade Classifier](#1-haar-cascade-classifier)
2. [MediaPipe Face Mesh](#2-mediapipe-face-mesh)
3. [HSV Skin Color Analysis](#3-hsv-skin-color-analysis)
4. [MobileNetV2 CNN](#4-mobilenetv2-cnn)
5. [Decision Fusion Layer](#5-decision-fusion-layer)
6. [Data Augmentation Pipeline](#6-data-augmentation-pipeline)

---

## 1. Haar Cascade Classifier

### Purpose
Detect face bounding boxes in each video frame.

### How It Works
The **Viola-Jones** framework (2001) uses Haar-like features — rectangular patterns that measure intensity differences between adjacent image regions. A sliding window scans the image at multiple scales.

**Key concepts:**
- **Integral Image**: Enables O(1) computation of any rectangular sum, making feature evaluation extremely fast.
- **AdaBoost**: Selects the most discriminative Haar features from a pool of ~160,000 candidates.
- **Cascade of Classifiers**: A chain of progressively complex stages. Early stages reject obvious non-face regions quickly (>95% rejection rate), so the full classifier only runs on promising candidates.

### Parameters Used
| Parameter | Value | Description |
|-----------|-------|-------------|
| `scaleFactor` | 1.1 | Image pyramid scale step (10% reduction per level) |
| `minNeighbors` | 5 | Minimum overlapping detections required to confirm a face |
| `minSize` | (60, 60) | Smallest face size to detect (pixels) |

### File
- **Cascade XML**: `Data/haarcascade_frontalface_default.xml`
- **Usage**: `app.py` → `FaceMaskDetector.detect_faces()`

---

## 2. MediaPipe Face Mesh

### Purpose
Extract 468 facial landmarks for precise feature localization (eyes, nose, mouth).

### How It Works
Google's **MediaPipe Face Mesh** uses a two-stage ML pipeline:

1. **Face Detection**: A lightweight BlazeFace SSD model localizes the face and estimates a face bounding box.
2. **Landmark Regression**: A dedicated neural network predicts 468 3D landmark coordinates (x, y, z) within the detected face region.

The model runs in real-time on CPU (10+ FPS) with high precision, providing sub-pixel accuracy for facial features.

### Key Landmarks Used

| Feature | Landmark Indices | Color (BGR) |
|---------|-----------------|-------------|
| **Eyes** | 33, 133, 362, 263 | Cyan (0, 220, 255) |
| **Nose** | 1, 2, 98, 327 | Orange (255, 100, 0) |
| **Mouth** | 13, 14, 78, 308 | Blue (0, 100, 255) |

### Configuration
| Parameter | Value |
|-----------|-------|
| `static_image_mode` | False (tracking mode for video) |
| `max_num_faces` | 10 |
| `refine_landmarks` | True (includes iris landmarks) |
| `min_detection_confidence` | 0.5 |
| `min_tracking_confidence` | 0.5 |

### File
- **Task file**: `Model/face_landmarker.task`
- **Usage**: `app.py` → `FaceMaskDetector.get_landmarks()`

---

## 3. HSV Skin Color Analysis

### Purpose
Determine if the nose and mouth are exposed (uncovered by a mask) by detecting skin-colored pixels in those regions.

### How It Works

1. **ROI Extraction**: Using MediaPipe landmarks, extract rectangular regions around the nose and mouth with a 15-pixel padding.
2. **Color Space Conversion**: Convert the ROI from BGR to **HSV** (Hue-Saturation-Value), which separates color information from intensity, making skin detection more robust to lighting changes.
3. **Skin Thresholding**: Apply a range filter to isolate skin-colored pixels.
4. **Ratio Calculation**: Compute `skin_pixels / total_pixels`. If > 40%, the feature is considered "visible" (uncovered).

### HSV Thresholds

| Channel | Lower | Upper | Rationale |
|---------|-------|-------|-----------|
| **Hue** | 0 | 25 | Covers typical human skin tones (reddish-yellow range) |
| **Saturation** | 40 | 255 | Excludes very pale/white pixels (fabric, paper) |
| **Value** | 60 | 255 | Excludes very dark pixels (shadows, dark masks) |

### Decision Rule
- **Nose skin ratio > 40%** → Nose is visible (not covered)
- **Mouth skin ratio > 40%** → Mouth is visible (not covered)

### File
- **Usage**: `app.py` → `FaceMaskDetector.hsv_analysis()`

---

## 4. MobileNetV2 CNN

### Purpose
Classify cropped face images into three categories: `proper_mask`, `incorrect_mask`, `no_mask`.

### Architecture

**MobileNetV2** (Sandler et al., 2018) is a lightweight convolutional neural network designed for mobile/edge deployment. Key innovations:

- **Inverted Residual Blocks**: Unlike ResNet (wide→narrow→wide), MobileNetV2 uses narrow→wide→narrow bottlenecks with depthwise separable convolutions.
- **Linear Bottlenecks**: The final layer in each block uses a linear activation (no ReLU) to preserve information in low-dimensional space.
- **Depthwise Separable Convolutions**: Factorizes standard convolution into depthwise (per-channel) + pointwise (1×1) convolutions, reducing computation by ~8-9×.

### Transfer Learning Setup

```
Input (224 × 224 × 3)
    │
    ▼
MobileNetV2 Base (ImageNet weights, FROZEN)
    │
    ▼
GlobalAveragePooling2D → (1280,)
    │
    ▼
Dropout(0.3)
    │
    ▼
Dense(128, ReLU)
    │
    ▼
Dropout(0.2)
    │
    ▼
Dense(3, Softmax) → [proper_mask, no_mask, incorrect_mask]
```

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1e-4 |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 32 |
| Epochs | 20 (with early stopping) |
| Input Size | 224 × 224 × 3 |
| Base Weights | ImageNet (frozen) |

### Callbacks
| Callback | Configuration |
|----------|--------------|
| **EarlyStopping** | Monitor `val_loss`, patience=5, restore best weights |
| **ModelCheckpoint** | Save best model by `val_accuracy` |
| **ReduceLROnPlateau** | Monitor `val_loss`, factor=0.5, patience=3, min_lr=1e-6 |

### Saved Artifacts
| File | Description |
|------|-------------|
| `Model/mask_detector_mobilenetv2.h5` | Trained model weights |
| `Model/class_labels.json` | `{0: "proper_mask", 1: "no_mask", 2: "incorrect_mask"}` |
| `Model/training_history.png` | Accuracy & loss curves |

### File
- **Training**: `train_model.py`
- **Inference**: `app.py` → `FaceMaskDetector.cnn_predict()`

---

## 5. Decision Fusion Layer

### Purpose
Combine the CNN classification with HSV skin analysis to produce a robust final verdict that is more accurate than either signal alone.

### Algorithm

```python
def decide(cnn_label, confidence, nose_visible, mouth_visible):
    # Rule 1: Low confidence + full exposure → No Mask
    if confidence < 0.5 and nose_visible and mouth_visible:
        return "no_mask"

    # Rule 2: CNN says no mask → trust it
    if cnn_label == "no_mask":
        return "no_mask"

    # Rule 3: CNN says incorrect → trust it
    if cnn_label == "incorrect_mask":
        return "incorrect_mask"

    # Rule 4: CNN says proper but skin is visible → override
    if cnn_label == "proper_mask":
        if nose_visible or mouth_visible:
            return "incorrect_mask"
        return "proper_mask"
```

### Fusion Truth Table

| CNN Prediction | Confidence | Nose Visible | Mouth Visible | **Final** |
|---|---|---|---|---|
| proper_mask | ≥ 50% | No | No | ✅ Correct Mask |
| proper_mask | ≥ 50% | Yes | No | ⚠️ Incorrect Mask |
| proper_mask | ≥ 50% | Yes | Yes | ⚠️ Incorrect Mask |
| incorrect_mask | any | any | any | ⚠️ Incorrect Mask |
| no_mask | any | any | any | ❌ No Mask |
| any | < 50% | Yes | Yes | ❌ No Mask |

### Rationale
- The CNN may misclassify chin-strap or under-nose masks as "proper" — HSV analysis catches these cases.
- Low-confidence predictions with visible skin are conservatively classified as "no mask."

### File
- **Usage**: `app.py` → `FaceMaskDetector.decide()`

---

## 6. Data Augmentation Pipeline

### Purpose
Balance class distributions and increase dataset diversity to improve model generalization.

### Augmentation Techniques

| Technique | Parameters | Effect |
|-----------|-----------|--------|
| Horizontal Flip | — | Simulates left/right face orientation |
| Rotation | ±15° | Accounts for head tilt |
| Brightness | ×0.7 – ×1.3 | Simulates different lighting conditions |
| Zoom | ±10% (center crop + resize) | Simulates varying face distances |

### Balancing Strategy
1. Count samples per class after merging Dataset1 + Dataset2.
2. Determine the maximum class count.
3. Multiply by `AUGMENTATION_FACTOR` (4×) to set a target.
4. Apply augmentations cyclically to minority classes until all classes reach the target.
5. Stratified 80/20 train/val split preserves class ratios.

### File
- **Usage**: `preprocess_data.py` → `augment_image()`, `balance_and_split()`
