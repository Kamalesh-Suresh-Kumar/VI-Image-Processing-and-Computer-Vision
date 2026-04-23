# Face Mask Compliance Detection System

**Author:** [Kamalesh S P](https://github.com/Kamalesh-Suresh-Kumar)

## Project Overview

A real-time **hybrid AI system** for detecting face mask compliance using a multi-stage pipeline combining classical computer vision with deep learning. The system identifies three states: **Correct Mask**, **Incorrect Mask**, and **No Mask**, and streams annotated video through a premium web dashboard.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: Webcam Frame                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
        ┌──────────────▼──────────────┐
        │  1. Haar Cascade Face       │
        │     Detection               │
        │     → Bounding boxes        │
        └──────────────┬──────────────┘
                       │
          ┌────────────┴────────────┐
          │                         │
  ┌───────▼───────┐       ┌────────▼────────┐
  │ 2. MediaPipe  │       │ 4. MobileNetV2  │
  │    Face Mesh  │       │    CNN Classifier│
  │  (468 landmarks)│     │  (3-class softmax)│
  └───────┬───────┘       └────────┬────────┘
          │                        │
  ┌───────▼───────┐                │
  │ 3. HSV Skin   │                │
  │    Analysis   │                │
  │  (nose/mouth  │                │
  │   visibility) │                │
  └───────┬───────┘                │
          │                        │
          └───────┬────────────────┘
                  │
         ┌────────▼────────┐
         │ 5. Decision     │
         │    Fusion Layer │
         └────────┬────────┘
                  │
         ┌────────▼────────┐
         │ OUTPUT: Annotated│
         │ Frame + Label    │
         └─────────────────┘
```

## Detection Pipeline

| Stage | Technology | Purpose |
|-------|-----------|---------|
| 1 | Haar Cascade (OpenCV) | Face bounding box detection |
| 2 | MediaPipe Face Mesh | 468-point facial landmark extraction |
| 3 | HSV Color Analysis | Skin exposure detection in nose/mouth ROI |
| 4 | MobileNetV2 (TensorFlow/Keras) | CNN-based mask classification |
| 5 | Decision Fusion | Combine signals for final verdict |

## Decision Fusion Logic

| CNN Prediction | Nose Visible | Mouth Visible | **Final Decision** |
|---|---|---|---|
| proper_mask | No | No | ✅ Correct Mask |
| proper_mask | Yes | No | ⚠️ Incorrect Mask |
| proper_mask | Yes | Yes | ⚠️ Incorrect Mask |
| incorrect_mask | * | * | ⚠️ Incorrect Mask |
| no_mask | * | * | ❌ No Mask |
| * (confidence < 50%) | Yes | Yes | ❌ No Mask |

## Project Structure

```
MiniProject - Face Mask Compliance Detection/
├── app.py                     # Flask backend — full detection pipeline + web server
├── preprocess_data.py         # Data preprocessing — crop, augment, balance, split
├── train_model.py             # CNN training — MobileNetV2 transfer learning
├── requirements.txt           # Python dependencies
├── Project.md                 # This file — project documentation
├── ALGORITHMS_DOCUMENTATION.md# Detailed algorithm descriptions
│
├── Data/
│   └── haarcascade_frontalface_default.xml
│
├── Model/
│   ├── mask_detector_mobilenetv2.h5    # Trained CNN weights
│   ├── class_labels.json               # Class index → label mapping
│   ├── face_landmarker.task            # MediaPipe Face Landmark Task
│   └── training_history.png            # Accuracy/loss plots
│
├── Template/
│   └── face.html              # Dashboard HTML (Jinja2)
│
└── Script/
    ├── face.css               # Dashboard CSS (glassmorphism dark theme)
    └── face.js                # Dashboard JS (live stats polling)
```

## Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Preprocess Data
```bash
python preprocess_data.py
```
This parses both datasets, crops faces, augments to balance classes, resizes to 224×224, and splits into train/val sets.

### 3. Train Model
```bash
python train_model.py
```
Trains MobileNetV2 with transfer learning (~20 epochs). Saves model to `Model/`.

### 4. Run Detection System
```bash
python app.py
```
Open browser at **http://localhost:5000** to view the live dashboard.

## Datasets

**🌟 Official Project Dataset:** [Face Mask Compliance Detection System Dataset v1](https://www.kaggle.com/datasets/kamaleshsp/face-mask-compliance-detection-system-dataset-v1)

This final, balanced dataset was pre-processed, merged, and augmented from the following source datasets:

| Dataset | Source | Description |
|---------|--------|-------------|
| Dataset1 | [Andrew MVD — Face Mask Detection](https://www.kaggle.com/datasets/andrewmvd/face-mask-detection) | 853 images with Pascal VOC XML annotations. Multi-face bounding boxes. Labels: `with_mask`, `without_mask`, `mask_weared_incorrect` |
| Dataset2 | [Shiekhburhan — Face Mask Dataset](https://www.kaggle.com/datasets/shiekhburhan/face-mask-dataset) | ~14,500 pre-cropped face images in class folders with `simple`/`complex` subfolders |
| Haar Cascade XML | [OpenCV — haarcascade_frontalface_default.xml](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml) | Pre-trained frontal face detector cascade |

## Model Performance

- **Architecture**: MobileNetV2 (ImageNet pre-trained) + custom classification head
- **Input Size**: 224 × 224 × 3
- **Classes**: `proper_mask`, `incorrect_mask`, `no_mask`
- **Validation Accuracy**: ~92.79%
- **F1 Score**: ~92.80

## Technologies

- **Python 3.10+**
- **TensorFlow / Keras** — CNN training & inference
- **OpenCV** — image processing, Haar cascades, video capture
- **MediaPipe** — face mesh landmark detection
- **Flask** — web server, MJPEG streaming
- **scikit-learn** — train/test split, metrics
- **Pillow / Matplotlib** — image augmentation, plots

