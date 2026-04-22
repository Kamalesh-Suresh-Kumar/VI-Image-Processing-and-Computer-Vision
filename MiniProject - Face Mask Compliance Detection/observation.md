# Observation — Face Mask Compliance Detection System

## Abstract

This project implements a **real-time Face Mask Compliance Detection System** using a hybrid approach that combines classical computer vision techniques with deep learning. The system processes live webcam feeds and classifies each detected face into one of three categories — **Correct Mask**, **Incorrect Mask**, or **No Mask**. A multi-stage detection pipeline is employed: faces are first localized using Haar Cascade classifiers, then analyzed through MediaPipe Face Mesh for landmark extraction, followed by HSV-based skin visibility analysis, and finally classified by a MobileNetV2-based CNN. A decision fusion layer merges the outputs of the CNN and HSV analysis to produce a robust final verdict, reducing misclassification in edge cases such as chin-strap or under-nose masks. The system achieves approximately **92.79% validation accuracy** and is served through a Flask-based web dashboard with real-time MJPEG video streaming.

---

## Models Used

| # | Model / Algorithm | Type | Role in Pipeline |
|---|---|---|---|
| 1 | **Haar Cascade Classifier** | Classical CV (Viola-Jones) | Detects face bounding boxes in each frame using a pre-trained frontal face cascade XML |
| 2 | **MediaPipe Face Mesh** | Pre-trained ML (BlazeFace + Landmark Regression) | Extracts 468 3D facial landmarks to precisely locate eyes, nose, and mouth regions |
| 3 | **HSV Skin Color Analysis** | Rule-based CV | Detects exposed skin in the nose/mouth region to verify whether a mask truly covers the face |
| 4 | **MobileNetV2 CNN** | Deep Learning (Transfer Learning) | Classifies cropped face images into `proper_mask`, `incorrect_mask`, or `no_mask` |
| 5 | **Decision Fusion Layer** | Rule-based Logic | Combines CNN prediction with HSV skin visibility signals to produce the final classification |

---

## How the Model is Fine-Tuned

The core deep learning model (**MobileNetV2**) is fine-tuned using **transfer learning** with the following approach:

### Base Model
- **MobileNetV2** pre-trained on **ImageNet** (1.4M images, 1000 classes) is loaded as the feature extractor.
- All layers of the base model are **frozen** (`trainable = False`), meaning their weights are not updated during training. This preserves the rich, general-purpose feature representations learned from ImageNet.

### Custom Classification Head
A lightweight classification head is appended on top of the frozen base:

```
MobileNetV2 Base (frozen) → GlobalAveragePooling2D → Dropout(0.3) → Dense(128, ReLU) → Dropout(0.2) → Dense(3, Softmax)
```

Only this custom head is trained on the mask detection dataset, which allows the model to learn task-specific features while leveraging the base model's pre-learned visual representations.

### Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 1×10⁻⁴ |
| Loss Function | Categorical Cross-Entropy |
| Batch Size | 32 |
| Epochs | 20 (with Early Stopping, patience=5) |
| Input Size | 224 × 224 × 3 |
| LR Scheduler | ReduceLROnPlateau (factor=0.5, patience=3, min_lr=1×10⁻⁶) |

### Data Augmentation (during training)
Real-time augmentations are applied via `ImageDataGenerator` to improve generalization:
- Rotation (±15°), width/height shift (±10%), horizontal flip, brightness jitter (0.7–1.3×), zoom (±10%).

### Why This Approach Works
- **Freezing the base** avoids overfitting on the relatively small face mask dataset by not disturbing the robust low-level and mid-level features (edges, textures, shapes) already learned from ImageNet.
- **Training only the head** is computationally efficient and converges faster since only ~166K parameters (out of ~2.4M total) are trainable.
- **Early stopping and LR scheduling** prevent overfitting and allow the optimizer to settle into a good minimum.
