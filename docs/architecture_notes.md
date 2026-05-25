# 🧠 Architecture Overview

## Pipeline:
1. Input Video
2. YOLOv11 Object Detection
3. Temporal Buffer (16 frames)
4. 3D CNN Model
5. Softmax Classification
6. Event Alert System

## Key Components:
- Spatial feature extraction (YOLOv8)
- Temporal modeling (3D CNN)
- Event memory for smoothing predictions

## Limitations:
- No tracking module yet
- No transformer-based temporal attention
