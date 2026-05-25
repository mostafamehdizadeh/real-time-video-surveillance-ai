# 🔥 Unsafe-Net: Real-time Video Surveillance AI

## 🎯 Problem
Detect unsafe industrial actions in real-time using video understanding models.

---

## 🧠 Method
- YOLOv8 for object detection
- Temporal modeling using 3D CNN
- Action classification (8 classes)
- Event-based alert system

---

## 📊 Classes
Safe / Unsafe industrial behaviors:
- unsafe_walkway_violation
- unauthorized_intervention
- opened_panel_cover
- carrying_overload_with_forklift
- safe_walkway
- authorized_intervention
- closed_panel_cover
- safe_carrying

---

## 🚀 Pipeline
Input Video → YOLO Detection → Temporal Buffer → Video Model → Alert System

---

## 🛠 Tech Stack
- PyTorch
- OpenCV
- YOLOv8
- Python

---

## 📈 Status
- [x] Model training
- [x] Inference pipeline
- [ ] Multi-object tracking
- [ ] Paper writing
- [ ] Deployment

---

## 📌 Goal
Research-grade real-time surveillance AI system
