import torch
import cv2
import numpy as np
import time

from src.model import VideoSwin
from configs.config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🔥 Device:", device)

# -------------------------
# LOAD YOUR MODEL
# -------------------------
model = VideoSwin(num_classes=8).to(device)

checkpoint = torch.load(MODEL_PATH, map_location=device)
model.load_state_dict(checkpoint)

model.eval()

print("✅ Custom trained model loaded")

# -------------------------
cap = cv2.VideoCapture(VIDEO_PATH)

frames_buffer = []

while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    frames_buffer.append(frame)

    if len(frames_buffer) > NUM_FRAMES:
        frames_buffer.pop(0)

    if len(frames_buffer) == NUM_FRAMES:

        clip = np.array(frames_buffer)

        clip = torch.tensor(clip).float() / 255.0

        clip = clip.permute(0, 3, 1, 2)   # (T,C,H,W)

        clip = clip.unsqueeze(0).to(device)  # (1,T,C,H,W)

        with torch.no_grad():
            start = time.time()
            out = model(clip)
            pred = torch.argmax(out, dim=1).item()
            end = time.time()

        label = CLASS_NAMES[pred]

        print(f"🧠 {label} | {(end-start)*1000:.1f}ms")

        color = (0,0,255) if pred in UNSAFE_CLASSES else (0,255,0)

        cv2.putText(frame, label, (30,40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    cv2.imshow("Unsafe-Net v2", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
