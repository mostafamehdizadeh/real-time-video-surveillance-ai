import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import time
import os
import random
import numpy as np

from dataset import VideoDataset
from src.model import VideoSwin

# =====================================================
# SEED
# =====================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(42)

# =====================================================
# DEVICE
# =====================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("🔥 Device:", device)

# =====================================================
# LOAD DATASET (SAFE MODE)
# =====================================================
print("📦 Loading dataset...")

dataset = VideoDataset(
    "dataset/train",
    num_frames=16
)

print("✅ Dataset loaded:", len(dataset))

# =====================================================
# SPLIT
# =====================================================
train_size = int(0.85 * len(dataset))
val_size = len(dataset) - train_size

train_ds, val_ds = random_split(dataset, [train_size, val_size])

# =====================================================
# DATA LOADERS (FIXED)
# =====================================================
train_loader = DataLoader(
    train_ds,
    batch_size=8,
    shuffle=True,
    num_workers=4,        # 🔥 IMPORTANT
    pin_memory=True,      # 🔥 GPU fast transfer
    persistent_workers=True,
    prefetch_factor=2
)

val_loader = DataLoader(
    val_ds,
    batch_size=8,
    shuffle=False,
    num_workers=2,
    pin_memory=True
)

print(f"📦 Train: {len(train_ds)} | Val: {len(val_ds)}")

# =====================================================
# MODEL
# =====================================================
model = VideoSwin(num_classes=8).to(device)

criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    weight_decay=1e-4
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=10
)

scaler = torch.amp.GradScaler()

# =====================================================
# TRAIN SETTINGS
# =====================================================
epochs = 20
best_acc = 0

# =====================================================
# TRAIN LOOP
# =====================================================
for epoch in range(epochs):

    print(f"\n🔥 Epoch {epoch+1}/{epochs} START")
    start_time = time.time()

    # ================= TRAIN =================
    model.train()
    train_loss = 0

    for i, (clips, labels) in enumerate(train_loader):

        clips = clips.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda"):
            outputs = model(clips)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        train_loss += loss.item()

        if i % 10 == 0:
            print(f"[Train] step {i} loss={loss.item():.4f}")

    avg_train_loss = train_loss / len(train_loader)

    # ================= VAL =================
    model.eval()
    correct = 0
    total = 0
    val_loss = 0

    with torch.no_grad():
        for clips, labels in val_loader:

            clips = clips.permute(0, 2, 1, 3, 4).to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(clips)
            loss = criterion(outputs, labels)

            val_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    acc = correct / total
    avg_val_loss = val_loss / len(val_loader)

    scheduler.step()

    epoch_time = time.time() - start_time

    print("\n====================================")
    print(f"🔥 Epoch {epoch+1}")
    print(f"⏱ Time: {epoch_time:.2f}s")
    print(f"📉 Train Loss: {avg_train_loss:.4f}")
    print(f"📉 Val Loss: {avg_val_loss:.4f}")
    print(f"🎯 Val Acc: {acc*100:.2f}%")
    print("====================================\n")

    # ================= SAVE =================
    torch.save(model.state_dict(), f"checkpoints/epoch_{epoch+1}.pth")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "best_model.pth")
        print("🏆 BEST MODEL SAVED!")

print("✅ TRAINING DONE")
