# 📊 Dataset

This project uses a custom video dataset for industrial safety monitoring.

## Classes (8):
- unsafe_walkway_violation
- unauthorized_intervention
- opened_panel_cover
- carrying_overload_with_forklift
- safe_walkway
- authorized_intervention
- closed_panel_cover
- safe_carrying

## Format:
- Input: video clips (16 frames each)
- Resolution: 224x224
- FPS: variable (normalized during preprocessing)

## Preprocessing:
- Frame extraction
- Resize to 224x224
- Temporal sampling (16 frames per clip)
