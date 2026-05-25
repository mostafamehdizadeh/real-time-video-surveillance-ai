VIDEO_PATH = "0_te26.mp4"
MODEL_PATH = "best_model.pth"

IMG_SIZE = 224
NUM_FRAMES = 16

CLASS_NAMES = [
    "safe_walkway_violation",
    "unauthorized_intervention",
    "opened_panel_cover",
    "carrying_overload_with_forklift",
    "safe_walkway",
    "authorized_intervention",
    "closed_panel_cover",
    "safe_carrying"
]

UNSAFE_CLASSES = {
    0,1,2,3
}
