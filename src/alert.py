from configs.config import UNSAFE_CLASSES

def is_alert(label_id):
    return label_id in UNSAFE_CLASSES
