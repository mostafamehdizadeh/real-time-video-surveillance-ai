import json
from datetime import datetime

def log_event(track_id, label):

    data = {
        "time": str(datetime.now()),
        "track_id": track_id,
        "event": label
    }

    with open("events.json", "a") as f:
        f.write(json.dumps(data) + "\n")
