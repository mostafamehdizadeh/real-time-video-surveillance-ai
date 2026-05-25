from collections import defaultdict, deque

class FrameBuffer:

    def __init__(self, max_len=16):
        self.buffers = defaultdict(lambda: deque(maxlen=max_len))

    def add(self, track_id, frame):
        self.buffers[track_id].append(frame)

    def get(self, track_id):
        return self.buffers[track_id]

    def ready(self, track_id, n=16):
        return len(self.buffers[track_id]) == n
