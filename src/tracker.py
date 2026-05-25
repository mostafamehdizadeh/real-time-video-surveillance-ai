class SimpleTracker:

    def __init__(self):
        self.next_id = 0

    def update(self, detections):

        tracked = []

        for det in detections:

            tracked.append((self.next_id, det))
            self.next_id += 1

        return tracked
