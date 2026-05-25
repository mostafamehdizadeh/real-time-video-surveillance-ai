from collections import defaultdict

class EventMemory:

    def __init__(self):
        self.memory = defaultdict(list)

    def update(self, track_id, label):

        self.memory[track_id].append(label)

        if len(self.memory[track_id]) > 10:
            self.memory[track_id].pop(0)

        return max(set(self.memory[track_id]),
                   key=self.memory[track_id].count)
