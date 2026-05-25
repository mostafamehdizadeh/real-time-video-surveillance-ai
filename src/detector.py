from ultralytics import YOLO

class PersonDetector:

    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def detect(self, frame):

        results = self.model(frame)[0]

        persons = []

        for box in results.boxes:

            cls = int(box.cls[0])

            if cls != 0:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            persons.append((x1,y1,x2,y2))

        return persons
