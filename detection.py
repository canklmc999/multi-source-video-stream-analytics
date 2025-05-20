from ultralytics import YOLO

model = YOLO("yolo11s.pt")  

def detect_objects(frame):
    results = model(frame, verbose=False)
    return results[0].plot()  # Bounding Boxes and Results.