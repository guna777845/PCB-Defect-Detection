
from ultralytics import YOLO

model = YOLO(
    "/Users/gunarpatel/runs/pcb/improved/weights/best.pt"
)

image_path = "dataset/images/val/00041044.jpg"
label_path = "dataset/labels/val/00041044.txt"

names = model.names

print("\n========== ACTUAL ANNOTATIONS ==========")

with open(label_path, "r") as f:
    for line in f:
        values = list(map(float, line.split()))
        class_id = int(values[0])
        print(
            f"Class: {names[class_id]:18s} "
            f"Box: {values[1:]}"
        )

print("\n========== MODEL PREDICTIONS ==========")

results = model.predict(
    source=image_path,
    imgsz=640,
    conf=0.25,
    device="mps",
    verbose=False
)

for box in results[0].boxes:
    class_id = int(box.cls.item())
    confidence = float(box.conf.item())
    coordinates = box.xyxy[0].tolist()

    print(
        f"Class: {names[class_id]:18s} "
        f"Confidence: {confidence:.3f} "
        f"Box: {[round(x, 2) for x in coordinates]}"
    )

