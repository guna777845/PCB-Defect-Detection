
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = "/Users/gunarpatel/runs/pcb/improved/weights/best.pt"
IMAGE_DIR = Path("dataset/images/val")
LABEL_DIR = Path("dataset/labels/val")

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.50
NUMBER_OF_IMAGES = 10

model = YOLO(MODEL_PATH)

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


images = sorted(IMAGE_DIR.glob("*.jpg"))[:NUMBER_OF_IMAGES]

total_tp = 0
total_missed = 0
total_extra = 0

print("\n========== PCB PREDICTION VALIDATION ==========")

for image_path in images:
    label_path = LABEL_DIR / (image_path.stem + ".txt")

    ground_truth = []

    with open(label_path, "r") as file:
        for line in file:
            values = list(map(float, line.split()))
            class_id = int(values[0])

            xc, yc, w, h = values[1:]

            box = [
                (xc - w / 2) * 640,
                (yc - h / 2) * 640,
                (xc + w / 2) * 640,
                (yc + h / 2) * 640
            ]

            ground_truth.append((class_id, box))

    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=CONFIDENCE,
        device="mps",
        verbose=False
    )[0]

    predictions = []

    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            coordinates = box.xyxy[0].tolist()
            predictions.append((class_id, coordinates))

    matched_predictions = set()
    tp = 0

    for gt_class, gt_box in ground_truth:
        best_iou = 0
        best_index = None

        for index, (pred_class, pred_box) in enumerate(predictions):
            if index in matched_predictions:
                continue

            if pred_class != gt_class:
                continue

            iou = calculate_iou(gt_box, pred_box)

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if best_index is not None and best_iou >= IOU_THRESHOLD:
            tp += 1
            matched_predictions.add(best_index)

    missed = len(ground_truth) - tp
    extra = len(predictions) - tp

    total_tp += tp
    total_missed += missed
    total_extra += extra

    print(f"\nImage: {image_path.name}")
    print(f"Actual defects: {len(ground_truth)}")
    print(f"Predicted defects: {len(predictions)}")
    print(f"Correct detections: {tp}")
    print(f"Missed detections: {missed}")
    print(f"Extra predictions: {extra}")

print("\n========== OVERALL SUMMARY ==========")
print("Correct detections:", total_tp)
print("Missed detections:", total_missed)
print("Extra predictions:", total_extra)
print("Total actual defects:", total_tp + total_missed)

print("\nValidation completed!")

