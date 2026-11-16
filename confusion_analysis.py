
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

MODEL_PATH = "/Users/gunarpatel/runs/pcb/improved/weights/best.pt"

IMAGE_DIR = Path("dataset/images/val")
LABEL_DIR = Path("dataset/labels/val")

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.50

model = YOLO(MODEL_PATH)
names = model.names

confusion = Counter()
missed = Counter()
extra = Counter()

def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2-x1) * max(0, y2-y1)

    area1 = max(0, box1[2]-box1[0]) * max(0, box1[3]-box1[1])
    area2 = max(0, box2[2]-box2[0]) * max(0, box2[3]-box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union else 0

images = sorted(IMAGE_DIR.glob("*.jpg"))

for image_path in images:

    label_path = LABEL_DIR / (image_path.stem + ".txt")

    ground_truth = []

    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue

        values = list(map(float, line.split()))
        cls = int(values[0])

        xc, yc, w, h = values[1:]

        box = [
            (xc-w/2)*640,
            (yc-h/2)*640,
            (xc+w/2)*640,
            (yc+h/2)*640
        ]

        ground_truth.append((cls, box))

    result = model.predict(
        source=str(image_path),
        imgsz=640,
        conf=CONFIDENCE,
        device="mps",
        verbose=False
    )[0]

    predictions = []

    for box in result.boxes:
        cls = int(box.cls.item())
        coords = box.xyxy[0].tolist()
        predictions.append((cls, coords))

    candidates = []

    for gi, (gt_cls, gt_box) in enumerate(ground_truth):
        for pi, (pred_cls, pred_box) in enumerate(predictions):
            overlap = iou(gt_box, pred_box)

            if overlap >= IOU_THRESHOLD:
                candidates.append((overlap, gi, pi))

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()

    for overlap, gi, pi in candidates:
        if gi in matched_gt or pi in matched_pred:
            continue

        gt_cls = ground_truth[gi][0]
        pred_cls = predictions[pi][0]

        matched_gt.add(gi)
        matched_pred.add(pi)

        confusion[(gt_cls, pred_cls)] += 1

    for gi, (gt_cls, _) in enumerate(ground_truth):
        if gi not in matched_gt:
            missed[gt_cls] += 1

    for pi, (pred_cls, _) in enumerate(predictions):
        if pi not in matched_pred:
            extra[pred_cls] += 1

print("\n========== CLASS CONFUSION MATRIX ==========")

print("Actual \\ Predicted")

print("Actual classes:", names)

for gt_cls in range(len(names)):
    row = []

    for pred_cls in range(len(names)):
        row.append(confusion[(gt_cls, pred_cls)])

    print(names[gt_cls], row)

print("\n========== MISSED DEFECTS ==========")

for cls, name in names.items():
    print(name, missed[cls])

print("\n========== EXTRA PREDICTIONS ==========")

for cls, name in names.items():
    print(name, extra[cls])

print("\nAnalysis completed on", len(images), "validation images.")

