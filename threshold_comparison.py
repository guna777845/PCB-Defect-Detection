
from ultralytics import YOLO
import os
import glob

MODEL_PATH = "/Users/gunarpatel/runs/pcb/improved/weights/best.pt"

IMAGE_DIR = "dataset/images/val"
LABEL_DIR = "dataset/labels/val"

THRESHOLDS = [0.25, 0.35, 0.45, 0.55]
IOU_THRESHOLD = 0.50

model = YOLO(MODEL_PATH)

image_paths = sorted(
    glob.glob(os.path.join(IMAGE_DIR, "*.jpg"))
)


def calculate_iou(a, b):
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])

    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


def read_labels(path, width, height):
    boxes = []

    with open(path) as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            cls = int(values[0])
            xc, yc, w, h = map(float, values[1:])

            boxes.append({
                "class": cls,
                "box": [
                    (xc - w / 2) * width,
                    (yc - h / 2) * height,
                    (xc + w / 2) * width,
                    (yc + h / 2) * height
                ]
            })

    return boxes


print("\n========== CONFIDENCE THRESHOLD COMPARISON ==========")

for threshold in THRESHOLDS:

    correct = 0
    wrong_class = 0
    missed = 0
    extra = 0
    actual_total = 0
    predicted_total = 0

    print(f"\nTesting confidence threshold: {threshold}")

    for index, image_path in enumerate(image_paths):

        filename = os.path.basename(image_path)

        label_path = os.path.join(
            LABEL_DIR,
            os.path.splitext(filename)[0] + ".txt"
        )

        if not os.path.exists(label_path):
            continue

        result = model.predict(
            source=image_path,
            imgsz=640,
            conf=threshold,
            device="mps",
            verbose=False
        )[0]

        height, width = result.orig_shape

        gt = read_labels(label_path, width, height)

        predictions = []

        if result.boxes is not None:
            for box in result.boxes:
                predictions.append({
                    "class": int(box.cls.item()),
                    "box": box.xyxy[0].cpu().numpy().tolist()
                })

        actual_total += len(gt)
        predicted_total += len(predictions)

        candidates = []

        for gi, actual in enumerate(gt):
            for pi, predicted in enumerate(predictions):

                iou = calculate_iou(
                    actual["box"],
                    predicted["box"]
                )

                if iou >= IOU_THRESHOLD:
                    candidates.append((iou, gi, pi))

        candidates.sort(reverse=True)

        matched_gt = set()
        matched_pred = set()

        for iou, gi, pi in candidates:

            if gi not in matched_gt and pi not in matched_pred:

                matched_gt.add(gi)
                matched_pred.add(pi)

                if gt[gi]["class"] == predictions[pi]["class"]:
                    correct += 1
                else:
                    wrong_class += 1

        missed += len(gt) - len(matched_gt)
        extra += len(predictions) - len(matched_pred)

        if (index + 1) % 50 == 0:
            print(f"Processed {index + 1}/{len(image_paths)} images")

    # Wrong-class matches count as both a false positive
    # and a false negative in class-aware evaluation.
    precision_denominator = correct + wrong_class + extra
    recall_denominator = correct + wrong_class + missed

    precision = (
        correct / precision_denominator * 100
        if precision_denominator else 0
    )

    recall = (
        correct / recall_denominator * 100
        if recall_denominator else 0
    )

    print("\nResults for threshold:", threshold)
    print("Actual defects:", actual_total)
    print("Predictions:", predicted_total)
    print("Correct classifications:", correct)
    print("Wrong-class matches:", wrong_class)
    print("Missed defects:", missed)
    print("Extra predictions:", extra)
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")

print("\n========== COMPARISON COMPLETED ==========")

