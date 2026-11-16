
from ultralytics import YOLO
import cv2
import os
import glob
import numpy as np
import csv

MODEL_PATH = "/Users/gunarpatel/runs/pcb/improved/weights/best.pt"

IMAGE_DIR = "dataset/images/val"
LABEL_DIR = "dataset/labels/val"

OUTPUT_DIR = "/Users/gunarpatel/runs/pcb/visual_error_analysis"

CONFIDENCE = 0.25
IOU_THRESHOLD = 0.50

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)

CLASS_NAMES = model.names

COLORS = {
    "correct": (0, 200, 0),       # Green
    "wrong": (0, 255, 255),       # Yellow
    "extra": (0, 0, 255),         # Red
    "missed": (0, 165, 255)       # Orange
}


def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0


def read_ground_truth(label_path, width, height):
    ground_truth = []

    with open(label_path, "r") as file:
        for line in file:
            values = line.strip().split()

            if len(values) != 5:
                continue

            cls = int(values[0])
            xc, yc, w, h = map(float, values[1:])

            x1 = (xc - w / 2) * width
            y1 = (yc - h / 2) * height
            x2 = (xc + w / 2) * width
            y2 = (yc + h / 2) * height

            ground_truth.append({
                "class": cls,
                "box": [x1, y1, x2, y2]
            })

    return ground_truth


def draw_box(image, box, color, label, thickness=2):
    x1, y1, x2, y2 = map(int, box)

    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        thickness
    )

    cv2.putText(
        image,
        label,
        (x1, max(18, y1 - 6)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA
    )


image_paths = sorted(glob.glob(os.path.join(IMAGE_DIR, "*.jpg")))

summary = []
images_with_errors = 0

for image_path in image_paths:

    filename = os.path.basename(image_path)
    label_path = os.path.join(
        LABEL_DIR,
        os.path.splitext(filename)[0] + ".txt"
    )

    image = cv2.imread(image_path)

    if image is None or not os.path.exists(label_path):
        print("Skipping:", filename)
        continue

    height, width = image.shape[:2]

    gt = read_ground_truth(label_path, width, height)

    result = model.predict(
        source=image_path,
        imgsz=640,
        conf=CONFIDENCE,
        device="mps",
        verbose=False
    )[0]

    predictions = []

    if result.boxes is not None:
        for box in result.boxes:
            predictions.append({
                "class": int(box.cls.item()),
                "confidence": float(box.conf.item()),
                "box": box.xyxy[0].cpu().numpy().tolist()
            })

    # Match ground-truth boxes and predictions by IoU,
    # independently of class.
    candidates = []

    for gi, actual in enumerate(gt):
        for pi, predicted in enumerate(predictions):
            iou = calculate_iou(actual["box"], predicted["box"])

            if iou >= IOU_THRESHOLD:
                candidates.append((iou, gi, pi))

    candidates.sort(reverse=True)

    matched_gt = set()
    matched_pred = set()
    matches = []

    for iou, gi, pi in candidates:
        if gi not in matched_gt and pi not in matched_pred:
            matched_gt.add(gi)
            matched_pred.add(pi)
            matches.append((gi, pi))

    missed = [i for i in range(len(gt)) if i not in matched_gt]
    extra = [i for i in range(len(predictions)) if i not in matched_pred]

    wrong_class = []
    correct = []

    for gi, pi in matches:
        if gt[gi]["class"] == predictions[pi]["class"]:
            correct.append((gi, pi))
        else:
            wrong_class.append((gi, pi))

    has_errors = bool(missed or extra or wrong_class)

    if has_errors:
        images_with_errors += 1

        # Correct detections: green
        for gi, pi in correct:
            pred = predictions[pi]
            name = CLASS_NAMES[pred["class"]]

            draw_box(
                image,
                pred["box"],
                COLORS["correct"],
                f"CORRECT: {name} {pred['confidence']:.2f}"
            )

        # Wrong class: yellow
        for gi, pi in wrong_class:
            actual_name = CLASS_NAMES[gt[gi]["class"]]
            pred = predictions[pi]
            predicted_name = CLASS_NAMES[pred["class"]]

            draw_box(
                image,
                gt[gi]["box"],
                COLORS["wrong"],
                f"ACTUAL: {actual_name}"
            )

            draw_box(
                image,
                pred["box"],
                COLORS["wrong"],
                f"WRONG: {predicted_name} {pred['confidence']:.2f}"
            )

        # Missed defects: orange
        for gi in missed:
            actual_name = CLASS_NAMES[gt[gi]["class"]]

            draw_box(
                image,
                gt[gi]["box"],
                COLORS["missed"],
                f"MISSED: {actual_name}"
            )

        # Extra predictions: red
        for pi in extra:
            pred = predictions[pi]
            name = CLASS_NAMES[pred["class"]]

            draw_box(
                image,
                pred["box"],
                COLORS["extra"],
                f"EXTRA: {name} {pred['confidence']:.2f}"
            )

        cv2.imwrite(
            os.path.join(OUTPUT_DIR, filename),
            image
        )

    summary.append([
        filename,
        len(gt),
        len(predictions),
        len(correct),
        len(wrong_class),
        len(missed),
        len(extra)
    ])

    print(
        filename,
        "| Correct:", len(correct),
        "| Wrong class:", len(wrong_class),
        "| Missed:", len(missed),
        "| Extra:", len(extra)
    )


csv_path = os.path.join(OUTPUT_DIR, "error_summary.csv")

with open(csv_path, "w", newline="") as file:
    writer = csv.writer(file)

    writer.writerow([
        "Image",
        "Actual",
        "Predicted",
        "Correct",
        "Wrong_Class",
        "Missed",
        "Extra"
    ])

    writer.writerows(summary)


print("\n========== VISUAL ERROR ANALYSIS ==========")
print("Images processed:", len(summary))
print("Images containing errors:", images_with_errors)
print("Annotated images saved in:", OUTPUT_DIR)
print("CSV summary saved in:", csv_path)

