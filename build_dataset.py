
from pathlib import Path
from zipfile import ZipFile
import random
import shutil

SOURCE = Path("DeepPCB/PCBData")
GT_ZIP = Path("DeepPCB/evaluation/gt.zip")
OUTPUT = Path("dataset")

IMAGE_SIZE = 640
SEED = 42

# Stop if a dataset folder already exists.
if OUTPUT.exists():
    raise SystemExit(
        "ERROR: dataset folder already exists. "
        "Rename or inspect it before running this script again."
    )

# Read the official 500 test IDs.
with ZipFile(GT_ZIP) as archive:
    official_test = {
        Path(name).stem
        for name in archive.namelist()
        if name.endswith(".txt")
    }

# Find local images and annotations.
images = {
    p.stem.replace("_test", ""): p
    for p in SOURCE.rglob("*_test.jpg")
}

annotations = {
    p.stem: p
    for p in SOURCE.rglob("*.txt")
}

all_ids = set(images)
training_pool = sorted(all_ids - official_test)
test_ids = sorted(official_test)

if len(training_pool) != 1000 or len(test_ids) != 500:
    raise SystemExit("ERROR: Unexpected dataset counts.")

# Reproducible 80/20 split of the original training pool.
random.Random(SEED).shuffle(training_pool)

train_ids = training_pool[:800]
val_ids = training_pool[800:]

splits = {
    "train": train_ids,
    "val": val_ids,
    "test": test_ids,
}

# Convert original annotation rows to YOLO format.
def convert_annotation(annotation_path):
    output_lines = []

    for line_number, line in enumerate(
        annotation_path.read_text().splitlines(), start=1
    ):
        if not line.strip():
            continue

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Invalid annotation in {annotation_path}, "
                f"line {line_number}"
            )

        x1, y1, x2, y2 = map(float, parts[:4])
        original_class = int(parts[4])

        if original_class not in range(1, 7):
            raise ValueError(
                f"Unknown class ID in {annotation_path}"
            )

        if not (
            0 <= x1 < x2 <= IMAGE_SIZE
            and 0 <= y1 < y2 <= IMAGE_SIZE
        ):
            raise ValueError(
                f"Invalid bounding box in {annotation_path}: {line}"
            )

        # Convert original class IDs 1-6 to YOLO IDs 0-5.
        class_id = original_class - 1

        # Convert corners to normalized YOLO coordinates.
        center_x = ((x1 + x2) / 2) / IMAGE_SIZE
        center_y = ((y1 + y2) / 2) / IMAGE_SIZE
        width = (x2 - x1) / IMAGE_SIZE
        height = (y2 - y1) / IMAGE_SIZE

        output_lines.append(
            f"{class_id} {center_x:.6f} {center_y:.6f} "
            f"{width:.6f} {height:.6f}"
        )

    if not output_lines:
        raise ValueError(f"No defects found in {annotation_path}")

    return "\n".join(output_lines) + "\n"


# Create folders and copy images and labels.
for split_name, sample_ids in splits.items():
    image_dir = OUTPUT / "images" / split_name
    label_dir = OUTPUT / "labels" / split_name

    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    for sample_id in sample_ids:
        if sample_id not in images or sample_id not in annotations:
            raise FileNotFoundError(f"Missing sample: {sample_id}")

        shutil.copy2(
            images[sample_id],
            image_dir / f"{sample_id}.jpg"
        )

        yolo_text = convert_annotation(annotations[sample_id])

        (label_dir / f"{sample_id}.txt").write_text(yolo_text)

# Create YOLO configuration.
yaml_text = """path: dataset
train: images/train
val: images/val
test: images/test

names:
  0: open
  1: short
  2: mousebite
  3: spur
  4: spurious_copper
  5: pin_hole
"""

(OUTPUT / "data.yaml").write_text(yaml_text)

print("\n========== DATASET CREATED ==========")

for split_name, sample_ids in splits.items():
    print(
        f"{split_name.upper()}: "
        f"{len(sample_ids)} images, "
        f"{len(sample_ids)} labels"
    )

print("\nYOLO classes: 6")
print("Random seed:", SEED)
print("Configuration:", OUTPUT / "data.yaml")
print("\nSUCCESS: Dataset preparation complete!")
