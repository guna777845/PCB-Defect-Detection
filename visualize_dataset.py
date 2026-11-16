
from pathlib import Path
from PIL import Image, ImageDraw
import random

DATASET = Path("dataset")
PREVIEW = Path("dataset_preview")

CLASSES = [
    "Open",
    "Short",
    "Mousebite",
    "Spur",
    "Spurious copper",
    "Pin hole"
]

COLORS = [
    "red", "blue", "green",
    "orange", "purple", "magenta"
]

PREVIEW.mkdir(exist_ok=True)

random.seed(42)

for split in ["train", "val", "test"]:
    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    images = sorted(image_dir.glob("*.jpg"))

    selected = random.sample(images, min(3, len(images)))

    for image_path in selected:
        label_path = label_dir / (image_path.stem + ".txt")

        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)

        image_width, image_height = image.size

        for line in label_path.read_text().splitlines():
            class_id, cx, cy, w, h = map(float, line.split())
            class_id = int(class_id)

            x1 = (cx - w / 2) * image_width
            y1 = (cy - h / 2) * image_height
            x2 = (cx + w / 2) * image_width
            y2 = (cy + h / 2) * image_height

            color = COLORS[class_id]

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=color,
                width=3
            )

            draw.text(
                (x1, max(0, y1 - 15)),
                CLASSES[class_id],
                fill=color
            )

        output_path = PREVIEW / f"{split}_{image_path.stem}.jpg"
        image.save(output_path)

        print("Saved:", output_path)

print("\nVisual previews created successfully!")

