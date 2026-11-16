
from pathlib import Path
from collections import Counter

classes = {
    0: "open",
    1: "short",
    2: "mousebite",
    3: "spur",
    4: "spurious_copper",
    5: "pin_hole"
}

for split in ["train", "val"]:
    folder = Path(f"dataset/labels/{split}")

    counts = Counter()
    images = Counter()

    for file in folder.glob("*.txt"):
        found = set()

        for line in file.read_text().splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                counts[class_id] += 1
                found.add(class_id)

        for class_id in found:
            images[class_id] += 1

    print(f"\n========== {split.upper()} DATASET ==========")

    for class_id, name in classes.items():
        print(
            f"{name:18s} | "
            f"Defects: {counts[class_id]:4d} | "
            f"Images: {images[class_id]:4d}"
        )

