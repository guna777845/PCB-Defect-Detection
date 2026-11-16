
from pathlib import Path
from zipfile import ZipFile

DATA_DIR = Path("DeepPCB/PCBData")
GT_ZIP = Path("DeepPCB/evaluation/gt.zip")

# Official test IDs
with ZipFile(GT_ZIP) as archive:
    official_test = {
        Path(name).stem
        for name in archive.namelist()
        if name.endswith(".txt")
    }

# Collect local sample IDs
test_images = {
    p.stem.replace("_test", "")
    for p in DATA_DIR.rglob("*_test.jpg")
}

templates = {
    p.stem.replace("_temp", "")
    for p in DATA_DIR.rglob("*_temp.jpg")
}

annotations = {
    p.stem
    for p in DATA_DIR.rglob("*.txt")
}

# Identify original training samples
training_ids = test_images - official_test

print("========== DATASET AUDIT ==========")
print("Total defective images:", len(test_images))
print("Official test samples:", len(official_test))
print("Remaining training samples:", len(training_ids))

print("\n========== TRAINING CHECK ==========")
print("Training images:", len(training_ids))
print("Training templates:", len(training_ids & templates))
print("Training annotations:", len(training_ids & annotations))

print("\n========== TEST CHECK ==========")
print("Test images:", len(official_test & test_images))
print("Test templates:", len(official_test & templates))
print("Test annotations:", len(official_test & annotations))

missing_train_templates = training_ids - templates
missing_train_annotations = training_ids - annotations

print("\nMissing training templates:", len(missing_train_templates))
print("Missing training annotations:", len(missing_train_annotations))

print("\n========== FINAL RESULT ==========")

if (
    len(test_images) == 1500
    and len(official_test) == 500
    and len(training_ids) == 1000
    and len(official_test & templates) == 500
    and len(official_test & annotations) == 500
    and not missing_train_templates
    and not missing_train_annotations
):
    print("SUCCESS: Dataset is ready for splitting!")
    print("Next: 800 training + 200 validation + 500 final testing.")
else:
    print("WARNING: Investigate the counts before preparing the dataset.")
