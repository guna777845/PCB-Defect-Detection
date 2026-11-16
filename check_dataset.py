
from pathlib import Path
from zipfile import ZipFile

DATA_DIR = Path("DeepPCB/PCBData")
GT_ZIP = Path("DeepPCB/evaluation/gt.zip")

# Read official test filenames
with ZipFile(GT_ZIP) as archive:
    test_ids = {
        Path(name).stem
        for name in archive.namelist()
        if name.endswith(".txt")
    }

# Find all local files
test_images = {
    p.stem.replace("_test", ""): p
    for p in DATA_DIR.rglob("*_test.jpg")
}

template_images = {
    p.stem.replace("_temp", ""): p
    for p in DATA_DIR.rglob("*_temp.jpg")
}

annotations = {
    p.stem: p
    for p in DATA_DIR.rglob("*.txt")
}

missing_test = test_ids - test_images.keys()
missing_template = test_ids - template_images.keys()
missing_annotations = test_ids - annotations.keys()

print("Official test images:", len(test_ids))
print("Matching test images:", len(test_ids) - len(missing_test))
print("Matching templates:", len(test_ids) - len(missing_template))
print("Matching annotations:", len(test_ids) - len(missing_annotations))

print("\nMissing test images:", len(missing_test))
print("Missing templates:", len(missing_template))
print("Missing annotations:", len(missing_annotations))

if not missing_test and not missing_template and not missing_annotations:
    print("\nSUCCESS: All 500 official test samples match!")
else:
    print("\nSome files need investigation.")
