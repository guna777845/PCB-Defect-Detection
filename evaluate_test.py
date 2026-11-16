
from ultralytics import YOLO

model = YOLO(
    "/Users/gunarpatel/runs/pcb/improved/weights/best.pt"
)

print("Evaluating on official PCB test dataset...")

metrics = model.val(
    data="dataset/data.yaml",
    split="test",
    imgsz=640,
    batch=2,
    device="mps",
    workers=0,
    plots=True,
    project="/Users/gunarpatel/runs/pcb",
    name="official_test_evaluation",
    exist_ok=False
)

print("\n========== OFFICIAL TEST RESULTS ==========")
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)

print("\n========== CLASS-WISE RESULTS ==========")

for i, name in enumerate(model.names.values()):
    print(name, "mAP50:", metrics.box.ap50[i])
