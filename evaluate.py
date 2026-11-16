from ultralytics import YOLO

model = YOLO("/Users/gunarpatel/runs/detect/runs/pcb/baseline/weights/best.pt")

print("Evaluating PCB defect detection model...")

metrics = model.val(
    data="dataset/data.yaml",
    split="val",
    imgsz=640,
    batch=2,
    device="mps",
    workers=0,
    plots=True
)

print("\n========== VALIDATION RESULTS ==========")
print("mAP50:", metrics.box.map50)
print("mAP50-95:", metrics.box.map)
print("Precision:", metrics.box.mp)
print("Recall:", metrics.box.mr)
print("\n========== CLASS-WISE RESULTS ==========")

for i, name in enumerate(model.names.values()):
    print(name, "mAP50:", metrics.box.ap50[i])