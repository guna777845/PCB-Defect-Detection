
from ultralytics import YOLO

model = YOLO(
    "/Users/gunarpatel/runs/detect/runs/pcb/baseline/weights/last.pt"
)

print("Starting improved PCB defect training...")

results = model.train(
    data="dataset/data.yaml",
    epochs=20,
    imgsz=640,
    batch=2,
    device="mps",
    workers=0,
    cache=False,
    amp=False,
    plots=False,
    project="/Users/gunarpatel/runs/pcb",
    name="improved",
    seed=42,
    patience=7,
    save=True,
    exist_ok=False
)

print("Improved training completed.")
