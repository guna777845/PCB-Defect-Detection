
from ultralytics import YOLO
import torch

print("========== SYSTEM CHECK ==========")
print("PyTorch:", torch.__version__)
print("MPS available:", torch.backends.mps.is_available())

if not torch.backends.mps.is_available():
    raise RuntimeError("Apple GPU acceleration is unavailable.")

print("\nLoading YOLOv8 Nano model...")

model = YOLO("yolov8n.pt")

print("\nStarting PCB defect detection training...")

results = model.train(
    data="dataset/data.yaml",
    epochs=5,
    imgsz=640,
    batch=2,
    device="mps",
    workers=0,
    cache=False,
    amp=False,
    plots=False,
    project="runs/pcb",
    name="baseline",
    seed=42,
    patience=5,
    save=True,
    exist_ok=False
)

print("\n========== TRAINING FINISHED ==========")
print("Best model: runs/pcb/baseline/weights/best.pt")
print("Last model: runs/pcb/baseline/weights/last.pt")
