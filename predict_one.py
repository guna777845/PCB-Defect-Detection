from ultralytics import YOLO

# Load our improved trained model
model = YOLO(
    "/Users/gunarpatel/runs/pcb/improved/weights/best.pt"
)

# Select one PCB test image
image_path = "dataset/images/test/00041200.jpg"

print("Running PCB defect detection...")

results = model.predict(
    source=image_path,
    imgsz=640,
    conf=0.35,
    device="mps",
    save=True,
    project="/Users/gunarpatel/runs/pcb",
    name="single_prediction",
    exist_ok=True
)

print("Prediction completed!")
print("Annotated image saved in:")
print(results[0].save_dir)
