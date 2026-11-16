
from flask import Flask, request, render_template
from ultralytics import YOLO
from werkzeug.utils import secure_filename
from pathlib import Path
from collections import Counter
import uuid
import os

# Initialize Flask
app = Flask(__name__)

# Application directories
BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FOLDER = BASE_DIR / "results" / "web_inspection"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

# Upload configuration
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

# Trained YOLO model
MODEL_PATH = "/Users/gunarpatel/runs/pcb/improved/weights/best.pt"

print("Loading PCB defect detection model...")

model = YOLO(MODEL_PATH)

print("Model loaded successfully!")

CONFIDENCE_THRESHOLD = 0.35


def allowed_file(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


@app.route("/", methods=["GET", "POST"])
def index():

    if request.method == "POST":

        if "pcb_image" not in request.files:
            return "No image uploaded", 400

        file = request.files["pcb_image"]

        if file.filename == "":
            return "Please select an image", 400

        if not allowed_file(file.filename):
            return "Only JPG, JPEG and PNG images are allowed", 400

        # Create a unique filename
        original_name = secure_filename(file.filename)

        filename = f"{uuid.uuid4().hex}_{original_name}"

        upload_path = UPLOAD_FOLDER / filename

        file.save(upload_path)

        print("Processing image:", filename)

        # Run YOLO prediction
        results = model.predict(
            source=str(upload_path),
            imgsz=640,
            conf=CONFIDENCE_THRESHOLD,
            device="mps",
            save=True,
            project=str(BASE_DIR / "results"),
            name="web_inspection",
            exist_ok=True,
            verbose=False
        )

        result = results[0]

        # Count detected defects
        defect_counts = Counter()

        if result.boxes is not None:
            for box in result.boxes:
                class_id = int(box.cls.item())
                defect_name = model.names[class_id]
                defect_counts[defect_name] += 1

        total_defects = sum(defect_counts.values())

        print("Detection completed!")
        print("Total defects:", total_defects)

        return render_template(
            "result.html",
            image_name=filename,
            defect_counts=dict(defect_counts),
            total_defects=total_defects,
            confidence=CONFIDENCE_THRESHOLD
        )

    return render_template("index.html")


@app.route("/results/<filename>")
def result_image(filename):
    from flask import send_from_directory

    return send_from_directory(
        RESULT_FOLDER,
        filename
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=False
    )

