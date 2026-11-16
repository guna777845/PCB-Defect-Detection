# AI-Based PCB Defect Detection and Inspection System

An AI-powered PCB inspection system that uses **YOLOv8 Nano** to detect and identify defects in Printed Circuit Boards (PCBs).

The project includes a trained deep learning model and a Flask web application that allows users to upload PCB images and view detected defects with bounding boxes.

## Features

- Automatic PCB defect detection using YOLOv8 Nano.
- Upload PCB images through a web interface.
- Display detected defects with bounding boxes.
- Show the total number of detected defects.
- Support for six PCB defect categories.
- Trained model included for inference.

## Defect Classes

The model detects these six types of PCB defects:

1. Open
2. Short
3. Mousebite
4. Spur
5. Spurious Copper
6. Pin Hole

## Technologies Used

- Python
- YOLOv8 (Ultralytics)
- PyTorch
- OpenCV
- Flask
- HTML, CSS and JavaScript

## Model Performance

The trained model was evaluated on the official DeepPCB test set containing 500 images.

| Metric | Result |
|---|---:|
| Precision | 93.65% |
| Recall | 88.87% |
| mAP@50 | 94.62% |
| mAP@50-95 | 62.08% |

These results are from the held-out official test set.

## Project Structure

```text
PCB-Defect-Detection/
├── application/
│   ├── app.py
│   ├── templates/
│   │   ├── index.html
│   │   └── result.html
│   ├── static/
│   ├── uploads/
│   └── results/
├── models/
│   └── best.pt
├── dataset/
├── train.py
├── train_more.py
├── evaluate.py
├── evaluate_test.py
├── predict_one.py
├── requirements.txt
└── README.md
```

Generated data and local environment folders are excluded from Git.

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/guna777845/PCB-Defect-Detection.git
cd PCB-Defect-Detection
```

### 2. Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Application

Start the Flask application:

```bash
python application/app.py
```

Open the following address in your browser:

```text
http://127.0.0.1:5001
```

Upload a PCB image to view the detection results.

## Dataset

This project uses the **DeepPCB dataset** for training, validation and testing.

The dataset is not included in this repository. Obtain it separately before running the training scripts.

The trained model is included in `models/best.pt`, so the web application can be used for inference without downloading the training dataset.

## Important Note

This system performs image-based PCB defect detection. It identifies visible defects in PCB images and does not perform electrical testing or guarantee PCB functionality.

## Project Status

The YOLOv8 Nano model has been trained and evaluated. A Flask-based web application is implemented and tested locally.

## Contributors

Developed as an academic project in AI-based PCB defect inspection.
