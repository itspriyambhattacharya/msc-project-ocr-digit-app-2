# =========================================
# app.py (Robust OCR Version - Centered Detection)
# =========================================

import os
import uuid
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, request

# =========================================
# Flask Setup
# =========================================

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cpu")
model = None

# =========================================
# Model Architecture (Must Match training.py)
# =========================================


class PriyamDigitNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):

        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# =========================================
# Load Model
# =========================================


def load_model():
    global model
    if model is None:
        model = PriyamDigitNet().to(device)
        model.load_state_dict(
            torch.load(os.path.join(BASE_DIR, "digit_model.pth"),
                       map_location=device)
        )
        model.eval()

# =========================================
# ROBUST PREPROCESSING FUNCTION
# =========================================


def preprocess_and_center(image):
    """
    1. Convert to grayscale
    2. Detect digit region
    3. Crop bounding box
    4. Make square
    5. Resize to 32x32
    """

    # Convert PIL to numpy grayscale
    img = np.array(image.convert("L"))

    # Threshold to isolate digit (white background assumed)
    _, thresh = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY_INV)

    # Find non-zero pixels (digit region)
    coords = cv2.findNonZero(thresh)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

    # Make square canvas
    size = max(img.shape)
    square = np.ones((size, size), dtype=np.uint8) * 255

    h, w = img.shape
    square[(size-h)//2:(size-h)//2+h,
           (size-w)//2:(size-w)//2+w] = img

    # Resize to model input size
    square = cv2.resize(square, (32, 32))

    return Image.fromarray(square)

# =========================================
# Transform (Same as Training Normalization)
# =========================================


inference_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================================
# Prediction
# =========================================


def predict_digit(img_path):
    load_model()

    image = Image.open(img_path)

    # 🔥 Critical Step: Center Digit
    image = preprocess_and_center(image)

    tensor = inference_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)

    return pred.item(), round(conf.item() * 100, 2)

# =========================================
# Routes
# =========================================


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    confidence = None
    img_url = None
    filename = None

    if request.method == "POST" and "file" in request.files:

        file = request.files["file"]

        if file.filename != "":
            unique_name = str(uuid.uuid4()) + "_" + file.filename
            filepath = os.path.join(UPLOAD_FOLDER, unique_name)
            file.save(filepath)

            prediction, confidence = predict_digit(filepath)

            img_url = f"static/uploads/{unique_name}"
            filename = unique_name

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image=img_url,
        filename=filename
    )

# =========================================
# Run
# =========================================


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
