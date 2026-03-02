# =========================================
# app.py (For Computer-Typed Digits)
# =========================================

import os
import uuid
import shutil
import zipfile
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    abort
)

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADMIN_SECRET = "password"

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
FEEDBACK_FOLDER = os.path.join(BASE_DIR, "data", "live_feedback")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
for i in range(10):
    os.makedirs(os.path.join(FEEDBACK_FOLDER, str(i)), exist_ok=True)

device = torch.device("cpu")
model = None

# ==============================
# Same Model Architecture
# ==============================


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

# ==============================
# Transform (Must Match Training)
# ==============================


inference_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==============================
# Load Model
# ==============================


def load_model():
    global model
    if model is None:
        model = PriyamDigitNet().to(device)
        model.load_state_dict(
            torch.load(os.path.join(BASE_DIR, "digit_model.pth"),
                       map_location=device)
        )
        model.eval()

# ==============================
# Prediction
# ==============================


def predict_digit(img_path):
    load_model()

    image = Image.open(img_path)
    tensor = inference_tf(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        prob = torch.softmax(output, dim=1)
        conf, pred = torch.max(prob, 1)

    return pred.item(), round(conf.item() * 100, 2)

# ==============================
# Routes
# ==============================


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

# ==============================
# Run
# ==============================


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
