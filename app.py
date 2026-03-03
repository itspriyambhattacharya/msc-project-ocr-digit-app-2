import os
import uuid
import shutil
import zipfile
import cv2
import numpy as np
import torch
import torch.nn as nn
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    abort,
    redirect,
    send_from_directory
)

app = Flask(__name__)

# ==============================
# Paths & Configuration
# ==============================

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
# Model Architecture (MUST MATCH training.py)
# ==============================


class PriyamDigitNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.relu = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.dropout_conv = nn.Dropout2d(0.3)
        self.dropout_fc = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * 2 * 2, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))

        x = self.dropout_conv(x)
        x = x.view(x.size(0), -1)

        x = self.dropout_fc(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# ==============================
# Load Model
# ==============================


def load_model():
    global model
    if model is None:
        model = PriyamDigitNet().to(device)
        model.load_state_dict(
            torch.load(
                os.path.join(BASE_DIR, "digit_model.pth"),
                map_location=device
            )
        )
        model.eval()

# ==============================
# Robust Preprocessing
# ==============================


def preprocess_image(img_path):

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Blur to reduce noise
    img = cv2.GaussianBlur(img, (5, 5), 0)

    # Adaptive threshold (handles shadows & texture)
    img = cv2.adaptiveThreshold(
        img,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11,
        2
    )

    # Morphological closing to remove small gaps
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    coords = cv2.findNonZero(img)

    if coords is not None:
        x, y, w, h = cv2.boundingRect(coords)
        img = img[y:y+h, x:x+w]

    size = max(img.shape) + 10
    square = np.zeros((size, size), dtype=np.uint8)

    h, w = img.shape
    square[(size-h)//2:(size-h)//2+h,
           (size-w)//2:(size-w)//2+w] = img

    square = cv2.resize(square, (32, 32))

    square = square.astype("float32") / 255.0
    square = (square - 0.5) / 0.5

    tensor = torch.tensor(square).unsqueeze(0).unsqueeze(0)
    return tensor.to(device)

# ==============================
# Prediction
# ==============================


def predict_digit(img_path):
    load_model()

    tensor = preprocess_image(img_path)

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
# Feedback
# ==============================


@app.route("/feedback", methods=["POST"])
def feedback():

    filename = request.form.get("filename")
    predicted_digit = request.form.get("predicted")
    feedback_choice = request.form.get("feedback")
    correct_digit = request.form.get("correct_digit")

    if not filename or not feedback_choice:
        return render_template("thankyou.html")

    source_path = os.path.join(UPLOAD_FOLDER, filename)

    if feedback_choice == "yes":
        target_digit = predicted_digit
    elif feedback_choice == "no" and correct_digit and correct_digit.isdigit():
        target_digit = correct_digit
    else:
        return render_template("thankyou.html")

    target_folder = os.path.join(FEEDBACK_FOLDER, target_digit)
    os.makedirs(target_folder, exist_ok=True)

    if os.path.exists(source_path):
        shutil.copy(source_path, os.path.join(target_folder, filename))

    return render_template("thankyou.html")

# ==============================
# Admin Download
# ==============================


@app.route("/admin/download_feedback")
def download_feedback():

    key = request.args.get("key")
    if key != ADMIN_SECRET:
        abort(403)

    zip_path = os.path.join(BASE_DIR, "feedback_data.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(FEEDBACK_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, FEEDBACK_FOLDER)
                zipf.write(file_path, arcname)

    return send_file(zip_path, as_attachment=True)

# ==============================
# Run
# ==============================


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
