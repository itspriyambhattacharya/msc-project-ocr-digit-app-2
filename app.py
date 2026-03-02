# =========================================
# app.py (Sudoku OCR Phase 1)
# =========================================

import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
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
# Model Architecture (Same as training.py)
# =========================================


class PriyamDigitNet(nn.Module):
    def __init__(self):
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
# Digit Transform
# =========================================


digit_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# =========================================
# Digit Classification
# =========================================


def classify_digit(cell_img):

    cell_img = cv2.resize(cell_img, (32, 32))
    pil_img = Image.fromarray(cell_img)
    tensor = digit_tf(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, 1).item()

    return pred

# =========================================
# Sudoku Processing
# =========================================


def process_sudoku(image_path):

    load_model()

    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    largest = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(largest)
    grid = gray[y:y+h, x:x+w]

    grid = cv2.resize(grid, (450, 450))

    cell_size = 450 // 9

    board = [[0 for _ in range(9)] for _ in range(9)]

    for i in range(9):
        for j in range(9):

            y1 = i * cell_size
            y2 = (i+1) * cell_size
            x1 = j * cell_size
            x2 = (j+1) * cell_size

            cell = grid[y1:y2, x1:x2]

            _, cell_thresh = cv2.threshold(
                cell, 200, 255, cv2.THRESH_BINARY_INV
            )

            if cv2.countNonZero(cell_thresh) > 50:
                board[i][j] = classify_digit(cell)
            else:
                board[i][j] = ""

    return board

# =========================================
# Routes
# =========================================


@app.route("/", methods=["GET", "POST"])
def index():

    board = None

    if request.method == "POST":

        file = request.files["file"]

        if file.filename != "":
            filename = str(uuid.uuid4()) + "_" + file.filename
            path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(path)

            board = process_sudoku(path)

    return render_template("sudoku.html", board=board)

# =========================================
# Run
# =========================================


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
