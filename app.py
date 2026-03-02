# =========================================
# Sudoku OCR (Proper Grid + Digit Extraction)
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
# Perspective Correction
# =========================================


def warp_sudoku(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) != 4:
        return None

    pts = approx.reshape(4, 2)

    rect = np.zeros((4, 2), dtype="float32")

    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    dst = np.array([
        [0, 0],
        [450, 0],
        [450, 450],
        [0, 450]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(gray, M, (450, 450))

    return warped

# =========================================
# Extract Digit From Cell
# =========================================


def extract_digit(cell):

    margin = 8
    h, w = cell.shape
    cell = cell[margin:h-margin, margin:w-margin]

    thresh = cv2.threshold(
        cell, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )[1]

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    if cv2.contourArea(largest) < 40:
        return None

    x, y, w, h = cv2.boundingRect(largest)
    digit = thresh[y:y+h, x:x+w]

    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)

    square[(size-h)//2:(size-h)//2+h,
           (size-w)//2:(size-w)//2+w] = digit

    return square

# =========================================
# Classify Digit
# =========================================


digit_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


def classify(digit_img):

    digit_img = cv2.resize(digit_img, (32, 32))
    pil = Image.fromarray(digit_img)
    tensor = digit_tf(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(tensor)
        pred = torch.argmax(output, 1).item()

    return pred

# =========================================
# Full Sudoku Processing
# =========================================


def process_sudoku(path):

    load_model()

    image = cv2.imread(path)

    warped = warp_sudoku(image)

    if warped is None:
        return None

    board = [["" for _ in range(9)] for _ in range(9)]

    cell_size = 450 // 9

    for i in range(9):
        for j in range(9):

            y1 = i * cell_size
            y2 = (i+1) * cell_size
            x1 = j * cell_size
            x2 = (j+1) * cell_size

            cell = warped[y1:y2, x1:x2]

            digit_img = extract_digit(cell)

            if digit_img is not None:
                board[i][j] = classify(digit_img)

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
