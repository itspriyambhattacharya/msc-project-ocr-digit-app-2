import os
import uuid
import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from flask import Flask, render_template, request

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
device = torch.device("cpu")
model = None


class PriyamDigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(
                32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(
                64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(
                128), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 9)  # Matches training
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def load_model():
    global model
    if model is None:
        model = PriyamDigitNet().to(device)
        model.load_state_dict(torch.load(
            "digit_model.pth", map_location=device))
        model.eval()


def warp_sudoku(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    thresh = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    approx = cv2.approxPolyDP(
        largest, 0.02 * cv2.arcLength(largest, True), True)
    if len(approx) != 4:
        return None
    pts = approx.reshape(4, 2)
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0], rect[2] = pts[np.argmin(s)], pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1], rect[3] = pts[np.argmin(diff)], pts[np.argmax(diff)]
    dst = np.array([[0, 0], [450, 0], [450, 450], [0, 450]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return cv2.warpPerspective(gray, M, (450, 450))


def remove_grid_lines(img):
    thresh = cv2.adaptiveThreshold(
        img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
    grid = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, h_kern) + \
        cv2.morphologyEx(thresh, cv2.MORPH_OPEN, v_kern)
    return cv2.subtract(thresh, grid)


def extract_digit(cell):
    cell = cell[10:-10, 10:-10]
    contours, _ = cv2.findContours(
        cell, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 80:
        return None
    x, y, w, h = cv2.boundingRect(largest)
    digit = cell[y:y+h, x:x+w]
    size = max(w, h)
    square = np.zeros((size, size), dtype=np.uint8)
    square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = digit
    return square


digit_tf = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])


def classify(digit_img):
    digit_img = cv2.resize(digit_img, (32, 32))
    tensor = digit_tf(Image.fromarray(digit_img)).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
        conf, pred = torch.max(torch.softmax(output, dim=1), 1)
    if conf.item() < 0.75:
        return ""
    return pred.item() + 1  # Corrected: Mapping index 0-8 to digit 1-9


def process_sudoku(path):
    load_model()
    image = cv2.imread(path)
    warped = warp_sudoku(image)
    if warped is None:
        return None
    cleaned = remove_grid_lines(warped)
    board = [["" for _ in range(9)] for _ in range(9)]
    for i in range(9):
        for j in range(9):
            cell = cleaned[i*50:(i+1)*50, j*50:(j+1)*50]
            digit_img = extract_digit(cell)
            if digit_img is not None:
                board[i][j] = classify(digit_img)
    return board


@app.route("/", methods=["GET", "POST"])
def index():
    board = None
    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            path = os.path.join(UPLOAD_FOLDER, str(
                uuid.uuid4()) + "_" + file.filename)
            file.save(path)
            board = process_sudoku(path)
    return render_template("sudoku.html", board=board)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
