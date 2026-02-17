# =========================================
# Flask OCR Digit Prediction App
# =========================================

from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import os

# -----------------------------
# 1. APP INITIALIZATION
# -----------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# -----------------------------
# 2. DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# 3. TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# -----------------------------
# 4. MODEL ARCHITECTURE
# -----------------------------


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 16, 3)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(16, 32, 3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu3(self.fc1(x))
        x = self.fc2(x)
        return x


# -----------------------------
# 5. LOAD TRAINED MODEL
# -----------------------------
model = CustomCNN().to(device)
model.load_state_dict(torch.load("digit_model.pth", map_location=device))
model.eval()
print("Model loaded successfully!")

# -----------------------------
# 6. PREDICTION FUNCTION
# -----------------------------


def predict_digit(image_path):

    image = Image.open(image_path)
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)

    return predicted.item()

# -----------------------------
# 7. ROUTES
# -----------------------------


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    image_path = None

    if request.method == "POST":

        file = request.files["file"]

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)

            prediction = predict_digit(filepath)
            image_path = filepath

    return render_template("index.html",
                           prediction=prediction,
                           image_path=image_path)


# -----------------------------
# 8. RUN APP
# -----------------------------
if __name__ == "__main__":
    app.run(debug=True)
