import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import uuid
import shutil

# ================= FLASK APP =================

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
FEEDBACK_FOLDER = "data/live_feedback"
MODEL_PATH = "model.pth"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FEEDBACK_FOLDER, exist_ok=True)

# ================= MODEL =================


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


device = torch.device("cpu")
model = CNN()

# SAFE MODEL LOADING
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("Model loaded successfully.")
else:
    print("WARNING: model.pth not found. App will run but predictions disabled.")

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])

# ================= HOME =================


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    confidence = None
    image_path = None
    filename = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            image = Image.open(filepath)
            image = transform(image).unsqueeze(0)

            if os.path.exists(MODEL_PATH):
                with torch.no_grad():
                    output = model(image)
                    probs = torch.softmax(output, dim=1)
                    conf, pred = torch.max(probs, 1)

                prediction = pred.item()
                confidence = round(conf.item() * 100, 2)
            else:
                prediction = "Model Missing"
                confidence = 0

            image_path = "/" + filepath

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image=image_path,
        filename=filename
    )

# ================= FEEDBACK =================


@app.route("/feedback", methods=["POST"])
def feedback():

    filename = request.form.get("filename")
    feedback = request.form.get("feedback")
    correct_digit = request.form.get("correct_digit")

    if feedback == "no" and correct_digit:
        source_path = os.path.join(UPLOAD_FOLDER, filename)

        target_dir = os.path.join(FEEDBACK_FOLDER, correct_digit)
        os.makedirs(target_dir, exist_ok=True)

        new_name = str(uuid.uuid4()) + ".png"
        target_path = os.path.join(target_dir, new_name)

        shutil.copy(source_path, target_path)

    return redirect(url_for("home"))

# ================= RUN =================


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
