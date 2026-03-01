import os
import uuid
import shutil
import zipfile
import torch
import torch.nn as nn
from flask import Flask, render_template, request, send_file, abort
from torchvision import transforms
from PIL import Image, ImageOps

app = Flask(__name__)

# ==============================
# Paths & Configuration
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADMIN_SECRET = "password"   # Change this to something strong and private

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
FEEDBACK_FOLDER = os.path.join(BASE_DIR, "data", "live_feedback")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Create feedback folders 0-9 at startup
for i in range(10):
    os.makedirs(os.path.join(FEEDBACK_FOLDER, str(i)), exist_ok=True)

device = torch.device("cpu")
model = None


# ==============================
# Model Architecture
# ==============================

class PriyamDigitNet(nn.Module):
    def __init__(self):
        super(PriyamDigitNet, self).__init__()

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(-1, 128 * 4 * 4)
        x = self.dropout(self.relu(self.fc1(x)))
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
# Prediction Function
# ==============================

def predict_digit(img_path):
    load_model()

    img = Image.open(img_path).convert("L")

    # Match training preprocessing
    if img.getpixel((0, 0)) > 120:
        img = ImageOps.invert(img)

    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)

    w, h = img.size
    m = max(w, h) + 10

    new_img = Image.new("L", (m, m), 0)
    new_img.paste(img, ((m - w) // 2, (m - h) // 2))

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    tensor = transform(new_img).unsqueeze(0).to(device)

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
            original_name = file.filename
            unique_name = str(uuid.uuid4()) + "_" + original_name
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

    target_path = os.path.join(target_folder, filename)

    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)

    return render_template("thankyou.html")


# ==============================
# Admin Download Route (Protected)
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
# Admin Clear Feedback Route
# ==============================

@app.route("/admin/clear_feedback")
def clear_feedback():

    key = request.args.get("key")

    if key != ADMIN_SECRET:
        abort(403)

    # Delete all files inside each digit folder
    for digit in range(10):
        digit_folder = os.path.join(FEEDBACK_FOLDER, str(digit))

        if os.path.exists(digit_folder):
            for file in os.listdir(digit_folder):
                file_path = os.path.join(digit_folder, file)

                if os.path.isfile(file_path):
                    os.remove(file_path)

    return render_template("admin_success.html")

# ==============================
# Run
# ==============================


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
