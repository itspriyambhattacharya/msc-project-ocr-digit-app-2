import os
import uuid
import shutil
import torch
import torch.nn as nn
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    send_from_directory,
    session
)
from torchvision import transforms
from PIL import Image, ImageOps
from werkzeug.security import generate_password_hash, check_password_hash

# ==============================
# Flask Setup
# ==============================

app = Flask(__name__)
app.secret_key = "replace_this_with_a_long_random_secret_key"

# ==============================
# Paths
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
FEEDBACK_FOLDER = os.path.join(BASE_DIR, "data", "live_feedback")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

for i in range(10):
    os.makedirs(os.path.join(FEEDBACK_FOLDER, str(i)), exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None

# ==============================
# ADMIN CONFIGURATION
# ==============================

ADMINS = {
    "adminpriyam": {
        "name": "Priyam Bhattacharya",
        "password": generate_password_hash("Priyam Bhattacharya"),
        "image": "https://1drv.ms/u/c/71f0b9c95c099bd5/IQBNGckAppDZSZNNJiGBvAknAaLUq8TSUjOfwVlxmhCBASI"
    },
    "adminpritam": {
        "name": "Pritam Mondal",
        "password": generate_password_hash("Pritam Mondal"),
        "image": "https://1drv.ms/u/c/71f0b9c95c099bd5/IQCCEeOWQsSVTqDPvMn6_q0OAYcb-NoO6J-6fTFtRtXa_2c"
    },
    "adminsreena": {
        "name": "Sreena Mondal",
        "password": generate_password_hash("Sreena Mondal"),
        "image": "https://1drv.ms/u/c/71f0b9c95c099bd5/IQB5KWorh4ccRK2iblQQv2O7AWvK_JMsjsnbcECkJkjBqE8"
    }
}

# ==============================
# Model Architecture (IDENTICAL TO training.py)
# ==============================


class PriyamDigitNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PriyamDigitNet, self).__init__()

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
# Prediction
# ==============================


def predict_digit(img_path):
    load_model()

    img = Image.open(img_path).convert("L")

    # Same preprocessing as training
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
# MAIN ROUTES
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
    target_path = os.path.join(target_folder, filename)

    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)

    return render_template("thankyou.html")


# ==============================
# ADMIN LOGIN
# ==============================


@app.route("/admin", methods=["GET", "POST"])
def admin_login():

    if request.method == "POST":
        user_id = request.form.get("user_id")
        password = request.form.get("password")

        if user_id in ADMINS and check_password_hash(
                ADMINS[user_id]["password"], password):
            session["admin"] = user_id
            return redirect("/admin/dashboard")

        return render_template("admin_login.html", error="Invalid credentials")

    return render_template("admin_login.html")


@app.route("/admin/dashboard")
def admin_dashboard():

    if "admin" not in session:
        return redirect("/admin")

    admin_id = session["admin"]
    admin_info = ADMINS[admin_id]

    feedback_data = {}

    for digit in range(10):
        digit_folder = os.path.join(FEEDBACK_FOLDER, str(digit))
        images = []

        if os.path.exists(digit_folder):
            for file in os.listdir(digit_folder):
                if os.path.isfile(os.path.join(digit_folder, file)):
                    images.append(file)

        feedback_data[str(digit)] = images

    return render_template(
        "admin_dashboard.html",
        admin=admin_info,
        feedback_data=feedback_data
    )


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect("/admin")


# ==============================
# RUN
# ==============================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
