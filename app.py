import os
import uuid
import shutil
import zipfile
import torch
import torch.nn as nn
from flask import (
    Flask,
    render_template,
    request,
    send_file,
    abort,
    redirect,
    send_from_directory,
    session
)
from torchvision import transforms
from PIL import Image, ImageOps

app = Flask(__name__)
app.secret_key = "change_this_to_a_long_random_secret_key"

# ==============================
# Paths & Configuration
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
FEEDBACK_FOLDER = os.path.join(BASE_DIR, "data", "live_feedback")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

for i in range(10):
    os.makedirs(os.path.join(FEEDBACK_FOLDER, str(i)), exist_ok=True)

device = torch.device("cpu")
model = None

# ==============================
# ADMIN CONFIGURATION
# ==============================

ADMINS = {
    "adminpriyam": {
        "name": "Priyam Bhattacharya",
        "password": "Priyam Bhattacharya",
        "image": "admins/priyam.webp"
    },
    "adminpritam": {
        "name": "Pritam Mondal",
        "password": "Pritam Mondal",
        "image": "admins/pritam.webp"
    },
    "adminsreena": {
        "name": "Sreena Mondal",
        "password": "Sreena Mondal",
        "image": "admins/sreena.webp"
    }
}

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
# Prediction
# ==============================


def predict_digit(img_path):
    load_model()

    img = Image.open(img_path).convert("L")

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
    os.makedirs(target_folder, exist_ok=True)

    target_path = os.path.join(target_folder, filename)

    if os.path.exists(source_path):
        shutil.copy(source_path, target_path)

    return render_template("thankyou.html")


@app.route("/feedback_images/<digit>/<filename>")
def serve_feedback_image(digit, filename):
    return send_from_directory(
        os.path.join(FEEDBACK_FOLDER, digit),
        filename
    )

# ==============================
# ADMIN AUTH SYSTEM
# ==============================


@app.route("/admin", methods=["GET", "POST"])
def admin_login():

    if request.method == "POST":
        user_id = request.form.get("user_id")
        password = request.form.get("password")

        if user_id in ADMINS and ADMINS[user_id]["password"] == password:
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


@app.route("/admin/download_image/<digit>/<filename>")
def download_image(digit, filename):

    if "admin" not in session:
        return redirect("/admin")

    return send_from_directory(
        os.path.join(FEEDBACK_FOLDER, digit),
        filename,
        as_attachment=True
    )


@app.route("/admin/delete_image", methods=["POST"])
def delete_image():

    if "admin" not in session:
        return redirect("/admin")

    digit = request.form.get("digit")
    filename = request.form.get("filename")

    file_path = os.path.join(FEEDBACK_FOLDER, digit, filename)

    if os.path.exists(file_path):
        os.remove(file_path)

    return redirect("/admin/dashboard")


@app.route("/admin/download_all")
def download_all():

    if "admin" not in session:
        return redirect("/admin")

    zip_path = os.path.join(BASE_DIR, "feedback_data.zip")

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(FEEDBACK_FOLDER):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, FEEDBACK_FOLDER)
                zipf.write(file_path, arcname)

    return send_file(zip_path, as_attachment=True)


@app.route("/admin/delete_all", methods=["POST"])
def delete_all():

    if "admin" not in session:
        return redirect("/admin")

    for digit in range(10):
        digit_folder = os.path.join(FEEDBACK_FOLDER, str(digit))
        if os.path.exists(digit_folder):
            for file in os.listdir(digit_folder):
                os.remove(os.path.join(digit_folder, file))

    return redirect("/admin/dashboard")


@app.route("/admin/logout")
def admin_logout():
    session.pop("admin", None)
    return redirect("/admin")

# ==============================
# RUN
# ==============================


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
