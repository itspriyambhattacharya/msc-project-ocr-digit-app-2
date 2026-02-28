import os
import torch
import torch.nn as nn
from flask import Flask, render_template, request
from torchvision import transforms
from PIL import Image, ImageOps

app = Flask(__name__)

# ==============================
# Paths & Device
# ==============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

device = torch.device("cpu")

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
# Safe Model Loading
# ==============================

model = None


def load_model():
    global model
    if model is None:
        model = PriyamDigitNet().to(device)
        state_dict = torch.load(
            os.path.join(BASE_DIR, "digit_model.pth"),
            map_location=device
        )
        model.load_state_dict(state_dict)
        model.eval()


# ==============================
# Safe Prediction Function
# ==============================

def predict_digit(img_path):
    try:
        load_model()

        img = Image.open(img_path).convert("L")

        if img.size[0] == 0 or img.size[1] == 0:
            return "Invalid Image", 0

        try:
            if img.getpixel((0, 0)) > 120:
                img = ImageOps.invert(img)
        except:
            pass

        bbox = img.getbbox()
        if bbox is None:
            return "No Digit", 0

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
            prob = torch.nn.functional.softmax(output, dim=1)
            conf, pred = torch.max(prob, 1)

        return pred.item(), round(conf.item() * 100, 2)

    except Exception as e:
        print("PREDICTION ERROR:", str(e))
        return "Error", 0


# ==============================
# Routes
# ==============================

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_url = None
    filename = None

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            filename = file.filename
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            digit, conf = predict_digit(filepath)

            prediction = digit
            confidence = conf
            img_url = f"static/uploads/{filename}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image=img_url,
        filename=filename
    )
