import os
import uuid
import torch
from flask import Flask, render_template, request
from PIL import Image
from utils import get_transforms
from training import PriyamDigitNet

app = Flask(__name__)
device = torch.device("cpu")

UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load custom model
model = PriyamDigitNet().to(device)
MODEL_PATH = "digit_model.pth"

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model loaded from {MODEL_PATH}")
else:
    print("Warning: digit_model.pth not found. Please run training.py first.")

model.eval()
inference_transform = get_transforms(train=False)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction, confidence, img_url = None, None, None

    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]
        if file and file.filename != "":
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            img = Image.open(filepath)
            tensor = inference_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                prob = torch.softmax(output, dim=1)
                conf, pred = torch.max(prob, 1)

                prediction = pred.item()
                confidence = round(conf.item() * 100, 2)
                img_url = f"static/uploads/{filename}"

    return render_template("index.html", prediction=prediction, confidence=confidence, image=img_url)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
