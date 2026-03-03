import os
import uuid
import torch
from flask import Flask, render_template, request
from PIL import Image
from utils import get_transforms
from training import PriyamDigitNet

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load Model Once at Startup
model = PriyamDigitNet().to(device)
if os.path.exists("digit_model.pth"):
    model.load_state_dict(torch.load("digit_model.pth", map_location=device))
model.eval()

# Shared Transform
inference_transform = get_transforms(train=False)


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    img_url = None
    filename = None

    if request.method == "POST" and "file" in request.files:
        file = request.files["file"]
        if file.filename != "":
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Inference
            img = Image.open(filepath).convert("L")
            tensor = inference_transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(tensor)
                prob = torch.softmax(output, dim=1)
                conf, pred = torch.max(prob, 1)

                prediction = pred.item()
                confidence = round(conf.item() * 100, 2)
                img_url = f"static/uploads/{filename}"

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        image=img_url,
        filename=filename
    )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
