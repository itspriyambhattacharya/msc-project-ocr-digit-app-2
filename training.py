import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image

# ==============================
# Device
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# Robust Digit Preprocessor
# ==============================


class DigitPreprocessor:
    def __call__(self, img):

        # Convert to grayscale numpy array
        img = np.array(img.convert("L"))

        # Reduce noise
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Adaptive threshold (handles shadows & textures)
        img = cv2.adaptiveThreshold(
            img,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        # Morphological closing to remove small holes
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        # Crop bounding box
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y+h, x:x+w]

        # Make square
        size = max(img.shape) + 10
        square = np.zeros((size, size), dtype=np.uint8)

        h, w = img.shape
        square[(size-h)//2:(size-h)//2+h,
               (size-w)//2:(size-w)//2+w] = img

        return Image.fromarray(square)

# ==============================
# Transforms
# ==============================


train_tf = transforms.Compose([
    DigitPreprocessor(),
    transforms.Resize((32, 32)),
    transforms.RandomRotation(25),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.15, 0.15),
        scale=(0.8, 1.2),
        shear=10
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_tf = transforms.Compose([
    DigitPreprocessor(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# ==============================
# Model Architecture
# ==============================


class PriyamDigitNet(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

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
# Training Function
# ==============================


def train_model():

    # ------------------------------
    # Load Main Training Dataset
    # ------------------------------

    original_train_ds = datasets.ImageFolder(
        "data/custom/train",
        transform=train_tf
    )

    # ------------------------------
    # Load Feedback Dataset (if exists)
    # ------------------------------

    feedback_path = "data/live_feedback"

    if os.path.exists(feedback_path):

        feedback_train_ds = datasets.ImageFolder(
            feedback_path,
            transform=train_tf
        )

        if len(feedback_train_ds) > 0:
            print("Feedback samples found:", len(feedback_train_ds))
            train_ds = ConcatDataset(
                [original_train_ds, feedback_train_ds]
            )
        else:
            print("Feedback folder exists but empty.")
            train_ds = original_train_ds
    else:
        print("No feedback folder found.")
        train_ds = original_train_ds

    # ------------------------------
    # Validation Dataset
    # ------------------------------

    val_ds = datasets.ImageFolder(
        "data/custom/val",
        transform=val_tf
    )

    print("Total training samples:", len(train_ds))
    print("Total validation samples:", len(val_ds))

    # ------------------------------
    # Data Loaders
    # ------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=128,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=128,
        shuffle=False,
        num_workers=0
    )

    # ------------------------------
    # Model Setup
    # ------------------------------

    model = PriyamDigitNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.5)

    best_acc = 0

    # ------------------------------
    # Training Loop
    # ------------------------------

    for epoch in range(80):

        model.train()

        running_loss = 0

        for images, labels in train_loader:

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            running_loss += loss.item()

        scheduler.step()

        # ------------------------------
        # Validation
        # ------------------------------

        model.eval()
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:

                images, labels = images.to(device), labels.to(device)

                outputs = model(images)

                correct += (outputs.argmax(1) == labels).sum().item()

        acc = 100 * correct / len(val_ds)

        print(
            f"Epoch {epoch+1} | Loss: {running_loss:.4f} | Val Acc: {acc:.2f}%"
        )

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "digit_model.pth")

    print("Best Validation Accuracy:", round(best_acc, 2), "%")

# ==============================
# Entry Point
# ==============================


if __name__ == "__main__":
    train_model()
