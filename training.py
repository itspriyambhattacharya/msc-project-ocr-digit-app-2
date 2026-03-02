# =========================================
# training.py (For Computer-Typed Digits)
# =========================================

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset

# ==============================
# Device
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# Transforms (Simplified)
# ==============================

train_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.RandomRotation(5),          # small rotation
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05)
    ),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
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

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)

        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.dropout = nn.Dropout(0.4)

        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):

        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # 32x32 → 16x16
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # 16x16 → 8x8
        x = self.pool(self.relu(self.bn3(self.conv3(x))))  # 8x8 → 4x4

        x = x.view(x.size(0), -1)

        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)

        return x

# ==============================
# Training Function
# ==============================


def train_model():

    original_train_ds = datasets.ImageFolder(
        "data/custom/train",
        transform=train_tf
    )

    feedback_path = "data/live_feedback"

    if os.path.exists(feedback_path):
        feedback_train_ds = datasets.ImageFolder(
            feedback_path,
            transform=train_tf
        )

        if len(feedback_train_ds) > 0:
            print("Feedback samples found:", len(feedback_train_ds))
            train_ds = ConcatDataset([original_train_ds, feedback_train_ds])
        else:
            train_ds = original_train_ds
    else:
        train_ds = original_train_ds

    val_ds = datasets.ImageFolder(
        "data/custom/val",
        transform=val_tf
    )

    print("Training samples:", len(train_ds))
    print("Validation samples:", len(val_ds))

    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False)

    model = PriyamDigitNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_acc = 0

    for epoch in range(40):

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

        # Validation
        model.eval()
        correct = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                correct += (outputs.argmax(1) == labels).sum().item()

        acc = 100 * correct / len(val_ds)

        print(
            f"Epoch {epoch+1} | Loss: {running_loss:.4f} | Val Acc: {acc:.2f}%")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "digit_model.pth")

    print("Best Validation Accuracy:", round(best_acc, 2), "%")


if __name__ == "__main__":
    train_model()
