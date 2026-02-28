import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image, ImageOps
import os

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- CUSTOM PREPROCESSOR ---


class DigitPreprocessor:
    def __call__(self, img):
        img = ImageOps.grayscale(img)
        # Ensure black background/white digit (Standard for OCR)
        if img.getpixel((0, 0)) > 120:
            img = ImageOps.invert(img)
        bbox = img.getbbox()
        if bbox:
            img = img.crop(bbox)
        w, h = img.size
        m = max(w, h) + 10
        new_img = Image.new("L", (m, m), 0)
        new_img.paste(img, ((m-w)//2, (m-h)//2))
        return new_img


# --- DATA AUGMENTATION (Increases accuracy to 90%+) ---
train_tf = transforms.Compose([
    DigitPreprocessor(),
    transforms.Resize((32, 32)),
    transforms.RandomRotation(15),
    transforms.RandomAffine(0, translate=(0.1, 0.1), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

val_tf = transforms.Compose([
    DigitPreprocessor(),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load original training dataset
original_train_ds = datasets.ImageFolder(
    "data/custom/train", transform=train_tf)

# Load feedback dataset (if exists and not empty)
feedback_path = "data/live_feedback"

if os.path.exists(feedback_path) and len(os.listdir(feedback_path)) > 0:
    feedback_train_ds = datasets.ImageFolder(feedback_path, transform=train_tf)

    if len(feedback_train_ds) > 0:
        print(f"Feedback samples found: {len(feedback_train_ds)}")
        train_ds = ConcatDataset([original_train_ds, feedback_train_ds])
    else:
        print("Feedback folder exists but contains no images.")
        train_ds = original_train_ds
else:
    print("No feedback dataset found.")
    train_ds = original_train_ds

# Validation dataset remains same
val_ds = datasets.ImageFolder("data/custom/val", transform=val_tf)

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=32)

print(f"Total training samples: {len(train_ds)}")

# --- YOUR CUSTOM ARCHITECTURE (No Pre-existing Models) ---


class PriyamDigitNet(nn.Module):
    def __init__(self, num_classes=10):
        super(PriyamDigitNet, self).__init__()

        # Block 1: 1 -> 32 channels
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        # Block 2: 32 -> 64 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        # Block 3: 64 -> 128 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.4)

        # Fully Connected Layers
        # 32x32 -> 16x16 -> 8x8 -> 4x4 (After 3 poolings)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))

        x = x.view(-1, 128 * 4 * 4)  # Flatten
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


model = PriyamDigitNet().to(device)

# --- TRAINING CONFIGURATION ---
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

best_acc = 0
for epoch in range(50):
    model.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(images), labels)
        loss.backward()
        optimizer.step()

    # Validation Phase
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()

    acc = 100 * correct / len(val_ds)
    print(f"Epoch {epoch+1} | Val Acc: {acc:.2f}%")
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), "digit_model.pth")

print(f"Project Best Accuracy: {best_acc}%")
