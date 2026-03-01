import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, ConcatDataset
from PIL import Image, ImageOps

# ==============================
# Device Configuration
# ==============================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ==============================
# Custom Preprocessor
# ==============================


class DigitPreprocessor:
    def __call__(self, img):
        img = ImageOps.grayscale(img)

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


# ==============================
# Data Augmentation
# ==============================

train_tf = transforms.Compose([
    DigitPreprocessor(),
    transforms.Resize((32, 32)),
    transforms.RandomRotation(20),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.15, 0.15),
        scale=(0.8, 1.2),
        shear=8
    ),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
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
# Custom CNN
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
# Main Training Function
# ==============================

def train_model():

    # Dataset loading
    original_train_ds = datasets.ImageFolder(
        "data/custom/train", transform=train_tf)

    feedback_path = "data/live_feedback"

    if os.path.exists(feedback_path):
        feedback_train_ds = datasets.ImageFolder(
            feedback_path, transform=train_tf)
        if len(feedback_train_ds) > 0:
            train_ds = ConcatDataset([original_train_ds, feedback_train_ds])
            print("Feedback samples found:", len(feedback_train_ds))
        else:
            train_ds = original_train_ds
    else:
        train_ds = original_train_ds

    val_ds = datasets.ImageFolder("data/custom/val", transform=val_tf)

    print("Total training samples:", len(train_ds))
    print("Total validation samples:", len(val_ds))

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        num_workers=0,      # Windows safe
        pin_memory=False    # CPU safe
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        num_workers=0,
        pin_memory=False
    )

    model = PriyamDigitNet().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=15,
        gamma=0.5
    )

    best_acc = 0

    for epoch in range(50):

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


# ==============================
# Windows Safe Entry Point
# ==============================

if __name__ == "__main__":
    train_model()
