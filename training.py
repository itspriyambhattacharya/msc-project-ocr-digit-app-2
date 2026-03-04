import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from utils import get_transforms

device = torch.device("cpu")

# --- Architecture remains the same ---


class ResBlock(nn.Module):
    def __init__(self, in_f, out_f, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_f, out_f, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_f),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_f, out_f, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_f)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or in_f != out_f:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_f, out_f, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_f)
            )

    def forward(self, x):
        return nn.functional.relu(self.conv(x) + self.shortcut(x))


class PriyamDigitNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True)
        )
        self.layer1 = ResBlock(32, 32)
        self.layer2 = ResBlock(32, 64, stride=2)
        self.layer3 = ResBlock(64, 128, stride=2)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.layer3(self.layer2(self.layer1(self.pre(x)))))


def train_model():
    train_ds = datasets.ImageFolder(
        "data/custom/train", transform=get_transforms(train=True))
    val_ds = datasets.ImageFolder(
        "data/custom/val", transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=32,
                            num_workers=0, pin_memory=False)

    model = PriyamDigitNet().to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    # DYNAMIC CONFIG: Adjusts if you add those 200 images
    dataset_size = len(train_ds)
    epochs = 200 if dataset_size > 800 else 150
    patience = 25 if dataset_size > 800 else 15

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-3,
                                              steps_per_epoch=len(train_loader), epochs=epochs)

    best_acc, counter = 0, 0
    print(
        f"Training on {dataset_size} images. Patience: {patience} | Max Epochs: {epochs}")

    for epoch in range(epochs):
        model.train()
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(imgs), labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                correct += (model(imgs).argmax(1) == labels).sum().item()

        acc = 100 * correct / len(val_ds)

        if acc > best_acc:
            best_acc = acc
            counter = 0
            torch.save(model.state_dict(), "digit_model.pth")
            print(f"Epoch {epoch+1:03d} | New Best: {best_acc:.2f}%")
        else:
            counter += 1
            if (epoch + 1) % 10 == 0:
                print(
                    f"Epoch {epoch+1:03d} | Val Acc: {acc:.2f}% | Stall: {counter}/{patience}")

        if counter >= patience:
            print(f"Stopping early. Best accuracy achieved: {best_acc:.2f}%")
            break


if __name__ == "__main__":
    train_model()
