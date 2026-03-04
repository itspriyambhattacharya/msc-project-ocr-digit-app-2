import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import DataLoader
from torchvision import datasets
from utils import get_transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_workers = os.cpu_count() if device.type == "cuda" else 0


class PriyamDigitNet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_f, out_f):
            return nn.Sequential(
                nn.Conv2d(in_f, out_f, 3, padding=1),
                nn.BatchNorm2d(out_f),
                nn.LeakyReLU(0.1),
                nn.MaxPool2d(2)
            )
        self.features = nn.Sequential(
            block(1, 32), block(32, 64),
            block(64, 128), block(128, 256)
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.4),  # Reduced dropout slightly for better convergence
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def train_model():
    train_ds = datasets.ImageFolder(
        "data/custom/train", transform=get_transforms(train=True))
    val_ds = datasets.ImageFolder(
        "data/custom/val", transform=get_transforms(train=False))

    train_loader = DataLoader(train_ds, batch_size=32,
                              shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=32, num_workers=num_workers)

    model = PriyamDigitNet().to(device)
    criterion = nn.CrossEntropyLoss()

    # Optimizer with Weight Decay to prevent erratic jumps
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    # Scheduler: Reduces LR when accuracy stops improving (Crucial for stability)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3)

    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_acc = 0
    epoches = 100
    print(f"Total Number of Epoches: {epoches}")
    for epoch in range(epoches):
        model.train()
        total_loss = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                loss = criterion(model(imgs), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        model.eval()
        correct = 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                correct += (outputs.argmax(1) == labels).sum().item()

        acc = 100 * correct / len(val_ds)
        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"Epoch {epoch+1:02d} | Val Acc: {acc:.2f}% | Loss: {total_loss/len(train_loader):.4f} | LR: {current_lr}")

        # Step the scheduler based on validation accuracy
        scheduler.step(acc)

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), "digit_model.pth")
            print("--> Model Saved")


if __name__ == "__main__":
    train_model()
