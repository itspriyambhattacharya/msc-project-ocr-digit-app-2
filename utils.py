import torch
from torchvision import transforms
from PIL import Image, ImageOps, ImageFilter
import numpy as np


class HandDrawnProcessor:
    def __call__(self, img):
        # 1. Grayscale & Contrast
        img = img.convert("L")
        img = ImageOps.autocontrast(img, cutoff=2)

        # 2. Binary Thresholding (Otsu-style logic)
        # Convert to numpy to find the optimal split between ink and paper
        np_img = np.array(img)
        # Assuming paper is light (>127) and ink is dark (<127)
        # We force pixels to be either 0 (black) or 255 (white)
        threshold = 110
        binary_mask = np.where(np_img < threshold, 0, 255).astype(np.uint8)

        # 3. Auto-Crop (Find the digit anywhere in the frame)
        # We look for black pixels (0) in the binary mask
        coords = np.argwhere(binary_mask == 0)

        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            # Add padding so the digit isn't touching the box edges
            h, w = y_max - y_min, x_max - x_min
            pad = int(max(h, w) * 0.15)

            # Crop from the binary mask to ensure the model sees ONLY black/white
            left = max(0, x_min - pad)
            top = max(0, y_min - pad)
            right = min(img.width, x_max + pad)
            bottom = min(img.height, y_max + pad)

            # Create a cropped binary image
            cropped_np = binary_mask[top:bottom, left:right]
            img = Image.fromarray(cropped_np)
        else:
            # If no digit found, return the binary version of the whole image
            img = Image.fromarray(binary_mask)

        return img


def get_transforms(train=False):
    t_list = [
        HandDrawnProcessor(),
        transforms.Resize((32, 32)),
    ]
    if train:
        t_list += [
            # Small variations to make the model "tough"
            transforms.RandomAffine(
                degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ]
    t_list += [
        transforms.ToTensor(),
        # For Binary images, 0.5 mean/std is standard
        transforms.Normalize((0.5,), (0.5,))
    ]
    return transforms.Compose(t_list)
