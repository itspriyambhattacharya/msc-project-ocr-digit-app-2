import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


class DigitPreprocessor:
    def __call__(self, img):
        # Convert PIL to Grayscale Numpy
        img = np.array(img.convert("L"))

        # Noise reduction and Adaptive Thresholding
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )

        # Crop to bounding box of the digit
        coords = cv2.findNonZero(img)
        if coords is not None:
            x, y, w, h = cv2.boundingRect(coords)
            img = img[y:y+h, x:x+w]

        # Pad to square to prevent stretching
        size = max(img.shape) + 10
        square = np.zeros((size, size), dtype=np.uint8)
        h, w = img.shape
        square[(size-h)//2:(size-h)//2+h, (size-w)//2:(size-w)//2+w] = img
        return Image.fromarray(square)


def get_transforms(train=False):
    t_list = [DigitPreprocessor(), transforms.Resize((32, 32))]
    if train:
        t_list += [
            transforms.RandomRotation(20),
            transforms.RandomAffine(
                0, translate=(0.15, 0.15), scale=(0.8, 1.2))
        ]
    t_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
    return transforms.Compose(t_list)
