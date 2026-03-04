import torch
from torchvision import transforms
from PIL import Image, ImageOps


class DigitStandardizer:
    """Standardizes input so the digit is light and background is dark."""

    def __call__(self, img):
        img = img.convert("L")
        # If the image is mostly white (paper), invert it
        if sum(img.getdata()) / len(img.getdata()) > 127:
            img = ImageOps.invert(img)
        return img


def get_transforms(train=False):
    t_list = [
        DigitStandardizer(),
        transforms.Resize((32, 32)),
    ]

    if train:
        t_list += [
            transforms.RandomAffine(
                degrees=15, translate=(0.1, 0.1), scale=(0.8, 1.2)),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        ]

    t_list += [
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ]
    return transforms.Compose(t_list)
