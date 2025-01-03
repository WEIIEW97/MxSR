import torch
import cv2
from PIL import Image
import numpy as np


def get_torch_device() -> torch.device:
    """Get the Torch device."""
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    return device


def imread_cv2(path):
    im = cv2.imread(path, cv2.IMREAD_ANYCOLOR)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    return im


def imread_pil(path):
    return np.array(Image.open(path).convert("RGB"))
