"""Evaluation preprocessing, mirrored exactly by web/src/preprocess.ts.

Plain bilinear resize (align_corners=False, no antialiasing) so the shorter
side is 256 px, keeping the aspect ratio; centre crop 224x224; ImageNet
mean/std normalisation; CHW float32. Keep both files in step:
web/test/parity.test.ts fails if they drift.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

RESIZE = 256
SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def resized_shape(height: int, width: int) -> tuple[int, int]:
    """Shape after scaling the shorter side to RESIZE (rounds half up, like JS Math.round)."""
    scale = RESIZE / min(height, width)
    return max(SIZE, math.floor(height * scale + 0.5)), max(SIZE, math.floor(width * scale + 0.5))


def pixels_to_tensor(rgb: np.ndarray) -> torch.Tensor:
    """HxWx3 uint8 RGB array -> 3x224x224 normalised float tensor."""
    x = torch.from_numpy(np.array(rgb, copy=True)).permute(2, 0, 1).float().unsqueeze(0) / 255.0
    h, w = resized_shape(x.shape[2], x.shape[3])
    x = F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False, antialias=False)
    top, left = (h - SIZE) // 2, (w - SIZE) // 2
    x = x[:, :, top:top + SIZE, left:left + SIZE]
    mean = torch.tensor(MEAN).view(1, 3, 1, 1)
    std = torch.tensor(STD).view(1, 3, 1, 1)
    return ((x - mean) / std).squeeze(0)


def load_eval(path) -> torch.Tensor:
    with Image.open(path) as img:
        return pixels_to_tensor(np.asarray(img.convert("RGB")))
