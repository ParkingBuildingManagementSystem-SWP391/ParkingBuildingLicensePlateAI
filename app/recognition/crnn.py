from __future__ import annotations

import cv2
import numpy as np
import torch
from torch import nn


# Keep the recognizer permissive. Vietnamese plate grammar belongs to validation,
# not to the neural network's character vocabulary.
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK_INDEX = 0


class CRNN(nn.Module):
    """Compact convolutional/recurrent recognizer trained with CTC loss."""

    def __init__(self, num_classes: int = len(ALPHABET) + 1, hidden_size: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),
        )
        self.sequence = nn.Sequential(
            nn.LSTM(512, hidden_size, bidirectional=True, batch_first=True),
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(images)
        features = features.mean(dim=2).permute(0, 2, 1)  # B, time, channels
        sequence, _ = self.sequence(features)
        return self.classifier(sequence)  # B, time, classes


def normalize_plate_image(
    image: np.ndarray,
    image_height: int = 32,
    image_width: int = 160,
) -> torch.Tensor:
    """Resize with preserved aspect ratio and right-pad to a fixed canvas."""
    if image is None or image.size == 0:
        raise ValueError("Empty license-plate image")

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]
    scale = min(image_width / max(w, 1), image_height / max(h, 1))
    new_w = max(1, min(image_width, int(round(w * scale))))
    new_h = max(1, min(image_height, int(round(h * scale))))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    resized = cv2.resize(gray, (new_w, new_h), interpolation=interpolation)

    canvas = np.full((image_height, image_width), 255, dtype=np.uint8)
    y = (image_height - new_h) // 2
    canvas[y:y + new_h, :new_w] = resized
    tensor = torch.from_numpy(canvas).float().div_(127.5).sub_(1.0)
    return tensor.unsqueeze(0)


def greedy_decode(logits: torch.Tensor, alphabet: str = ALPHABET) -> list[str]:
    """Collapse CTC predictions and remove blank tokens."""
    token_batches = logits.argmax(dim=-1).detach().cpu().tolist()
    decoded: list[str] = []
    for tokens in token_batches:
        chars: list[str] = []
        previous = None
        for token in tokens:
            if token != BLANK_INDEX and token != previous:
                chars.append(alphabet[token - 1])
            previous = token
        decoded.append("".join(chars))
    return decoded

