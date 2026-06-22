from __future__ import annotations

import os

import cv2
import numpy as np
import torch

from app.recognition.crnn import ALPHABET, CRNN, greedy_decode, normalize_plate_image


class PlateTextRecognizer:
    """Inference wrapper for the project-specific CRNN checkpoint."""

    def __init__(self, checkpoint_path: str, device: str = "auto"):
        self.checkpoint_path = checkpoint_path
        self.device = self._resolve_device(device)
        self.model: CRNN | None = None
        self.alphabet = ALPHABET
        self.image_height = 32
        self.image_width = 160
        self.load()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @property
    def available(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        if not os.path.exists(self.checkpoint_path):
            return
        try:
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=True)
            self.alphabet = checkpoint.get("alphabet", ALPHABET)
            self.image_height = int(checkpoint.get("image_height", 32))
            self.image_width = int(checkpoint.get("image_width", 160))
            self.model = CRNN(num_classes=len(self.alphabet) + 1)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device).eval()
            print(f"[INFO] Loaded custom OCR model: {self.checkpoint_path}")
        except Exception as exc:
            print(f"[WARNING] Could not load custom OCR model: {exc}")
            self.model = None

    def _recognize_lines(self, line_images: list[np.ndarray]) -> list[str]:
        if self.model is None:
            return []
        batch = torch.stack([
            normalize_plate_image(img, self.image_height, self.image_width)
            for img in line_images
        ]).to(self.device)
        with torch.inference_mode():
            logits = self.model(batch)
        return greedy_decode(logits, self.alphabet)

    @staticmethod
    def _split_two_lines(image: np.ndarray) -> list[np.ndarray]:
        h = image.shape[0]
        # A small overlap prevents horizontal strokes near the midpoint being cut.
        overlap = max(1, int(h * 0.04))
        middle = h // 2
        return [image[:middle + overlap], image[middle - overlap:]]

    def recognize(self, image: np.ndarray, is_motorbike: bool = False) -> str:
        if self.model is None or image is None or image.size == 0:
            return ""
        h, w = image.shape[:2]
        is_two_line = is_motorbike and (w / max(h, 1) < 2.0)
        line_images = self._split_two_lines(image) if is_two_line else [image]
        return "".join(self._recognize_lines(line_images))

