import cv2
import numpy as np
from dataclasses import dataclass


TARGET_CROP_WIDTH = 400
MAX_DESKEW_ANGLE = 20.0


@dataclass(frozen=True)
class PlateImageVariants:
    contrasted: np.ndarray
    binary: np.ndarray
    adaptive: np.ndarray


def deskew_plate(crop_img: np.ndarray) -> np.ndarray:
    """Correct a small in-plane rotation using long plate/character edges."""
    gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY) if crop_img.ndim == 3 else crop_img
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    min_line_length = max(20, int(gray.shape[1] * 0.25))
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=max(20, int(gray.shape[1] * 0.12)),
        minLineLength=min_line_length,
        maxLineGap=max(5, int(gray.shape[1] * 0.04)),
    )
    if lines is None:
        return crop_img.copy()

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = float(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
        if abs(angle) <= MAX_DESKEW_ANGLE:
            angles.append(angle)
    if not angles:
        return crop_img.copy()

    angle = float(np.median(angles))
    if abs(angle) < 0.25:
        return crop_img.copy()

    height, width = crop_img.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    return cv2.warpAffine(
        crop_img,
        matrix,
        (width, height),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


def preprocess_plate_variants(crop_img: np.ndarray) -> PlateImageVariants:
    """Create contrast and binary OCR inputs from one normalized plate crop."""
    if crop_img is None or crop_img.size == 0:
        raise ValueError("The detected license-plate crop is empty")

    deskewed = deskew_plate(crop_img)
    scale = TARGET_CROP_WIDTH / max(deskewed.shape[1], 1)
    interpolation = cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA
    target_height = max(1, int(round(deskewed.shape[0] * scale)))
    resized = cv2.resize(
        deskewed,
        (TARGET_CROP_WIDTH, target_height),
        interpolation=interpolation,
    )

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY) if resized.ndim == 3 else resized
    denoised = cv2.bilateralFilter(gray, d=5, sigmaColor=45, sigmaSpace=45)
    contrasted = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(denoised)
    binary = cv2.threshold(
        contrasted,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU,
    )[1]
    adaptive = cv2.adaptiveThreshold(
        contrasted,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        11,
    )

    # Keep the output convention stable: dark characters on a light background.
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)
    if np.mean(adaptive) < 127:
        adaptive = cv2.bitwise_not(adaptive)

    foreground = cv2.bitwise_not(binary)
    adaptive_foreground = cv2.bitwise_not(adaptive)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel, iterations=1)
    morphed = cv2.bitwise_not(cleaned)
    adaptive_closed = cv2.morphologyEx(adaptive_foreground, cv2.MORPH_CLOSE, kernel, iterations=1)
    adaptive_cleaned = cv2.morphologyEx(adaptive_closed, cv2.MORPH_OPEN, kernel, iterations=1)
    adaptive_morphed = cv2.bitwise_not(adaptive_cleaned)
    return PlateImageVariants(contrasted=contrasted, binary=morphed, adaptive=adaptive_morphed)


def preprocess_plate(crop_img: np.ndarray) -> np.ndarray:
    """Return the full Deskew -> ... -> Morphology pipeline output."""
    return preprocess_plate_variants(crop_img).binary
