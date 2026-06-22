import base64
import os

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

from app.core.config import settings
from app.utils.helpers import preprocess_license_plate_text
from app.utils.plate_image import preprocess_plate


class LicensePlateDetector:
    def __init__(self):
        self.yolo_model = None
        self.ocr_reader = None
        self.load_models()

    def load_models(self):
        """Load YOLO and EasyOCR once when the service starts."""
        model_path = settings.YOLO_MODEL_PATH
        if not os.path.exists(model_path):
            print(f"[WARNING] Custom YOLO model not found: {model_path}")
            print(f"[INFO] Falling back to: {settings.FALLBACK_MODEL_PATH}")
            model_path = settings.FALLBACK_MODEL_PATH

        try:
            print(f"[INFO] Loading YOLO model from: {model_path}")
            self.yolo_model = YOLO(model_path)
            print("[INFO] YOLO model loaded successfully")
        except Exception as exc:
            print(f"[ERROR] Could not load YOLO model: {exc}")
            self.yolo_model = None

        try:
            print(f"[INFO] Loading EasyOCR (gpu={settings.OCR_GPU})")
            self.ocr_reader = easyocr.Reader(
                settings.OCR_LANGUAGES,
                gpu=settings.OCR_GPU,
            )
            print("[INFO] EasyOCR loaded successfully")
        except Exception as exc:
            print(f"[WARNING] Could not load EasyOCR: {exc}")
            self.ocr_reader = None

    def detect_and_recognize(self, image: np.ndarray, is_motorbike: bool = False):
        """Detect the best plate and return text plus annotated/cropped images."""
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if self.yolo_model is None:
            raise RuntimeError("YOLO model is not available")

        if self.ocr_reader is None:
            raise RuntimeError("EasyOCR is not available")

        results = self.yolo_model.predict(image, verbose=False)
        best_box = None
        best_confidence = 0.0

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                confidence = float(box.conf[0])
                if confidence > best_confidence:
                    best_confidence = confidence
                    best_box = box

        if best_box is None:
            return None, None, None

        x1, y1, x2, y2 = map(int, best_box.xyxy[0].detach().cpu().tolist())
        box_width = x2 - x1
        box_height = y2 - y1
        pad_x = int(box_width * 0.15)
        pad_y = int(box_height * (0.18 if is_motorbike else 0.10))

        image_height, image_width = image.shape[:2]
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(image_width, x2 + pad_x)
        y2 = min(image_height, y2 + pad_y)

        crop_img = image[y1:y2, x1:x2]
        final_crop = preprocess_plate(crop_img)

        ocr_results = self.ocr_reader.readtext(
            final_crop,
            allowlist="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ",
            decoder="beamsearch",
        )
        print(f"[EASYOCR] Raw result: {ocr_results}")
        license_plate = preprocess_license_plate_text(
            ocr_results,
            is_motorbike,
            crop_shape=final_crop.shape,
        )

        annotated_image = image.copy()
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 3)
        if license_plate:
            cv2.putText(
                annotated_image,
                license_plate,
                (x1, max(y1 - 10, 0)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (36, 255, 12),
                2,
            )

        annotated_ok, encoded_annotated = cv2.imencode(".jpg", annotated_image)
        crop_ok, encoded_crop = cv2.imencode(".jpg", crop_img)
        if not annotated_ok or not crop_ok:
            raise RuntimeError("Could not encode recognition result images")

        annotated_base64 = base64.b64encode(encoded_annotated).decode("utf-8")
        crop_base64 = base64.b64encode(encoded_crop).decode("utf-8")
        return license_plate, annotated_base64, crop_base64


detector_service = LicensePlateDetector()
