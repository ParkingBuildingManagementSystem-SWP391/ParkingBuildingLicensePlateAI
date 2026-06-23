import os
import base64
from typing import Optional

import cv2
import easyocr
import numpy as np
from ultralytics import YOLO

from app.core.config import settings
from app.utils.helpers import (
    is_fast_accept_ocr_candidate,
    ocr_result_confidence,
    preprocess_license_plate_text,
    select_best_ocr_candidate,
)
from app.utils.plate_image import preprocess_plate_variants


EASYOCR_ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
TWO_LINE_RATIO_THRESHOLD = 2.0
FAST_ACCEPT_CONFIDENCE = 0.88


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

    def _read_easyocr(self, image: np.ndarray):
        return self.ocr_reader.readtext(
            image,
            allowlist=EASYOCR_ALLOWLIST,
            decoder="beamsearch",
        )

    def _read_two_line_easyocr(self, image: np.ndarray):
        height = image.shape[0]
        if height < 2:
            return self._read_easyocr(image)

        split_y = height // 2
        top_results = self._read_easyocr(image[:split_y, :])
        bottom_results = self._read_easyocr(image[split_y:, :])

        adjusted_bottom = []
        for bbox, text, confidence in bottom_results:
            adjusted_bbox = [[point[0], point[1] + split_y] for point in bbox]
            adjusted_bottom.append((adjusted_bbox, text, confidence))

        return top_results + adjusted_bottom

    def detect_and_recognize(self, image: np.ndarray, is_motorbike: Optional[bool] = None):
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

        original_x1, original_y1, original_x2, original_y2 = map(int, best_box.xyxy[0].detach().cpu().tolist())
        box_width = original_x2 - original_x1
        box_height = original_y2 - original_y1

        modes = [is_motorbike] if is_motorbike is not None else [False, True]
        candidates = []
        candidate_images = {}
        fast_accept_variant = None

        for mode_is_motorbike in modes:
            mode_label = "motorbike" if mode_is_motorbike else "car"
            pad_x = int(box_width * 0.15)
            pad_y = int(box_height * (0.18 if mode_is_motorbike else 0.10))

            image_height, image_width = image.shape[:2]
            x1 = max(0, original_x1 - pad_x)
            y1 = max(0, original_y1 - pad_y)
            x2 = min(image_width, original_x2 + pad_x)
            y2 = min(image_height, original_y2 + pad_y)

            crop_img = image[y1:y2, x1:x2]
            variants = preprocess_plate_variants(crop_img)
            for variant_name, ocr_image in (
                ("binary", variants.binary),
                ("contrast", variants.contrasted),
                ("adaptive", variants.adaptive),
            ):
                ocr_attempts = [("full", self._read_easyocr(ocr_image))]
                ratio = ocr_image.shape[1] / max(ocr_image.shape[0], 1)
                if mode_is_motorbike or ratio < TWO_LINE_RATIO_THRESHOLD:
                    ocr_attempts.append(("split", self._read_two_line_easyocr(ocr_image)))

                for layout_name, ocr_results in ocr_attempts:
                    candidate_key = f"{mode_label}:{variant_name}:{layout_name}"
                    license_plate = preprocess_license_plate_text(
                        ocr_results,
                        mode_is_motorbike,
                        crop_shape=ocr_image.shape,
                    )
                    confidence = ocr_result_confidence(ocr_results)
                    candidates.append((candidate_key, license_plate, confidence))
                    candidate_images[candidate_key] = (crop_img, (x1, y1, x2, y2))
                    print(
                        f"[EASYOCR:{candidate_key}] Raw: {ocr_results} | "
                        f"plate: {license_plate} | confidence: {confidence:.3f}"
                    )
                    if is_fast_accept_ocr_candidate(
                        license_plate,
                        confidence,
                        FAST_ACCEPT_CONFIDENCE,
                    ):
                        fast_accept_variant = candidate_key
                        break
                if fast_accept_variant:
                    break
            if fast_accept_variant:
                break

        if fast_accept_variant:
            selected_variant, license_plate, selected_confidence = next(
                candidate for candidate in candidates if candidate[0] == fast_accept_variant
            )
        else:
            selected_variant, license_plate, selected_confidence = select_best_ocr_candidate(candidates)
        print(
            f"[EASYOCR] Selected: {selected_variant} | "
            f"plate: {license_plate} | confidence: {selected_confidence:.3f}"
        )

        crop_img, (x1, y1, x2, y2) = candidate_images.get(
            selected_variant,
            (
                image[original_y1:original_y2, original_x1:original_x2],
                (original_x1, original_y1, original_x2, original_y2),
            ),
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
