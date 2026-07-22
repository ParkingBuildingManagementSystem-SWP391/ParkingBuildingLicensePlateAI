import os
import base64
import time

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
FAST_ACCEPT_CONFIDENCE = 0.82
SPLIT_FALLBACK_CONFIDENCE = 0.75


def resize_for_detection(image: np.ndarray) -> np.ndarray:
    max_side = settings.MAX_DETECT_IMAGE_SIDE
    if not max_side or max_side <= 0:
        return image

    height, width = image.shape[:2]
    longest_side = max(height, width)
    if longest_side <= max_side:
        return image

    scale = max_side / longest_side
    target_size = (int(round(width * scale)), int(round(height * scale)))
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


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

        if settings.WARMUP_ON_STARTUP:
            self.warmup()

    def warmup(self):
        """Run tiny startup predictions so the first real request is faster."""
        try:
            if self.yolo_model is not None:
                warmup_image = np.zeros((320, 320, 3), dtype=np.uint8)
                self.yolo_model.predict(warmup_image, verbose=False)
            if self.ocr_reader is not None:
                warmup_plate = np.full((80, 240), 255, dtype=np.uint8)
                cv2.putText(
                    warmup_plate,
                    "59A12345",
                    (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    0,
                    2,
                )
                self._read_easyocr(warmup_plate)
            print("[INFO] Model warmup completed")
        except Exception as exc:
            print(f"[WARNING] Model warmup skipped: {exc}")

    def _read_easyocr(self, image: np.ndarray):
        return self.ocr_reader.readtext(
            image,
            allowlist=EASYOCR_ALLOWLIST,
            decoder=settings.OCR_DECODER,
            beamWidth=settings.OCR_BEAM_WIDTH,
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

    def detect_and_recognize(self, image: np.ndarray, include_images: bool = True):
        """Detect the best plate and return text plus annotated/cropped images."""
        started_at = time.perf_counter()
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")
        if self.yolo_model is None:
            raise RuntimeError("YOLO model is not available")

        if self.ocr_reader is None:
            raise RuntimeError("EasyOCR is not available")

        original_height, original_width = image.shape[:2]
        image = resize_for_detection(image)
        detect_height, detect_width = image.shape[:2]

        yolo_started_at = time.perf_counter()
        results = self.yolo_model.predict(image, verbose=False)
        yolo_ms = (time.perf_counter() - yolo_started_at) * 1000
        print(
            f"[PERF] yolo_ms={yolo_ms:.0f} "
            f"original={original_width}x{original_height} "
            f"detect={detect_width}x{detect_height}"
        )
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

        box_ratio = box_width / max(box_height, 1)
        primary_mode = box_ratio < TWO_LINE_RATIO_THRESHOLD
        if settings.OCR_FAST_MODE and not settings.OCR_TRY_ALTERNATE_MODE:
            modes = [primary_mode]
        else:
            modes = [True, False] if primary_mode else [False, True]
        candidates = []
        candidate_images = {}
        fast_accept_variant = None
        ocr_total_ms = 0.0
        ocr_call_count = 0

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
            if settings.OCR_FAST_MODE:
                variant_images = [("binary", variants.binary)]
            else:
                variant_images = [
                    ("binary", variants.binary),
                    ("contrast", variants.contrasted),
                    ("adaptive", variants.adaptive),
                ]

            for variant_name, ocr_image in variant_images:
                ratio = ocr_image.shape[1] / max(ocr_image.shape[0], 1)

                ocr_started_at = time.perf_counter()
                ocr_results = self._read_easyocr(ocr_image)
                ocr_ms = (time.perf_counter() - ocr_started_at) * 1000
                ocr_total_ms += ocr_ms
                ocr_call_count += 1
                candidate_key = f"{mode_label}:{variant_name}:full"
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
                    mode_is_motorbike,
                ):
                    fast_accept_variant = candidate_key
                    break

                should_try_split = (
                    settings.OCR_TRY_SPLIT_FALLBACK
                    and
                    (mode_is_motorbike or ratio < TWO_LINE_RATIO_THRESHOLD)
                    and (
                        not settings.OCR_FAST_MODE
                        or confidence < SPLIT_FALLBACK_CONFIDENCE
                    )
                )
                if should_try_split:
                    ocr_started_at = time.perf_counter()
                    ocr_results = self._read_two_line_easyocr(ocr_image)
                    ocr_ms = (time.perf_counter() - ocr_started_at) * 1000
                    ocr_total_ms += ocr_ms
                    ocr_call_count += 2
                    candidate_key = f"{mode_label}:{variant_name}:split"
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
                        mode_is_motorbike,
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
        total_ms = (time.perf_counter() - started_at) * 1000
        print(
            f"[PERF] detect_total_ms={total_ms:.0f} "
            f"ocr_total_ms={ocr_total_ms:.0f} ocr_calls={ocr_call_count}"
        )

        crop_img, (x1, y1, x2, y2) = candidate_images.get(
            selected_variant,
            (
                image[original_y1:original_y2, original_x1:original_x2],
                (original_x1, original_y1, original_x2, original_y2),
            ),
        )

        if not include_images:
            return license_plate, None, None

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
