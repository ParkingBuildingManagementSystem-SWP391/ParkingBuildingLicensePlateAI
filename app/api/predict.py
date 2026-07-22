import os
import time
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile
from pydantic import BaseModel
from typing import Optional
from app.core.config import settings
from app.services.detector import detector_service

router = APIRouter()

# Schema đầu ra cho Web UI
class PredictWebResponse(BaseModel):
    status: str
    license_plate: Optional[str] = None
    annotated_image: Optional[str] = None
    crop_image: Optional[str] = None
    message: Optional[str] = None


async def decode_uploaded_image(file: UploadFile) -> tuple[np.ndarray, int]:
    """Read one uploaded image and decode it into OpenCV BGR format."""
    contents = await file.read()
    image_bytes = np.frombuffer(contents, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Ảnh tải lên không hợp lệ")
    return image, len(contents)


@router.post("/predict-file-fast")
async def predict_file_fast(file: UploadFile = File(...)):
    """
    API nhẹ cho Backend .NET: upload file ảnh trực tiếp và chỉ trả biển số,
    không trả ảnh base64 để tránh response nặng và timeout.
    """
    try:
        request_started_at = time.perf_counter()
        image, upload_bytes = await decode_uploaded_image(file)

        license_plate, _, _ = detector_service.detect_and_recognize(
            image,
            include_images=False
        )
        total_ms = (time.perf_counter() - request_started_at) * 1000
        print(
            f"[PERF] endpoint=/predict-file-fast total_ms={total_ms:.0f} "
            f"upload_bytes={upload_bytes}"
        )

        if not license_plate:
            return {"status": "error", "message": "Không phát hiện thấy biển số xe"}

        return {
            "status": "success",
            "license_plate": license_plate
        }
    except Exception as e:
        import traceback
        log_path = os.path.join(settings.BASE_DIR, "error_traceback.log")
        with open(log_path, "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        return {"status": "error", "message": str(e)}


@router.post("/predict-file", response_model=PredictWebResponse)
async def predict_file(file: UploadFile = File(...)):
    """
    API hỗ trợ giao diện Web (upload trực tiếp file ảnh để test nhanh).
    """
    try:
        image, _ = await decode_uploaded_image(file)

        license_plate, annotated_b64, crop_b64 = detector_service.detect_and_recognize(image)
        
        if not license_plate:
            return {"status": "error", "message": "Không phát hiện thấy biển số xe"}
            
        return {
            "status": "success",
            "license_plate": license_plate,
            "annotated_image": annotated_b64,
            "crop_image": crop_b64
        }
    except Exception as e:
        import traceback
        log_path = os.path.join(settings.BASE_DIR, "error_traceback.log")
        with open(log_path, "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        return {"status": "error", "message": str(e)}
