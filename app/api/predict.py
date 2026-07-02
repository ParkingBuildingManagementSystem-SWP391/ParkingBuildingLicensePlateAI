import os
import cv2
import numpy as np
import requests
from fastapi import APIRouter, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
from app.core.config import settings
from app.services.detector import detector_service

router = APIRouter()

# Schema đầu vào cho API .NET
class PredictRequest(BaseModel):
    image_url: str

# Schema đầu ra cho Web UI
class PredictWebResponse(BaseModel):
    status: str
    license_plate: Optional[str] = None
    annotated_image: Optional[str] = None
    crop_image: Optional[str] = None
    message: Optional[str] = None


def get_image_from_url(url: str) -> np.ndarray:
    headers = {
        # Giả lập trình duyệt để tránh bị chặn và yêu cầu định dạng tương thích cao
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "image/jpeg,image/png,image/*;q=0.8"
    }

    # Thiết lập timeout hợp lý (ví dụ: 10 giây) để tránh treo API khi Cloudinary gặp sự cố
    response = requests.get(url, headers=headers, timeout=10)

    if response.status_code != 200:
        raise Exception(f"Không thể tải ảnh từ URL Cloudinary. HTTP Code: {response.status_code}")

    image_bytes = np.frombuffer(response.content, dtype=np.uint8)
    image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    if image is None:
        raise Exception("Không thể giải mã định dạng ảnh. Hãy chắc chắn URL trả về định dạng JPG/PNG.")

    return image


@router.post("/predict")
async def predict(payload: PredictRequest):
    """
    Endpoint API chính tiếp nhận URL ảnh lưu trên Cloudinary từ Backend .NET gửi sang.
    """
    try:
        image_url = payload.image_url
        if not image_url:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "message": "Tham số 'image_url' không được để trống."}
            )

        print(f"[API /predict] Nhận yêu cầu từ URL: {image_url}")

        # Tải và giải mã ảnh bằng hàm helper dùng chung
        try:
            image = get_image_from_url(image_url)
        except Exception as img_err:
            return {"status": "error", "message": str(img_err)}

        # Chạy nhận dạng biển số xe
        license_plate, _, _ = detector_service.detect_and_recognize(image, include_images=False)

        if not license_plate:
            return {
                "status": "error",
                "message": "Không phát hiện thấy biển số xe trong hình ảnh."
            }

        # Trả về kết quả khớp DTO C#
        return {
            "status": "success",
            "license_plate": license_plate
        }

    except Exception as e:
        print(f"[API ERROR] {str(e)}")
        return {
            "status": "error",
            "message": f"Lỗi hệ thống: {str(e)}"
        }


@router.post("/predict-web", response_model=PredictWebResponse)
async def predict_web(payload: PredictRequest):
    """
    API hỗ trợ giao diện Web (gửi URL ảnh, trả về thêm base64 để hiển thị trực quan).
    """
    try:
        image_url = payload.image_url
        
        # Tải và giải mã ảnh bằng hàm helper dùng chung
        try:
            image = get_image_from_url(image_url)
        except Exception as img_err:
            return {"status": "error", "message": str(img_err)}

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


@router.post("/predict-file-fast")
async def predict_file_fast(file: UploadFile = File(...)):
    """
    API nhẹ cho Backend .NET: upload file ảnh trực tiếp và chỉ trả biển số,
    không trả ảnh base64 để tránh response nặng và timeout.
    """
    try:
        contents = await file.read()
        image_bytes = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

        if image is None:
            return {"status": "error", "message": "Ảnh tải lên không hợp lệ"}

        license_plate, _, _ = detector_service.detect_and_recognize(
            image,
            include_images=False
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
        contents = await file.read()
        image_bytes = np.frombuffer(contents, dtype=np.uint8)
        image = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"status": "error", "message": "Ảnh tải lên không hợp lệ"}

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
