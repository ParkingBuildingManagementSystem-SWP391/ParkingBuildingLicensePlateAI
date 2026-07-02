import os

class Settings:
    PROJECT_NAME: str = "Parking Building License Plate AI"
    
    # Thiết lập đường dẫn tương đối tới các thư mục
    BASE_DIR: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    # Cấu hình đường dẫn Model YOLO
    MODEL_DIR: str = os.path.join(BASE_DIR, "models")
    YOLO_MODEL_PATH: str = os.path.join(MODEL_DIR, "best.pt")
    
    # Sử dụng mô hình nhận diện biển số xe chuyên dụng của Keremberke từ Hugging Face làm Fallback
    # Nhờ đó người dùng test được ngay lập tức mà không cần tự download best.pt
    FALLBACK_MODEL_PATH: str = "keremberke/yolov8n-license-plate-detector"
    
    # Cấu hình ngôn ngữ OCR
    OCR_LANGUAGES: list = ["en"]
    OCR_GPU: bool = False  # Đổi thành True nếu hệ thống có GPU NVIDIA đã cài CUDA
    OCR_FAST_MODE: bool = True
    OCR_DECODER: str = "greedy"
    OCR_BEAM_WIDTH: int = 3
    OCR_TRY_ALTERNATE_MODE: bool = False
    OCR_TRY_SPLIT_FALLBACK: bool = False
    MAX_DETECT_IMAGE_SIDE: int = 960
    WARMUP_ON_STARTUP: bool = True

settings = Settings()

# Đảm bảo thư mục models luôn tồn tại
os.makedirs(settings.MODEL_DIR, exist_ok=True)
