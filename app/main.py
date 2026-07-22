import os
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from app.api.predict import router as predict_router
from app.core.config import settings

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="Dịch vụ AI phát hiện và nhận diện biển số xe Việt Nam sử dụng YOLOv8 và EasyOCR.",
    version="1.0.0"
)

# Đăng ký router nhận diện biển số
app.include_router(predict_router, prefix="")


@app.get("/health")
async def health_check():
    """Lightweight endpoint used by the hosting platform and Backend."""
    return {"status": "ok", "service": settings.PROJECT_NAME}

# Cấu hình Templates cho giao diện Dashboard HTML
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
templates = Jinja2Templates(directory=templates_dir)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """
    Route chính trả về giao diện Dashboard để người dùng test trực tiếp trên trình duyệt.
    """
    # Đã sửa lỗi tương thích Starlette/FastAPI phiên bản mới
    return templates.TemplateResponse(request=request, name="index.html")

if __name__ == "__main__":
    print("=" * 80)
    print(" BẮT ĐẦU KHỞI CHẠY DỊCH VỤ AI NHẬN DIỆN BIỂN SỐ XE VIỆT NAM (FASTAPI)")
    print(" Địa chỉ Web UI Dashboard: http://127.0.0.1:8000")
    print(" API Endpoint cho Backend .NET: [POST] http://127.0.0.1:8000/predict-file-fast")
    print("=" * 80)
    
    # Chạy uvicorn server cục bộ
    # Lưu ý: Chạy ở dạng uvicorn app.main:app để hot-reload hoạt động đúng
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port)
