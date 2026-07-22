# Quy trình nhận diện và xác nhận biển số xe

Tài liệu này mô tả luồng hiện tại từ Frontend qua Backend .NET đến dịch vụ Python AI. Backend gửi trực tiếp file ảnh sang AI; Cloudinary không tham gia bước nhận diện.

## 1. Sơ đồ tuần tự

```mermaid
sequenceDiagram
    autonumber
    actor Staff as Nhân viên
    participant FE as Frontend
    participant BE as Backend .NET
    participant AI as Python FastAPI
    participant Model as YOLO + EasyOCR
    participant DB as Database

    Staff->>FE: Chụp hoặc chọn ảnh phương tiện
    FE->>BE: Gửi file ảnh
    BE->>AI: POST /predict-file-fast (multipart/form-data, field file)
    AI->>AI: Đọc byte và giải mã ảnh bằng OpenCV
    AI->>Model: Phát hiện vùng biển và nhận dạng ký tự
    Model-->>AI: Kết quả OCR
    AI->>AI: Chuẩn hóa biển số Việt Nam
    AI-->>BE: {status, license_plate}
    BE-->>FE: Trả biển số dự đoán
    FE->>Staff: Hiển thị để kiểm tra và chỉnh sửa nếu cần
    Staff->>FE: Xác nhận biển số
    FE->>BE: Gửi dữ liệu check-in/check-out đã xác nhận
    BE->>DB: Lưu biển số và thông tin phiên gửi xe
```

## 2. Hợp đồng API giữa Backend và Python AI

### Request

```http
POST /predict-file-fast
Content-Type: multipart/form-data
```

Tên field chứa ảnh bắt buộc là `file`.

```bash
curl -X POST "http://127.0.0.1:8000/predict-file-fast" \
  -F "file=@vehicle.jpg"
```

### Response thành công

```json
{
  "status": "success",
  "license_plate": "59X312345"
}
```

### Response thất bại

```json
{
  "status": "error",
  "message": "Không phát hiện thấy biển số xe"
}
```

Backend phải đọc trường `license_plate`. Dịch vụ AI không trả trường `predicted_plate`.

## 3. Các bước xử lý bên trong AI

1. FastAPI nhận `UploadFile` và đọc dữ liệu ảnh.
2. OpenCV giải mã byte thành ma trận ảnh BGR.
3. Ảnh lớn được thu nhỏ để giảm thời gian suy luận.
4. YOLO phát hiện các vùng biển số và chọn bounding box có confidence cao nhất.
5. Vùng biển số được mở rộng nhẹ, cắt khỏi ảnh, sửa nghiêng và tăng tương phản.
6. EasyOCR đọc các ký tự trong tập `0-9` và `A-Z`.
7. Hậu xử lý sắp xếp một dòng/hai dòng, lọc confidence thấp và sửa lỗi OCR thường gặp.
8. Ứng viên phù hợp nhất với cấu trúc biển số Việt Nam được trả cho Backend.

## 4. Phân chia trách nhiệm

- Frontend chụp/chọn ảnh và cho nhân viên kiểm tra kết quả.
- Backend điều phối nghiệp vụ, gửi file sang AI và lưu dữ liệu đã xác nhận.
- Python AI chỉ nhận diện ảnh và trả chuỗi biển số.
- Nếu cần lưu ảnh, Backend xử lý lưu trữ độc lập sau hoặc song song với nhận diện; AI không tải ảnh từ dịch vụ lưu trữ.
