---
title: Parking License Plate AI
emoji: 🚗
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# Parking Building License Plate Recognition (FastAPI + YOLOv8 + EasyOCR)

Dịch vụ AI độc lập chịu trách nhiệm phát hiện và nhận diện biển số xe Việt Nam. Backend .NET gửi trực tiếp file ảnh sang API, dịch vụ thực hiện phân tích và trả về biển số đã được chuẩn hóa. Luồng nhận diện chính không yêu cầu upload ảnh lên Cloudinary.

---

## 📂 Cấu Trúc Dự Án Chuyên Nghiệp

Dự án được thiết kế theo cấu trúc phân lớp rõ ràng (Layered Architecture):

```text
ParkingBuildingLicensePlateAI/
│
├── app/
│   ├── api/               # Chứa các router định nghĩa API Endpoints
│   │   └── predict.py     # Route nhận diện biển số (dành cho .NET & Web UI)
│   │
│   ├── core/              # Cấu hình hệ thống & biến môi trường
│   │   └── config.py
│   │
│   ├── services/          # Chứa Business Logic tích hợp mô hình AI
│   │   └── detector.py    # LicensePlateDetector Service (YOLO + EasyOCR)
│   │
│   ├── templates/         # Giao diện web UI
│   │   └── index.html     # Dashboard tối giản, Glassmorphism tối
│   │
│   ├── utils/             # Các hàm Helper tiện ích
│   │   └── helpers.py     # Hàm hậu xử lý chuẩn hóa biển số Việt Nam
│   │
│   └── main.py            # Entry point chính để chạy ứng dụng FastAPI
│
├── models/                # Lưu trữ file mô hình YOLO (.pt)
│   └── README.md
│
├── .gitignore             # Các tệp/thư mục Git bỏ qua (venv, cache, model)
├── requirements.txt       # Danh sách các thư viện cần thiết
└── README.md              # Tài liệu hướng dẫn sử dụng
```

---

## ⚡ Hướng Dẫn Vận Hành Dự Án Trong PyCharm

### Bước 1: Khởi tạo Project trong PyCharm
1. Mở **PyCharm** -> **Open** -> Chọn thư mục `ParkingBuildingLicensePlateAI`.
2. Tạo môi trường ảo **`.venv`** khi PyCharm hỏi, hoặc cấu hình trong *Settings -> Project Interpreter*.

### Bước 2: Cài đặt các thư viện phụ thuộc
Mở Terminal của PyCharm tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### Bước 3: Đặt mô hình YOLO vào thư mục `models/`
- Copy file mô hình nhận diện biển số của bạn đặt vào thư mục `models/` và đổi tên thành `best.pt`.
- Nếu chưa có, hệ thống sử dụng model fallback `keremberke/yolov8n-license-plate-detector`.

### Bước 4: Chạy dự án
- Cách 1: Click chuột phải vào file `app/main.py` -> **Run 'main'**.
- Cách 2: Chạy lệnh sau tại terminal:
  ```bash
  python app/main.py
  ```

---

## 🌐 Các API Endpoint Chính

### 1. Nhận diện từ file ảnh — endpoint chính cho Backend .NET

- **Method:** `POST`
- **Path:** `/predict-file-fast`
- **Content-Type:** `multipart/form-data`
- **Tên field chứa ảnh:** `file`
- **Kết quả:** Chỉ trả biển số, không trả ảnh Base64 để response nhẹ hơn.

Ví dụ request bằng cURL:

```bash
curl -X POST "http://127.0.0.1:8000/predict-file-fast" \
  -F "file=@vehicle.jpg"
```

Response thành công:

```json
{
  "status": "success",
  "license_plate": "29A112345"
}
```

Response khi không phát hiện được biển số:

```json
{
  "status": "error",
  "message": "Không phát hiện thấy biển số xe"
}
```

### 2. Quy trình xử lý ảnh

```text
Frontend chọn/chụp ảnh
    → Backend .NET nhận file ảnh
    → Backend gửi trực tiếp file đến POST /predict-file-fast
    → FastAPI đọc byte và OpenCV giải mã ảnh
    → YOLO phát hiện vùng biển số có độ tin cậy cao nhất
    → Cắt, sửa nghiêng, tăng tương phản và nhị phân hóa vùng biển số
    → EasyOCR đọc ký tự
    → Hậu xử lý và chuẩn hóa theo cấu trúc biển số Việt Nam
    → FastAPI trả về trường license_plate
    → Backend .NET đọc kết quả
```

Backend không cần upload ảnh lên Cloudinary trước khi gọi dịch vụ AI. Nếu hệ thống cần lưu ảnh để phục vụ nghiệp vụ check-in/check-out, Backend có thể thực hiện việc lưu trữ đó độc lập với bước nhận diện.

### 3. Web Dashboard kiểm thử trực tiếp

Truy cập `http://127.0.0.1:8000` trên trình duyệt để tải ảnh lên và xem kết quả nhận diện trực quan, bao gồm ảnh đã vẽ bounding box và vùng biển số được cắt.
