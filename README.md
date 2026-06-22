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

> Pipeline huấn luyện OCR chuyên biệt đã được bổ sung. Xem [TRAINING.md](TRAINING.md) để tạo dataset, pretrain bằng dữ liệu tổng hợp và fine-tune bằng ảnh biển số thật.

Dịch vụ AI độc lập chịu trách nhiệm phát hiện và nhận diện biển số xe Việt Nam. Tiếp nhận dữ liệu từ Backend .NET gửi sang, thực hiện phân tích và trả về thông tin biển số xe sạch (đã chuẩn hóa).

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

### Bước 3: Cài đặt các thư viện phụ thuộc
Mở Terminal của PyCharm tại thư mục dự án và chạy:
```bash
pip install -r requirements.txt
```

### Bước 4: Đặt mô hình YOLO vào thư mục `models/`
- Copy file mô hình nhận diện biển số của bạn đặt vào thư mục `models/` và đổi tên thành `best.pt`.
- Nếu chưa có, hệ thống sẽ tự động tải file `yolov8n.pt` về thư mục này để chạy demo luồng API.

### Bước 5: Chạy dự án
- Cách 1: Click chuột phải vào file `app/main.py` -> **Run 'main'**.
- Cách 2: Chạy lệnh sau tại terminal:
  ```bash
  python app/main.py
  ```

---

## 🌐 Các API Endpoint Chính

1. **API Nhận diện biển số xe (Cho Backend .NET):**
   - **Method:** `POST`
   - **Path:** `/predict`
   - **Payload:** `{"image_url": "url_anh_cloudinary"}`
   - **Response:** `{"status": "success", "license_plate": "29A112345"}`

2. **Web Dashboard kiểm thử trực tiếp:**
   - Truy cập: `http://127.0.0.1:8000` trên trình duyệt để sử dụng giao diện tải ảnh lên và xem kết quả nhận diện trực quan (vẽ box + crop ảnh).
